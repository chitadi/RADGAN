# src/models/dcctn.py
if __name__ == "__main__":
    from base_model import BaseModel
else:
    from .base_model import BaseModel

import math
import torch
import torch.nn.functional as F
from auraloss.time import SISDRLoss
from auraloss.freq import STFTLoss, MultiResolutionSTFTLoss, RandomResolutionSTFTLoss, MelSTFTLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import WandbLogger
from typing import Any, Dict, Optional

class DCCTN(BaseModel):
    """
    Lightning wrapper that keeps RASE’s BaseModel metrics/steps
    but swaps in the DCCTN backbone plus the SISDR + STFT loss recipe.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        betas=(0.5, 0.999),
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        stft_loss_config: Optional[Dict[str, Any]] = None,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        from .dcctn_core import DCCTN as DCCTNBackbone  # imported lazily to avoid circulars

        backbone_kwargs = backbone_kwargs or {}
        self.backbone = DCCTNBackbone(**backbone_kwargs)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor

        stft_cfg = stft_loss_config.copy() if stft_loss_config else {}
        self._stft_weight = stft_cfg.pop("weight", 25.0)
        self.si_sdr_loss = SISDRLoss()
        # self.stft_loss = STFTLoss(**stft_cfg) if stft_cfg else STFTLoss()
        loss_kind = stft_cfg.pop("kind", "single")
        if loss_kind == "multi":
            self.stft_loss = MultiResolutionSTFTLoss(**stft_cfg)
        elif loss_kind == "random":
            self.stft_loss = RandomResolutionSTFTLoss(**stft_cfg)
        elif loss_kind == "mel":
            self.stft_loss = MelSTFTLoss(**stft_cfg)
        else:
            self.stft_loss = STFTLoss(**stft_cfg)

        
        self._diag_cfg = {
          "fft_size": stft_cfg.get("fft_size", 256),
          "hop_size": stft_cfg.get("hop_size", 128),
          "win_length": stft_cfg.get("win_length", stft_cfg.get("fft_size", 256)),
        }
        self.register_buffer(
            "diag_window",
            torch.hann_window(self._diag_cfg["win_length"], periodic=False),
            persistent=False,
        )
        self._band_edges = [(0, 1_000), (1_000, 3_000), (3_000, 4_000)]
        self._diag_eps = 1e-7


        self.loss_function = self._combined_loss
        self.save_hyperparameters(ignore=["backbone"])

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        BaseModel.common_step feeds a 2D tensor (batch, samples).
        DCCTN expects (batch, channels, samples), so we add/remove the singleton dim.
        """
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)
        enhanced = self.backbone(noisy)
        if enhanced.dim() == 3:
            enhanced = enhanced.squeeze(1)
        return enhanced

    def _combined_loss(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        clean_for_stft = clean.unsqueeze(1) if clean.dim() == 2 else clean
        enhanced_for_stft = enhanced.unsqueeze(1) if enhanced.dim() == 2 else enhanced
        return (
            self.si_sdr_loss(enhanced, clean)
            + self._stft_weight * self.stft_loss(enhanced_for_stft, clean_for_stft)
        )
    
    def _log_diagnostics(self, clean, enhanced, batch, mode: str, batch_idx: int) -> None:
        cfg = self._diag_cfg
        window = self.diag_window.to(clean.device)

        clean_spec = torch.stft(clean, cfg["fft_size"], cfg["hop_size"],
                                cfg["win_length"], window=window,
                                return_complex=True, center=True)
        enh_spec = torch.stft(enhanced, cfg["fft_size"], cfg["hop_size"],
                              cfg["win_length"], window=window,
                              return_complex=True, center=True)

        mag_clean = clean_spec.abs().clamp_min(self._diag_eps)
        mag_enh = enh_spec.abs().clamp_min(self._diag_eps)
        residual_pow = (enh_spec - clean_spec).abs().pow(2)

        delta_mag = (mag_enh - mag_clean).flatten(1)
        stft_sc = torch.linalg.norm(delta_mag, dim=1) / (
            torch.linalg.norm(mag_clean.flatten(1), dim=1).clamp_min(self._diag_eps)
        )
        stft_logmag = F.l1_loss(torch.log(mag_enh), torch.log(mag_clean),
                                reduction="none").mean(dim=(1, 2))
        sisdr = -self.si_sdr_loss(enhanced, clean).detach()

        clean_flat = clean.reshape(clean.size(0), -1)
        enh_flat = enhanced.reshape_as(clean_flat)
        diff_flat = enh_flat - clean_flat
        signal_pow = clean_flat.pow(2).mean(dim=1)
        noise_pow = diff_flat.pow(2).mean(dim=1)
        snr = 10.0 * torch.log10(signal_pow.clamp_min(self._diag_eps) /
                                 noise_pow.clamp_min(self._diag_eps))
        sdsdr_num = clean_flat.pow(2).sum(dim=1)
        sdsdr_den = diff_flat.pow(2).sum(dim=1)
        sdsdr = 10.0 * torch.log10(sdsdr_num.clamp_min(self._diag_eps) /
                                   sdsdr_den.clamp_min(self._diag_eps))
        logcosh = (F.softplus(2.0 * diff_flat) - diff_flat - math.log(2.0)).mean(dim=1)

        scale_clean = batch["scale_clean"]
        if not isinstance(scale_clean, torch.Tensor):
            scale_clean = torch.tensor(scale_clean, device=clean.device, dtype=clean.dtype)
        else:
            scale_clean = scale_clean.to(clean.device, clean.dtype)
        clean_dn = clean * scale_clean.unsqueeze(-1)
        enh_dn = enhanced * scale_clean.unsqueeze(-1)
        mfcc_cos = torch.tensor(
            [self.csmfcc_fn(c.cpu().numpy(), e.cpu().numpy())
             for c, e in zip(clean_dn.detach(), enh_dn.detach())],
            device=clean.device,
            dtype=clean.dtype,
        )

        batch_sz = clean.size(0)
        self.log(f"{mode}/sisdr", sisdr.mean(), prog_bar=(mode == "val"), batch_size=batch_sz)
        self.log(f"{mode}/stft_sc", stft_sc.mean(), batch_size=batch_sz)
        self.log(f"{mode}/stft_logmag", stft_logmag.mean(), batch_size=batch_sz)
        self.log(f"{mode}/mfcc_cos", mfcc_cos.mean(), batch_size=batch_sz)
        self.log(f"{mode}/snr", snr.mean(), batch_size=batch_sz)
        self.log(f"{mode}/sdsdr", sdsdr.mean(), batch_size=batch_sz)
        self.log(f"{mode}/logcosh", logcosh.mean(), batch_size=batch_sz)

        fs = batch.get("fs", 8000.0)
        if isinstance(fs, torch.Tensor):
            fs = float(fs.flatten()[0].item())
        elif isinstance(fs, (list, tuple)):
            fs = float(fs[0])

        freqs = torch.linspace(0.0, fs / 2.0, steps=residual_pow.size(1), device=residual_pow.device)
        band_curves = []
        for low, high in self._band_edges:
            high = min(high, fs / 2.0)
            mask = (freqs >= low) & (freqs < high)
            if mask.any():
                band_curves.append(residual_pow[:, mask].mean(dim=1))
            else:
                band_curves.append(torch.zeros(batch_sz, residual_pow.size(2), device=residual_pow.device))
        band_curves = torch.stack(band_curves, dim=1)

        band_means = band_curves.mean(dim=(0, 2))
        for (low, high), mean_val in zip(self._band_edges, band_means):
            high = min(high, fs / 2.0)
            tag = f"{int(low // 1000)}-{int(high // 1000)}k"
            self.log(f"{mode}/band_residual_{tag}", mean_val, batch_size=batch_sz)

        time_profile = band_curves.sum(dim=1)   # (B, frames)
        onset_peak = time_profile.max(dim=1).values.mean()
        sorted_tp = torch.sort(time_profile, dim=1).values
        top_idx = max(0, min(int(0.95 * (sorted_tp.size(-1) - 1)), sorted_tp.size(-1) - 1))
        top_idx = torch.full(
            (sorted_tp.size(0), 1),
            top_idx,
            device=sorted_tp.device,
            dtype=torch.long,
        )
        onset_p95 = sorted_tp.gather(1, top_idx).squeeze(-1).mean()

        self.log(f"{mode}/onset_error_peak", onset_peak, batch_size=batch_sz)
        self.log(f"{mode}/onset_error_p95", onset_p95, batch_size=batch_sz)

        wandb_logger = None
        if isinstance(self.logger, WandbLogger):
            wandb_logger = self.logger
        else:
            for lg in getattr(self.logger, "loggers", []):
                if isinstance(lg, WandbLogger):
                    wandb_logger = lg
                    break

        if wandb_logger is not None and batch_idx == 0:
            import wandb
            heatmap = 10.0 * torch.log10(band_curves.mean(dim=0).clamp_min(self._diag_eps)).cpu().numpy()
            wandb_logger.experiment.log(
                {f"{mode}/band_error_heatmap": wandb.Image(heatmap, caption="Residual band error (dB)")},
                commit=False,
            )
    
    def validation_step(self, batch, batch_idx):
        enhanced, clean, task = self.common_step(batch, batch_idx, mode="val")
        loss = self.loss_function(enhanced, clean)
        self.log("val/loss", loss, logger=True)
        with torch.no_grad():
            self._log_diagnostics(clean, enhanced, batch, mode="val", batch_idx=batch_idx)
        if self.heavy_eval:
            self.val_outputs.append(enhanced.detach().cpu().numpy())
            self.val_targets.append(clean.detach().cpu().numpy())
            self.val_tasks.append(task)
            scale_clean = batch["scale_clean"]
            if isinstance(scale_clean, torch.Tensor):
                scale_clean = scale_clean.detach().cpu().numpy()
            else:
                scale_clean = np.asarray(scale_clean)
            self.val_clean_scales.append(scale_clean)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
            # verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True,
            },
        }

dcctn = DCCTN  # alias for easier imports
