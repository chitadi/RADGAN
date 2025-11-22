import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from .base_model import BaseModel
from .ap_bwe_core.ap_bwe_model import (
    APNet_BWE_Model,
    MultiPeriodDiscriminator,
    MultiResolutionAmplitudeDiscriminator,
    MultiResolutionPhaseDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
    phase_losses,
)
from .ap_bwe_core.dataset import amp_pha_stft, amp_pha_istft
from .ap_bwe_core.env import AttrDict
from bwe_example import preprocess_train
from bwe_example_for_inference import preprocess_val


APBWE_CONFIG = AttrDict(
    dict(
        num_gpus=0,
        batch_size=16,
        learning_rate=2.0e-4,
        adam_b1=0.8,
        adam_b2=0.99,
        lr_decay=0.999,
        seed=1234,

        ConvNeXt_channels=512,
        ConvNeXt_layers=8,

        segment_size=8000,
        n_fft=1024,
        hop_size=80,
        win_size=320,

        hr_sampling_rate=16000,
        lr_sampling_rate=2000,

        num_workers=4,
        dist_config=dict(
            dist_backend="nccl",
            dist_url="tcp://localhost:54321",
            world_size=1,
        ),
    )
)


class ap_bwe(BaseModel):
    """
    Simple Lightning wrapper around APNet_BWE_Model that plugs into
    the existing BaseModel training/validation/test loops.
    """

    def __init__(
        self,
        learning_rate: float = 2.0e-4,
        n_fft: int = 1024,
        hop_size: int = 80,
        win_size: int = 320,
        convnext_channels: int = 512,
        convnext_layers: int = 8,
    ):
        super().__init__()

        # Build a lightweight h-config for APNet_BWE_Model
        cfg = dict(APBWE_CONFIG)
        cfg["learning_rate"] = learning_rate
        cfg["n_fft"] = n_fft
        cfg["hop_size"] = hop_size
        cfg["win_size"] = win_size
        cfg["ConvNeXt_channels"] = convnext_channels
        cfg["ConvNeXt_layers"] = convnext_layers
        self.h = AttrDict(cfg)

        # Generator backbone from ap_bwe_core
        self.generator = APNet_BWE_Model(self.h)
        self.mpd = MultiPeriodDiscriminator()
        self.mrad = MultiResolutionAmplitudeDiscriminator()
        self.mrpd = MultiResolutionPhaseDiscriminator()

        # Loss weights from ap_bwe_train.py
        # self.w_mag = 45.0
        # self.w_pha = 100.0
        # self.w_com = 90.0
        # self.w_stft = 90.0
        # self.w_adv_scale = 0.1
        self.w_mag = 20.0
        self.w_pha = 50.0
        self.w_com = 45.0
        self.w_stft = 45.0
        self.w_adv_scale = 0.05


        self.learning_rate = learning_rate
        # Non‑adversarial mag/phase/complex/STFT loss for val/test
        self.loss_function = self._mag_phase_complex_loss

        # GAN: use manual optimization like HiFiGAN
        self.automatic_optimization = False

        # Keep checkpoint size reasonable
        self.save_hyperparameters(ignore=["generator", "mpd", "mrad", "mrpd"])
    
    def _mag_phase_complex_loss(self, clean: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Non‑adversarial validation/test loss:
        L2(mag) + anti‑wrap phase + L2(complex) + STFT‑consistency.
        """
        mag_wb, pha_wb, com_wb = amp_pha_stft(
            clean,
            n_fft=self.h.n_fft,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            center=True,
        )
        mag_enh, pha_enh, com_enh = amp_pha_stft(
            enhanced,
            n_fft=self.h.n_fft,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            center=True,
        )

        # Consistency STFT of enhanced audio
        audio_enh = amp_pha_istft(
            mag_enh,
            pha_enh,
            n_fft=self.h.n_fft,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            center=True,
        )
        _, _, com_enh_hat = amp_pha_stft(
            audio_enh,
            n_fft=self.h.n_fft,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            center=True,
        )

        loss_mag = F.mse_loss(mag_wb, mag_enh) * self.w_mag
        ip, gd, iaf = phase_losses(pha_wb, pha_enh)
        loss_pha = (ip + gd + iaf) * self.w_pha
        loss_com = F.mse_loss(com_wb, com_enh) * self.w_com
        loss_stft = F.mse_loss(com_enh, com_enh_hat) * self.w_stft

        return loss_mag + loss_pha + loss_com + loss_stft



    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        noisy: (B, T) or (T,) mono waveform from DataModule.
        Returns enhanced waveform with the same length.
        """
        if noisy.dim() == 1:
            noisy = noisy.unsqueeze(0)  # (1, T)
        
        # --- BWE preprocessing for inference / val / test ---
        noisy_bwe_list = []
        for x in noisy:
            x_np = x.detach().cpu().numpy()               # 1D numpy
            # If all data are e.g. 8000 Hz, just pass that here.
            x_bwe = preprocess_val(x_np, 8000)       # direct call, as in bwe_example_for_inference.py
            # x_bwe is a 1D torch.Tensor; move to device and keep batch dim
            noisy_bwe_list.append(x_bwe.to(self.device).unsqueeze(0))
        noisy_bwe = torch.cat(noisy_bwe_list, dim=0)      # (B, T)

        # STFT -> APNet_BWE_Model -> ISTFT
        log_mag_nb, pha_nb, _ = amp_pha_stft(
            noisy_bwe,
            n_fft=self.h.n_fft,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            center=True,
        )

        log_mag_wb, pha_wb, _ = self.generator(log_mag_nb, pha_nb)

        enhanced = amp_pha_istft(
            log_mag_wb,
            pha_wb,
            n_fft=self.h.n_fft,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            center=True,
        )

        # Ensure output length matches input
        target_len = noisy.size(-1)
        cur_len = enhanced.size(-1)
        if cur_len > target_len:
            enhanced = enhanced[..., :target_len]
        elif cur_len < target_len:
            enhanced = F.pad(enhanced, (0, target_len - cur_len))

        return enhanced
    
    def training_step(self, batch, batch_idx):
        """
        AP‑BWE adversarial training step:
        - D: MPD + MR amplitude/phase discriminators
        - G: mag/phase/complex/STFT + adv + feature matching
        """
        opt_d, opt_g = self.optimizers()

        recorded = batch["recorded"].to(self.device).to(torch.float32)  # (B, T)
        clean = batch["clean"].to(self.device).to(torch.float32)        # (B, T)
        fs = batch.get("fs", 8000.0)
        
        # --- BWE preprocessing for training (loop over batch) ---
        clean_hr_list = []
        recorded_bwe_list = []
        for i in range(recorded.size(0)):
            rec_np = recorded[i].detach().cpu().numpy()
            clean_np = clean[i].detach().cpu().numpy()

            if fs is None:
                orig_fs = 8000.0  # or your known dataset rate
            else:
                if isinstance(fs, torch.Tensor):
                    orig_fs = float(fs[i].item())
                else:
                    orig_fs = float(fs[i])

            # Direct call into bwe_example.dataset
            audio_hr, audio_lr = preprocess_train(rec_np, clean_np, orig_fs)

            clean_hr_list.append(audio_hr.to(self.device).unsqueeze(0))
            recorded_bwe_list.append(audio_lr.to(self.device).unsqueeze(0))

        clean_hr = torch.cat(clean_hr_list, dim=0)        # (B, T_hr)
        recorded_bwe = torch.cat(recorded_bwe_list, dim=0)  # (B, T_hr)

        
        # STFTs for target WB and NB input
        mag_wb, pha_wb, com_wb = amp_pha_stft(
            clean_hr,
            n_fft=self.h.n_fft,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            center=True,
        )
        mag_nb, pha_nb, com_nb = amp_pha_stft(
            recorded_bwe,
            n_fft=self.h.n_fft,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            center=True,
        )

        # ---------- Discriminator update ----------
        with torch.no_grad():
            mag_wb_g, pha_wb_g, com_wb_g = self.generator(mag_nb, pha_nb)
            audio_wb_g = amp_pha_istft(
                mag_wb_g,
                pha_wb_g,
                n_fft=self.h.n_fft,
                hop_size=self.h.hop_size,
                win_size=self.h.win_size,
                center=True,
            )

        audio_wb = clean
        audio_wb, audio_wb_g = audio_wb.unsqueeze(1), audio_wb_g.unsqueeze(1)  # (B, 1, T)

        opt_d.zero_grad()

        # MPD
        audio_df_r, audio_df_g, _, _ = self.mpd(audio_wb, audio_wb_g.detach())
        loss_disc_f, _, _ = discriminator_loss(audio_df_r, audio_df_g)

        # MR amplitude disc
        spec_da_r, spec_da_g, _, _ = self.mrad(audio_wb, audio_wb_g.detach())
        loss_disc_a, _, _ = discriminator_loss(spec_da_r, spec_da_g)

        # MR phase disc
        spec_dp_r, spec_dp_g, _, _ = self.mrpd(audio_wb, audio_wb_g.detach())
        loss_disc_p, _, _ = discriminator_loss(spec_dp_r, spec_dp_g)

        loss_disc_all = (loss_disc_a + loss_disc_p) * self.w_adv_scale + loss_disc_f
        self.manual_backward(loss_disc_all)
        opt_d.step()

        # ---------- Generator update ----------
        opt_g.zero_grad()

        mag_wb_g, pha_wb_g, com_wb_g = self.generator(mag_nb, pha_nb)
        audio_wb_g = amp_pha_istft(
            mag_wb_g,
            pha_wb_g,
            n_fft=self.h.n_fft,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            center=True,
        )
        mag_wb_g_hat, pha_wb_g_hat, com_wb_g_hat = amp_pha_stft(
            audio_wb_g,
            n_fft=self.h.n_fft,
            hop_size=self.h.hop_size,
            win_size=self.h.win_size,
            center=True,
        )

        # Core reconstruction losses
        loss_mag = F.mse_loss(mag_wb, mag_wb_g) * self.w_mag
        ip, gd, iaf = phase_losses(pha_wb, pha_wb_g)
        loss_pha = (ip + gd + iaf) * self.w_pha
        loss_com = F.mse_loss(com_wb, com_wb_g) * self.w_com
        loss_stft = F.mse_loss(com_wb_g, com_wb_g_hat) * self.w_stft

        # Adversarial + feature matching
        audio_wb = clean.unsqueeze(1)
        audio_wb_g = audio_wb_g.unsqueeze(1)

        audio_df_r, audio_df_g, fmap_f_r, fmap_f_g = self.mpd(audio_wb, audio_wb_g)
        spec_da_r, spec_da_g, fmap_a_r, fmap_a_g = self.mrad(audio_wb, audio_wb_g)
        spec_dp_r, spec_dp_g, fmap_p_r, fmap_p_g = self.mrpd(audio_wb, audio_wb_g)

        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_a = feature_loss(fmap_a_r, fmap_a_g)
        loss_fm_p = feature_loss(fmap_p_r, fmap_p_g)

        loss_gen_f, _ = generator_loss(audio_df_g)
        loss_gen_a, _ = generator_loss(spec_da_g)
        loss_gen_p, _ = generator_loss(spec_dp_g)

        loss_gen = (loss_gen_a + loss_gen_p) * self.w_adv_scale + loss_gen_f
        loss_fm = (loss_fm_a + loss_fm_p) * self.w_adv_scale + loss_fm_f

        loss_gen_all = loss_mag + loss_pha + loss_com + loss_stft + loss_gen + loss_fm

        self.manual_backward(loss_gen_all)
        opt_g.step()

        bsz = clean.size(0)
        self.log("train/loss_disc", loss_disc_all, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train/loss", loss_gen_all, on_step=True, on_epoch=True, prog_bar=True, batch_size=bsz)

        return loss_gen_all
    
    def validation_step(self, batch, batch_idx):
        # Same core as BaseModel.validation_step
        enhanced, clean, task = self.common_step(batch, batch_idx, mode="val")
        loss = self.loss_function(clean, enhanced)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # --- PESQ / ESTOI / CSMFCC composite metric (DCCTN‑style) ---
        pesq_vals, estoi_vals, mfcc_vals = [], [], []
        for c_t, e_t in zip(clean, enhanced):
            c_np = c_t.detach().cpu().numpy()
            e_np = e_t.detach().cpu().numpy()
            try:
                pesq_vals.append(self.pesq_fn(c_np, e_np))
            except Exception:
                pesq_vals.append(self.pesq_fn.min())
            try:
                estoi_vals.append(self.estoi_fn(c_np, e_np))
            except Exception:
                estoi_vals.append(self.estoi_fn.min())
            try:
                mfcc_vals.append(self.csmfcc_fn(c_np, e_np))
            except Exception:
                mfcc_vals.append(self.csmfcc_fn.min())

        if pesq_vals:
            pesq_mean  = torch.tensor(pesq_vals,  device=self.device).mean()
            estoi_mean = torch.tensor(estoi_vals, device=self.device).mean()
            mfcc_mean  = torch.tensor(mfcc_vals,  device=self.device).mean()
            pesq_estoi_mfcc = (pesq_mean + estoi_mean + mfcc_mean) / 3.0
            self.log(
                "val/pesq_estoi_mfcc",
                pesq_estoi_mfcc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=clean.size(0),
            )

        # Preserve heavy_eval behaviour for final eval
        if self.heavy_eval:
            self.val_outputs.append(enhanced.detach().cpu().numpy())
            self.val_targets.append(clean.detach().cpu().numpy())
            self.val_tasks.append(task)

        return loss

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.h.adam_b1, self.h.adam_b2),
        )
        opt_d = torch.optim.AdamW(
            itertools.chain(
                self.mpd.parameters(),
                self.mrad.parameters(),
                self.mrpd.parameters(),
            ),
            lr=self.learning_rate,
            betas=(self.h.adam_b1, self.h.adam_b2),
        )
        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.h.lr_decay)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.h.lr_decay)
        return (
            [opt_d, opt_g],
            [
                {"scheduler": sch_d, "interval": "epoch", "name": "disc_lr"},
                {"scheduler": sch_g, "interval": "epoch", "name": "gen_lr"},
            ],
        )

ap_bwe = ap_bwe