# src/models/dcctn.py
if __name__ == "__main__":
    from base_model import BaseModel
else:
    from .base_model import BaseModel

import torch
from auraloss.time import SISDRLoss
from auraloss.freq import STFTLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        self.stft_loss = STFTLoss(**stft_cfg) if stft_cfg else STFTLoss()

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

    def _combined_loss(self, clean: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        clean_for_stft = clean.unsqueeze(1) if clean.dim() == 2 else clean
        enhanced_for_stft = enhanced.unsqueeze(1) if enhanced.dim() == 2 else enhanced
        return (
            self.si_sdr_loss(clean, enhanced)
            + self._stft_weight * self.stft_loss(clean_for_stft, enhanced_for_stft)
        )

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