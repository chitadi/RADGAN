# src/models/wnd_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

def _make_depthwise_conv(in_channels: int, kernel_size: int = 7, padding: int = 3) -> nn.Conv1d:
    # Depth-wise 1D convolution: groups=in_channels ensures channel-wise filtering
    return nn.Conv1d(
        in_channels,
        in_channels,
        kernel_size=kernel_size,
        padding=padding,
        # how was padding decided?
        groups=in_channels,
        bias=False,
    )


class ConvNeXtBlock1D(nn.Module):
    """
    ConvNeXt-style block for 1D signals:
        depth-wise conv -> LayerNorm (token-wise) -> point-wise MLP -> residual add.
    """

    def __init__(self, channels: int, mlp_ratio: int = 4):
        super().__init__()
        # how was mlp_ratio decided?
        hidden = channels * mlp_ratio
        self.depthwise = _make_depthwise_conv(channels)
        self.layer_norm = nn.LayerNorm(channels)
        self.pointwise = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.depthwise(x)
        x = x.transpose(1, 2)            # (B, C, T) -> (B, T, C) for LayerNorm
        x = self.layer_norm(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)            # back to (B, C, T)

        return x + residual


class DoubleConvBlock(nn.Module):
    """
    1x1 UNet-style convolution block:
        (Conv -> BN -> ReLU) x 2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualConvBlock(nn.Module):
    """
    Residual refinement block for decoder:
        Conv -> BN -> ReLU -> Conv -> BN + skip -> ReLU
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + x)

class EncoderBlock(nn.Module):
    """
    One encoder stage:
        ConvNeXt block -> double conv -> optional maxpool (handled outside for last stage).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.convnext = ConvNeXtBlock1D(in_channels)
        self.double_conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convnext(x)
        x = self.double_conv(x)
        return x


class DecoderBlock(nn.Module):
    """
    One decoder stage:
        ConvTranspose1d (upsample) -> double conv -> residual block -> skip fusion.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.activation = nn.ReLU(inplace=True)
        self.post_up = DoubleConvBlock(out_channels + skip_channels, out_channels)
        self.residual = ResidualConvBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.activation(x)
        if x.shape[-1] != skip.shape[-1]:
            # Pad/crop for odd lengths
            diff = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, diff))
        x = torch.cat([skip, x], dim=1)
        x = self.post_up(x)
        x = self.residual(x)
        return x

    # confirm the order of the forward function, there may also be a relu missing


class WnDUNet1D(nn.Module):
    """
    WnD-UNet backbone.

    Args:
        in_channels: concatenated channels of noisy waveform + pre-computed DWT features.
        base_channels: width multiplier for the first encoder stage.
        num_layers: number of encoder stages (must be 5 to match spec for now).
        out_channels: channels in final prediction (e.g., 1 for waveform).
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        channel_multipliers=(1, 2, 4, 8, 8),
        out_channels: int = 1,
    ):
        super().__init__()
        assert len(channel_multipliers) == 5, "Expect five encoder stages."
        widths = [base_channels * m for m in channel_multipliers]

        self.stem = nn.Conv1d(in_channels, widths[0], kernel_size=3, padding=1, bias=False)

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for idx in range(5):
            in_ch = widths[idx - 1] if idx > 0 else widths[0]
            out_ch = widths[idx]
            self.encoders.append(EncoderBlock(in_ch, out_ch))
            if idx < 4:
                self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Bottleneck residual refinement
        self.bottleneck = ResidualConvBlock(widths[-1])

        # Decoder path (4 blocks)
        self.decoders = nn.ModuleList()
        decoder_in = widths[-1]
        for idx in reversed(range(4)):
            self.decoders.append(DecoderBlock(decoder_in, widths[idx], widths[idx]))
            decoder_in = widths[idx]

        self.head = nn.Sequential(
            nn.Conv1d(widths[0], widths[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(widths[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(widths[0] // 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, time)
        x = self.stem(x)

        skips = []
        for idx, encoder in enumerate(self.encoders):
            x = encoder(x)
            if idx < len(self.pools):
                skips.append(x)
                x = self.pools[idx](x)
            else:
                skips.append(x)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = decoder(x, skip)

        return self.head(x)

class wnd_unet(BaseModel):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        base_channels: int = 64,
        channel_multipliers=(1, 2, 4, 8, 8),
        in_channels: int = 2,
        out_channels: int = 1,
        lr_scheduler_patience: int = 3,
        lr_scheduler_factor: float = 0.5,
    ):
        super().__init__()
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.learning_rate = learning_rate
        self.backbone = WnDUNet1D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            out_channels=out_channels,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.dim() != 3:
            raise ValueError(f"Expected (B,C,L), got {x.shape}")
        return self.backbone(x)

    def loss_function(self, clean: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(enhanced, clean)

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.lr_scheduler_factor,
                    patience=self.lr_scheduler_patience,
                    verbose=True
                ),
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1
            }
            return [optimizer], [scheduler]
