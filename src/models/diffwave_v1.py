# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

from math import sqrt
import math


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


def cosine_beta_schedule(T, s=0.008):
  """
  Returns a list of betas computed from the cosine schedule.
  Args:
      T (int): number of diffusion steps
      s (float): small offset to prevent singularities near t=0
  """
  steps = T
  def f(t):
      return math.cos((t / steps + s) / (1 + s) * math.pi / 2) ** 2

  alphas_cumprod = [f(t) / f(0) for t in range(T + 1)]
  betas = []
  for t in range(1, T + 1):
      beta_t = 1 - alphas_cumprod[t] / alphas_cumprod[t - 1]
      betas.append(min(beta_t, 0.999))  # clamp for numerical stability
  return betas


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


# class SpectrogramUpsampler(nn.Module):
#   def __init__(self, n_bands):
#     super().__init__()
#     self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
#     self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])

#   def forward(self, x):
#     x = torch.unsqueeze(x, 1)
#     x = self.conv1(x)
#     x = F.leaky_relu(x, 0.4)
#     x = self.conv2(x)
#     x = F.leaky_relu(x, 0.4)
#     x = torch.squeeze(x, 1)
#     return x


class WaveletUpsampler(nn.Module):
  def __init__(self, in_channels=1, out_channels=1, upsample_factors=[2, 2, 2]):
    super().__init__()
    layers = []
    channels = in_channels
    for factor in upsample_factors:
      layers.append(
        nn.ConvTranspose1d(
          channels,
          out_channels,
          kernel_size=factor * 2,
          stride=factor,
          padding=factor // 2,
          output_padding=0
        )
      )
      layers.append(nn.LeakyReLU(0.4))
      channels = out_channels
    self.net = nn.Sequential(*layers)

  def forward(self, x):
      return self.net(x)


class RadarConditioner(nn.Module):
  def __init__(self, in_channels=5, residual_channels=128, up_channels=128, target_length=32000):
    super().__init__()
    self.in_channels = in_channels
    self.layers = nn.Sequential(
      nn.ConvTranspose1d(in_channels, 64, kernel_size=16, stride=4, padding=6),
      nn.GroupNorm(8, 64),  # x = F.leaky_relu(x, 0.4)
      nn.LeakyReLU(0.4),
      nn.ConvTranspose1d(64, residual_channels, kernel_size=8, stride=2, padding=3),
      nn.GroupNorm(8, residual_channels),  # x = F.leaky_relu(x, 0.4)
      # nn.Conv1d(up_ch, 2*res_channels, kernel_size=1)
      nn.LeakyReLU(0.4)
    )
    self.target_length = target_length

  def forward(self, bands):
    out = self.layers(bands)
    out = F.interpolate(out, size=self.target_length, mode='linear', align_corners=False)
    return out


class ResidualBlock(nn.Module):
  def __init__(self, up_channels, residual_channels, dilation):
    '''
    :param n_bands: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    self.conditioner_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step, conditioner):
    assert (conditioner is not None and self.conditioner_projection is not None)

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    conditioner = self.conditioner_projection(conditioner)
    # print("conditioner.shape:", conditioner.shape)
    y = self.dilated_conv(y)
    # print("y.shape:", y.shape)
    y = y + conditioner

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class DiffWave(BaseModel):
  def __init__(self, learning_rate=2e-4, hop_samples=256, residual_layers=30, 
               residual_channels=64, up_channels=128, dilation_cycle_length=10, 
               n_bands=4, **kwargs):
    super().__init__()
    
    # Store parameters
    self.learning_rate = learning_rate
    self.hop_samples = hop_samples
    self.residual_layers = residual_layers
    self.residual_channels = residual_channels
    self.up_channels = up_channels
    self.dilation_cycle_length = dilation_cycle_length
    self.n_bands = n_bands
    
    self.inference_noise_schedule = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]
    # --- Cosine noise schedule (Improved DDPM) ---
    self.noise_schedule = cosine_beta_schedule(50, s=0.008)  # 50 diffusion steps
    beta = np.array(self.noise_schedule, dtype=np.float32)
    alpha = 1.0 - beta
    alpha_bar = np.cumprod(alpha)
    self.noise_level = torch.tensor(alpha_bar.astype(np.float32))

    self.input_projection = Conv1d(1, residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(len(self.noise_schedule))

    self.upsampler_a3 = WaveletUpsampler(1, 1, upsample_factors=[2,2,2])
    self.upsampler_d3 = WaveletUpsampler(1, 1, upsample_factors=[2,2,2])
    self.upsampler_d2 = WaveletUpsampler(1, 1, upsample_factors=[2,2])
    self.upsampler_d1 = WaveletUpsampler(1, 1, upsample_factors=[2])
    self.radar_conditioner = RadarConditioner()

    self.residual_layers = nn.ModuleList([
        ResidualBlock(residual_channels, residual_channels, 2**(i % dilation_cycle_length))
        for i in range(residual_layers)
    ])
    self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
    self.output_projection = Conv1d(residual_channels, 1, 1)
    nn.init.zeros_(self.output_projection.weight)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), 
                                lr=self.learning_rate)
    # warmup_steps = 1000
    total_steps = self.trainer.estimated_stepping_batches  # populated after setup

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,      # Number of epochs for first restart cycle
        T_mult=2,    # Multiplier for subsequent cycle lengths (2 means each cycle is 2x longer)
        eta_min=1e-6 # Minimum learning rate
    )
    
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",  # Update per epoch
            "frequency": 1,
        }
    }

  def forward(self, audio, diffusion_step, features):
    assert (features is not None and self.radar_conditioner is not None)

    # x = audio.unsqueeze(1)
    x = self.input_projection(audio)
    x = F.relu(x)

    diffusion_step = self.diffusion_embedding(diffusion_step)
    # if self.spectrogram_upsampler: # use conditional model
    #   spectrogram = self.spectrogram_upsampler(spectrogram)
    upsampled = {}
    upsampled["m_t"] = features["m_t"].unsqueeze(1)
    upsampled["a3"] = self.upsampler_a3(features["a3"].unsqueeze(1))
    upsampled["d3"] = self.upsampler_d3(features["d3"].unsqueeze(1))
    upsampled["d2"] = self.upsampler_d2(features["d2"].unsqueeze(1))
    upsampled["d1"] = self.upsampler_d1(features["d1"].unsqueeze(1))
    radar = torch.cat(
      [upsampled["m_t"],
      upsampled["a3"],
      upsampled["d3"],
      upsampled["d2"],
      upsampled["d1"]], dim=1)
    # print("radar shape before conditioner:", radar.shape)
    radar = self.radar_conditioner(radar)
    # print("radar shape after conditioner:", radar.shape)

    skip = None
    for layer in self.residual_layers:
      x, skip_connection = layer(x, diffusion_step, radar)
      skip = skip_connection if skip is None else skip_connection + skip

    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x

  def sample_timesteps(self, cur_epoch, max_epoch, total_T, batch_size, device):
    # Start with only the first 20% of noise levels
    frac = min(1.0, 0.2 + 0.8 * (cur_epoch / max_epoch))
    max_t = int(total_T * frac)
    t = torch.randint(0, max_t, (batch_size,), device=device)
    return t

  def compute_noise_prediction_loss(self, batch):
    features = batch["recorded"]
    audio = batch["clean"]

    # Move to correct device
    self.noise_level = self.noise_level.to(self.device)

    # Sample timesteps
    t = self.sample_timesteps(
        self.current_epoch, self.trainer.max_epochs,
        len(self.noise_schedule), features["m_t"].shape[0], self.device
    )

    # Compute noise scale and SNR
    alpha_bar = self.noise_level[t].unsqueeze(1)
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
    inv_snr_weight = (1.0 / (snr + 1e-8)).detach()
    inv_snr_weight = inv_snr_weight / inv_snr_weight.mean()  # normalize

    # Forward diffusion: add noise
    noise = torch.randn_like(audio)
    noisy_audio = alpha_bar.sqrt().unsqueeze(-1) * audio + \
                  (1.0 - alpha_bar).sqrt().unsqueeze(-1) * noise

    # Predict noise and compute weighted loss
    predicted_noise = self.forward(noisy_audio, t, features)
    per_sample_loss = F.l1_loss(predicted_noise.squeeze(1), noise.squeeze(1), reduction='none')
    per_sample_loss = per_sample_loss.mean(dim=[1])  # one loss per sample
    weighted_loss = (inv_snr_weight.squeeze() * per_sample_loss).mean() # TODO RuntimeError: The size of tensor a (16) must match the size of tensor b (32000) at non-singleton dimension 1

    return weighted_loss


  def inference(self, batch, fast_sampling=False):
    features = batch["recorded"]
    audio = batch["clean"]
    # for key in features:
    #   print(key, "shape:", features[key].shape)
    # print("audio.shape:", audio.shape)

    self.eval()
    with torch.inference_mode():
      training_noise_schedule = np.array(self.noise_schedule)
      inference_noise_schedule = np.array(self.inference_noise_schedule) if fast_sampling else training_noise_schedule
      talpha = 1 - training_noise_schedule
      talpha_cum = np.cumprod(talpha)

      beta = inference_noise_schedule
      alpha = 1 - beta
      alpha_cum = np.cumprod(alpha)

      T = []
      for s in range(len(inference_noise_schedule)):
        for t in range(len(training_noise_schedule) - 1):
          if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
            twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
            T.append(t + twiddle)
            break
      T = np.array(T, dtype=np.float32)

      output_length = features["m_t"].shape[-1] * self.hop_samples \
                      if len(features["m_t"].shape) == 2 \
                      else self.hop_samples * features["m_t"].shape[0]
      audio = torch.randn(features["m_t"].shape[0], output_length, device=self.device)
      noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(self.device)

      for n in range(len(alpha) - 1, -1, -1):
        c1 = 1 / alpha[n]**0.5
        c2 = beta[n] / (1 - alpha_cum[n])**0.5
        audio = c1 * (audio - c2 * self(audio, torch.tensor([T[n]], device=self.device), features).squeeze(1))
        if n > 0:
          noise = torch.randn_like(audio)
          sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
          audio += sigma * noise
        audio = torch.clamp(audio, -1.0, 1.0)
    return audio

  def training_step(self, batch, batch_idx):
    loss = self.compute_noise_prediction_loss(batch)
    self.log('train/loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    loss = self.compute_noise_prediction_loss(batch)
    self.log('val/loss', loss)
    return loss

  def test_step(self, batch, batch_idx):
    audio = self.inference(batch)
    self.test_outputs.append(audio.detach().cpu().numpy())
    self.test_targets.append(batch["clean"].detach().cpu().numpy())
    self.test_tasks.append(batch["task"])


# Alias for model registry compatibility
diffwave_v1 = DiffWave