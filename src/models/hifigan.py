from .hifi_gan.env import AttrDict
import os
import math
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import itertools
import torchaudio

from .base_model import BaseModel
from .hifi_gan.hifi_gan import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiMelDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from .hifi_gan.mel_dataset import mel_spectrogram, MAX_WAV_VALUE
from .hifi_gan.utils import scan_checkpoint, load_checkpoint
# from weiner_with_spectral_preprocessing import preprocessing_weiner_with_harmonics
# for the ablation 

from weiner_with_spectral_preprocessing import (
    preprocessing_weiner_with_harmonics,
    preprocessing_wiener_only,
    preprocessing_harmonics_only,
)

from auraloss.freq import MultiResolutionSTFTLoss

# from .cleanmel import CleanMel


HIFIGAN_CONFIG = AttrDict(
    dict(
        resblock="1",
        num_gpus=0,
        batch_size=16,
        learning_rate=1e-4,  # will be overridden from YAML
        adam_b1=0.9,
        adam_b2=0.99,
        lr_decay=0.999,
        seed=1234,

        upsample_rates=[8, 8, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],

        segment_size=32000,
        num_mels=80,
        num_freq=1025,
        n_fft=1024,
        hop_size=128,
        win_size=512,
        sampling_rate=8000,

        fmin=0,
        fmax=1000,
        fmax_for_loss=4000,

        num_workers=2,
        hf_mel_weight=5.0,
        hf_cutoff_hz=1000.0,
        mrstft_weight=5.0,

        dist_config=dict(
            dist_backend="nccl",
            dist_url="tcp://localhost:54321",
            world_size=1,
        ),
    )
)

class HiFiGAN(BaseModel):
    def __init__(self, learning_rate: float = 1e-4):
        super().__init__()

        h_dict = dict(HIFIGAN_CONFIG)       # shallow copy
        h_dict["learning_rate"] = learning_rate
        self.h = AttrDict(h_dict)

        # #load cleanmel first
        # CLEANMEL_CKPT = os.path.join(os.path.dirname(__file__), "cleanmel_utils", "loss=1.88.ckpt")
        # self.cleanmel = CleanMel(sr=self.h.sampling_rate)
        # ckpt = torch.load(CLEANMEL_CKPT, map_location="cpu")
        # state = ckpt.get("state_dict", ckpt)
        # self.cleanmel.load_state_dict(state, strict=False)
        # self.cleanmel.eval()
        # for p in self.cleanmel.parameters():
        #     p.requires_grad = False

        # add discriminators for finetuning

        self.generator = Generator(self.h)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.mmd = MultiMelDiscriminator()

        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes=[256, 512, 1024],
            hop_sizes=[64, 128, 256],
            win_lengths=[256, 512, 1024],
            w_sc=1.0,
            w_log_mag=1.0,
            w_lin_mag=0.0,
        )
        
        self.phase_fft_sizes   = [256, 512, 1024]
        self.phase_hop_sizes   = [64, 128, 256]
        self.phase_win_lengths = [256, 512, 1024]

        self.learning_rate = learning_rate
        self.loss_function = self._loss_fn
        self.automatic_optimization = False  # manual optim steps
        
        # --- Load Phase‑1 generator weights only (Phase‑1‑specific) ---
        # This is the only Phase‑1 dependency you still want.
        base_dir = os.path.join(os.path.dirname(__file__), "hifi_gan")
        phase1_dir = os.path.join(base_dir, "checkpoints_pretrain")
        g_phase1 = scan_checkpoint(phase1_dir, "g_")
        if g_phase1 is not None:
            state_g = load_checkpoint(g_phase1, device=torch.device("cpu"))
            self.generator.load_state_dict(state_g["generator"])
            
        self.save_hyperparameters(ignore=["generator", "mpd", "msd", "mmd", "mrstft"])
    
    def _recorded_to_phase2_mel(self, recorded: torch.Tensor, fs) -> torch.Tensor:
        """
        recorded: (B, T) from DataModule (float32, ~[-1,1])
        fs: scalar or tensor sample rate (expects 8000)
        Returns: (B, n_mels, frames) conditioning mel for Phase 2.
        """
        if recorded.dim() == 1:
            recorded = recorded.unsqueeze(0)

        if isinstance(fs, torch.Tensor):
            fs_val = float(fs.flatten()[0].item())
        else:
            fs_val = float(fs)
        assert int(fs_val) == self.h.sampling_rate, f"Expected fs={self.h.sampling_rate}, got {fs_val}"

        # no wiener or harmonics here
        # rec_np = recorded.detach().cpu().numpy()  # (B, T)
        # enh_list = []
        # for x in rec_np:
            # Match Phase‑2 pipeline scale: int16‑like → preprocess → normalize*0.95
            # x_int = x * MAX_WAV_VALUE
            # y_enh = preprocessing_weiner_with_harmonics(x_int, fs_val)  # 1D np
            # changed for ablation
            # y_enh = preprocessing_wiener_only(x_int, fs_val)
            # y_enh = preprocessing_harmonics_only(x_int, fs_val)
            # y_enh = y_enh.astype(np.float32) / MAX_WAV_VALUE
            # y_enh = librosa.util.normalize(y_enh) * 0.95
            # enh_list.append(y_enh)

        # enh_batch = torch.from_numpy(np.stack(enh_list)).to(self.device)  # (B, T')
        enh_batch = recorded.to(self.device).to(torch.float32)  # (B, T)

        mel = mel_spectrogram(
            enh_batch,
            self.h.n_fft,
            self.h.num_mels,
            self.h.sampling_rate,
            self.h.hop_size,
            self.h.win_size,
            self.h.fmin,
            self.h.fmax,          # 0–1 kHz band for conditioning
            center=False,
        )
        return mel
        # wav = recorded.to(self.device).float()  # (B,T)

        # with torch.no_grad():
        #     X, X_norm = self.cleanmel.input_stft(wav)
        #     mel_log = self.cleanmel.forward(X, inference=False)  # (B,80,frames)

        # return mel_log


    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        if noisy.dim() == 1:
            noisy = noisy.unsqueeze(0)  # (1, T)

        recorded = noisy.to(self.device).to(torch.float32)
        mel = self._recorded_to_phase2_mel(recorded, self.h.sampling_rate)
        y_hat = self.generator(mel).squeeze(1)  # (B, T_gen)

        target_len = noisy.size(-1)
        cur_len = y_hat.size(-1)
        if cur_len > target_len:
            y_hat = y_hat[..., :target_len]
        elif cur_len < target_len:
            y_hat = F.pad(y_hat, (0, target_len - cur_len))

        return y_hat
    
    def _loss_fn(self, clean: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        """
        clean, enhanced: (B, T)
        Use a simple MR‑STFT loss between waveforms.
        """
        clean_ = clean.to(self.device)
        enhanced_ = enhanced.to(self.device)

        if clean_.dim() == 2:
            clean_ = clean_.unsqueeze(1)       # (B, 1, T)
        if enhanced_.dim() == 2:
            enhanced_ = enhanced_.unsqueeze(1)

        return self.mrstft(enhanced_, clean_)
    
    def _phase_loss(self, y, y_hat):
        """
        Multi‑resolution phase loss (IP + GD + IAF) from train_hifi_gan_finetune.py.
        y, y_hat: (B,1,T)
        """
        device = y.device
        phase_windows = [
            torch.hann_window(wl, device=device)
            for wl in self.phase_win_lengths
        ]
        two_pi = 2.0 * math.pi

        def anti_wrap(x: torch.Tensor) -> torch.Tensor:
            return (x - two_pi * torch.round(x / two_pi)).abs()

        loss_phase = 0.0
        for N, hop, win_len, window in zip(
            self.phase_fft_sizes, self.phase_hop_sizes,
            self.phase_win_lengths, phase_windows
        ):
            Y = torch.stft(
                y.squeeze(1), N, hop_length=hop, win_length=win_len,
                window=window, center=True, return_complex=True,
            )
            Y_hat = torch.stft(
                y_hat.squeeze(1), N, hop_length=hop, win_length=win_len,
                window=window, center=True, return_complex=True,
            )

            phase      = torch.angle(Y)
            phase_hat  = torch.angle(Y_hat)

            # IP
            diff_ip = anti_wrap(phase_hat - phase)
            L_ip = diff_ip.mean()

            # GD
            gd     = phase[:, 1:, :] - phase[:, :-1, :]
            gd_hat = phase_hat[:, 1:, :] - phase_hat[:, :-1, :]
            diff_gd = anti_wrap(gd_hat - gd)
            L_gd = diff_gd.mean()

            # IAF
            iaf     = phase[:, :, 1:] - phase[:, :, :-1]
            iaf_hat = phase_hat[:, :, 1:] - phase_hat[:, :, :-1]
            diff_iaf = anti_wrap(iaf_hat - iaf)
            L_iaf = diff_iaf.mean()

            loss_phase = loss_phase + (L_ip + L_gd + L_iaf)

        loss_phase = loss_phase * (0.1 / len(self.phase_fft_sizes))
        return loss_phase

    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        # DataModule batch
        recorded = batch["recorded"].to(self.device).to(torch.float32)  # (B, T)
        clean    = batch["clean"].to(self.device).to(torch.float32)     # (B, T)
        fs       = batch.get("fs", self.h.sampling_rate)

        # 1) Build Phase‑2 conditioning mel from recorded (preprocessed)
        x = self._recorded_to_phase2_mel(recorded, fs)   # (B, n_mels, frames)

        # 2) Generator target: clean waveform
        y = clean.unsqueeze(1)                           # (B, 1, T)

        opt_d.zero_grad()
        # --- Discriminator step ---
        with torch.no_grad():
            y_g_hat = self.generator(x)
            # Align length
            tgt_len = y.size(-1)
            gen_len = y_g_hat.size(-1)
            if gen_len > tgt_len:
                y_g_hat = y_g_hat[..., :tgt_len]
            elif gen_len < tgt_len:
                y_g_hat = F.pad(y_g_hat, (0, tgt_len - gen_len))

        # 4) Mel targets for loss (0–4 kHz)
        y_mel = mel_spectrogram(
            y.squeeze(1),
            self.h.n_fft,
            self.h.num_mels,
            self.h.sampling_rate,
            self.h.hop_size,
            self.h.win_size,
            self.h.fmin,
            self.h.fmax_for_loss,
            center=False,
        )
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1),
            self.h.n_fft,
            self.h.num_mels,
            self.h.sampling_rate,
            self.h.hop_size,
            self.h.win_size,
            self.h.fmin,
            self.h.fmax_for_loss,
            center=False,
        )

        # 5) Adversarial + feature matching (Phase‑2)
        y_mel_disc     = y_mel.unsqueeze(1)         # (B,1,n_mels,T)
        y_hat_mel_disc = y_g_hat_mel.unsqueeze(1)

        # Discriminator step (you’d normally do a separate opt_d step;
        # in Lightning you can split optimizers, but this is the core math)
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        y_dm_hat_r, y_dm_hat_g, fmap_m_r, fmap_m_g = self.mmd(y_mel_disc, y_hat_mel_disc)

        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_disc_m, _, _ = discriminator_loss(y_dm_hat_r, y_dm_hat_g)
        loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_m
        self.manual_backward(loss_disc_all)
        # loss_disc_all.backward()
        opt_d.step()
        
        # --- Generator update ---
        opt_g.zero_grad()
        y_g_hat = self.generator(x)
        tgt_len = y.size(-1)
        gen_len = y_g_hat.size(-1)
        if gen_len > tgt_len:
            y_g_hat = y_g_hat[..., :tgt_len]
        elif gen_len < tgt_len:
            y_g_hat = F.pad(y_g_hat, (0, tgt_len - gen_len))
        
            # Mel spectrograms (0–4 kHz) for mel loss + mel disc
        y_mel = mel_spectrogram(
            y.squeeze(1),
            self.h.n_fft,
            self.h.num_mels,
            self.h.sampling_rate,
            self.h.hop_size,
            self.h.win_size,
            self.h.fmin,
            self.h.fmax_for_loss,
            center=False,
        )
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1),
            self.h.n_fft,
            self.h.num_mels,
            self.h.sampling_rate,
            self.h.hop_size,
            self.h.win_size,
            self.h.fmin,
            self.h.fmax_for_loss,
            center=False,
        )
        y_mel_disc     = y_mel.unsqueeze(1)
        y_hat_mel_disc = y_g_hat_mel.unsqueeze(1)
        # HF‑weighted mel loss (4.2) – you can copy hf_weights logic
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45.0
        
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        y_dm_hat_r, y_dm_hat_g, fmap_m_r, fmap_m_g = self.mmd(y_mel_disc, y_hat_mel_disc)

        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        loss_gen_m, _ = generator_loss(y_dm_hat_g)

        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_fm_m = feature_loss(fmap_m_r, fmap_m_g)

        # 6) MR‑STFT waveform loss (4.3)
        loss_mrstft = self.mrstft(y_g_hat, y) * float(self.h.mrstft_weight)
        
        # loss_phase = self._phase_loss(y, y_g_hat)

        # Combine generator losses (simplified from train_hifi_gan_finetune.py)
        loss_gen_all = (
            0.5 * (loss_gen_s + loss_gen_f + loss_gen_m)
            + loss_fm_s + loss_fm_f + loss_fm_m
            + loss_mel
            + loss_mrstft
            # + loss_phase
        )
        # loss_gen_all.backward()
        self.manual_backward(loss_gen_all)
        opt_g.step()
        
        self.log("train/loss_disc", loss_disc_all, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/loss",      loss_gen_all,  on_step=True, on_epoch=True, prog_bar=True)
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
            betas=[self.h.adam_b1, self.h.adam_b2],
        )
        opt_d = torch.optim.AdamW(
            itertools.chain(self.mpd.parameters(),
                            self.msd.parameters(),
                            self.mmd.parameters()),
            lr=self.learning_rate,
            betas=[self.h.adam_b1, self.h.adam_b2],
        )
        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.h.lr_decay)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.h.lr_decay)
        # you can add a scheduler mirroring train_hifi_gan_finetune.py later
        return (
            [opt_d, opt_g],
            [
                {"scheduler": sch_d, "interval": "epoch", "name": "disc_lr"},
                {"scheduler": sch_g, "interval": "epoch", "name": "gen_lr"},
            ],
        )


# so models.__init__ can expose this as "hifi_gan"
hifigan = HiFiGAN
