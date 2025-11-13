import os
import torch
import torchaudio
import soundfile as sf
from utils import safe_open_yaml
import models

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/train_dcctn.yaml")
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "logs/dcctn_learning_rate=0.0001_weight_decay=1e-05_betas=79791a4d_stft_loss_config=cfg-12aecd2a_backbone_kwargs=cfg-c0580252/version_5/checkpoints/epoch=009-step=420-val/loss=120.29.ckpt"
    )

# Point to your audio here
AUDIO_PATH = "/home/vedang/projects/RASE-Challenge-team-quazo/dataset/Task1/Recorded/train/14-212-0024_recorded_aligned.wav"
CLEAN_PATH = "/home/vedang/projects/RASE-Challenge-team-quazo/dataset/Task1/Clean/train/14-212-0024.wav"  # optional; set a path if you also want to save clean.wav

# Optional: crop to the training window length (4 s in your config)
CROP_TO_TRAIN_LENGTH = False
FIXED_START_SEC = 0.0  # used only if CROP_TO_TRAIN_LENGTH is True

config = safe_open_yaml(CONFIG_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cls = getattr(models, config["model"])
model = model_cls.load_from_checkpoint(
    CHECKPOINT_PATH, map_location=device, **config["model_params"]
).to(device)
model.eval()

# Load recorded audio (mimic datamodule: torchaudio.load(normalize=False))
wav, fs = torchaudio.load(AUDIO_PATH, normalize=False)  # shape: (C, T)
wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)  # mono
wav = wav.to(torch.float32)

# Optional crop/pad to training length
if CROP_TO_TRAIN_LENGTH:
    length_sec = float(config["datamodule"]["length_sec"])
    sample_len = int(length_sec * fs)
    start = int(round(FIXED_START_SEC * fs))
    start = max(0, min(start, max(0, wav.numel() - sample_len)))
    end = start + sample_len
    if wav.numel() < sample_len:
        wpad = torch.zeros(sample_len, dtype=wav.dtype)
        wpad[: wav.numel()] = wav
        wav = wpad
    else:
        wav = wav[start:end]

# Normalize like WavPairDataset
eps = 1e-8
scale_rec = float(max(wav.abs().max().item(), eps))
x = (wav / scale_rec).unsqueeze(0).to(device)  # (1, T)

with torch.no_grad():
    y = model(x).squeeze(0)  # (T,)

# Denormalize and save
enhanced = (y.cpu() * scale_rec).clamp(-1.0, 1.0)
noisy = (wav / scale_rec).clamp(-1.0, 1.0)  # normalized noisy for safe writing

sf.write("noisy.wav", noisy.numpy(), fs)
sf.write("enhanced.wav", enhanced.numpy(), fs)

# Optional: also save a clean reference if you have it
if CLEAN_PATH:
    clean, clean_fs = torchaudio.load(CLEAN_PATH, normalize=False)
    clean = clean.mean(dim=0) if clean.shape[0] > 1 else clean.squeeze(0)
    clean = clean.to(torch.float32)
    if clean_fs != fs:
        clean = torchaudio.functional.resample(clean, orig_freq=clean_fs, new_freq=fs)
    if CROP_TO_TRAIN_LENGTH:
        if clean.numel() < sample_len:
            cpad = torch.zeros(sample_len, dtype=clean.dtype)
            cpad[: clean.numel()] = clean
            clean = cpad
        else:
            clean = clean[start:end]
    clean = (clean / scale_rec).clamp(-1.0, 1.0)  # match dataset's scale_clean = scale_recorded
    sf.write("clean.wav", clean.numpy(), fs)
