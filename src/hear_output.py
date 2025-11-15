import os
import torch
import torchaudio
import soundfile as sf
from utils import safe_open_yaml
import models
from datamodule import WavPairDataset

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/train_dcctn.yaml")
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "logs/dcctn_learning_rate=0.0001_weight_decay=0.0001_betas=79791a4d_stft_loss_config=cfg-3a09dd93/version_5/checkpoints/epoch=017-step=756-val/loss=36.58.ckpt"
    )

# Point to your audio here
AUDIO_PATH = "/home/jagat/Chittem/RASE-Challenge-team-quazo/dataset/Task1/Recorded/train/14-212-0024_recorded_aligned.wav"
CLEAN_PATH = "/home/jagat/Chittem/RASE-Challenge-team-quazo/dataset/Task1/Clean/train/14-212-0024.wav"  # optional; set a path if you also want to save clean.wav

# Optional: crop to the training window length (4 s in your config)
CROP_TO_TRAIN_LENGTH = True
FIXED_START_SEC = 0.0  # used only if CROP_TO_TRAIN_LENGTH is True

config = safe_open_yaml(CONFIG_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cls = getattr(models, config["model"])
model = model_cls.load_from_checkpoint(
    CHECKPOINT_PATH, map_location=device, **config["model_params"]
).to(device)
model.eval()

# Build a tiny dataset for this exact pair
ds = WavPairDataset(
    [AUDIO_PATH], [CLEAN_PATH],
    task="Task1",
    length_sec=config["datamodule"]["length_sec"],
    random_crop=False,
    energy_crop=True,    # or True, to mimic val
)

sample = ds[0]
x = sample["recorded"].unsqueeze(0).to(device)
clean = sample["clean"].unsqueeze(0).to(device)
scale = float(sample["scale_recorded"])

with torch.no_grad():
    y = model(x).squeeze(0)

print("Model output y (normalized):", y.min().item(), y.max().item(), y.std().item())
enhanced_unclipped = (y.cpu() * scale)
print("Enhanced denorm (before clamp):", enhanced_unclipped.min().item(),
      enhanced_unclipped.max().item(), enhanced_unclipped.std().item())

noisy     = (x.cpu().squeeze(0) * scale)
clean     = (clean.cpu().squeeze(0) * scale)
enhanced  = (y.cpu() * scale)

# ensure safe range for PCM_16
def peak_norm(t):
    m = t.abs().max().item()
    return t / max(m, 1.0)

noisy_w     = peak_norm(noisy)
clean_w     = peak_norm(clean)
enhanced_w  = peak_norm(enhanced)

sf.write("noisy.wav",    noisy_w.numpy(),    sample["fs"], subtype="PCM_16")
sf.write("clean.wav",    clean_w.numpy(),    sample["fs"], subtype="PCM_16")
sf.write("enhanced.wav", enhanced_w.numpy(), sample["fs"], subtype="PCM_16")
