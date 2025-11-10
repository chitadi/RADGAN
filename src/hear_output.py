import torch
import soundfile as sf
from datamodule import DataModule
from utils import safe_open_yaml
import models
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/train_dcctn.yaml")
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    # "logs/dcctn__learning_rate=0.0003_weight_decay=1e-05_betas=[0.5, 0.999]_"
    # "stft_loss_config={'fft_size': 320, 'hop_size': 80, 'win_length': 320, "
    # "'scale_invariance': False, 'w_sc': 0.0, 'weight': 50.0}/version_8/"
    # "checkpoints/epoch=011-step=3972-val/loss=84.22.ckpt"
    "logs/dcctn_learning_rate=0.0001_weight_decay=1e-05_betas=8421bd69_stft_loss_config=cfg-1e1b5775_backbone_kwargs=cfg-c0580252/version_2/checkpoints/epoch=019-step=840-val/loss=102.34.ckpt"
)
SAMPLE_INDEX = 10

config = safe_open_yaml(CONFIG_PATH)
dm = DataModule(**config["datamodule"])
dm.setup("fit")
sample = dm.dataset["val"][SAMPLE_INDEX]
fs = sample["fs"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cls = getattr(models, config["model"])
model = model_cls.load_from_checkpoint(
    CHECKPOINT_PATH, map_location=device, **config["model_params"]
).to(device)
model.eval()

import torch
x = torch.as_tensor(sample["recorded"], dtype=torch.float32).unsqueeze(0).to(device)
scale_rec = float(torch.as_tensor(sample["scale_recorded"]))
scale_clean = float(torch.as_tensor(sample["scale_clean"]))
with torch.no_grad():
    y = model(x).squeeze(0)

enhanced = (y.cpu() * scale_rec).clamp_(-1.0, 1.0)
noisy = (torch.as_tensor(sample["recorded"], dtype=torch.float32) * scale_rec).clamp_(-1.0, 1.0)
clean = (torch.as_tensor(sample["clean"], dtype=torch.float32) * scale_clean).clamp_(-1.0, 1.0)

sf.write("noisy.wav", noisy.numpy(), fs)
sf.write("clean.wav", clean.numpy(), fs)
sf.write("enhanced.wav", enhanced.numpy(), fs)