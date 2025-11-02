import torch
import soundfile as sf
from datamodule import DataModule
from utils import safe_open_yaml
import models
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/train_dcctn.yaml")
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "logs/dcctn__learning_rate=0.0003_weight_decay=1e-05_betas=[0.5, 0.999]_"
    "stft_loss_config={'fft_size': 320, 'hop_size': 80, 'win_length': 320, "
    "'scale_invariance': False, 'w_sc': 0.0, 'weight': 50.0}/version_8/"
    "checkpoints/epoch=011-step=3972-val/loss=84.22.ckpt"
)
SAMPLE_INDEX = 0

config = safe_open_yaml(CONFIG_PATH)
dm = DataModule(**config["datamodule"])
dm.setup("fit")
recorded = dm.dataset["train"][SAMPLE_INDEX]
fs = recorded["fs"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cls = getattr(models, config["model"])
model = model_cls.load_from_checkpoint(
    CHECKPOINT_PATH, map_location=device, **config["model_params"]
).to(device)
model.eval()

inputs = recorded["recorded"].unsqueeze(0).to(device)
scale = recorded["scale"].to(device)

with torch.no_grad():
    enhanced = model(inputs).squeeze(0) * scale

# bring everything back to CPU for saving
enhanced = enhanced.cpu()
clean = (recorded["clean"] * recorded["scale"]).cpu()
noisy = (recorded["recorded"] * recorded["scale"]).cpu()

sf.write("noisy.wav", noisy.numpy(), fs)
sf.write("clean.wav", clean.numpy(), fs)
sf.write("enhanced.wav", enhanced.numpy(), fs)
