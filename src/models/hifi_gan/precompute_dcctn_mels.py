import os
import glob
import json
import sys
import numpy as np
import torch
import torchaudio

from env import AttrDict
from mel_dataset import mel_spectrogram, MAX_WAV_VALUE, list_recorded_clean_pairs_all_tasks

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # .../src

# Make `src/` importable so `import models` works
os.chdir(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import models
from utils import safe_open_yaml


def load_dcctn():
    # paths: adjust CKPT to your best DCCTN run
    dcctn_cfg_path = os.path.join(SRC_DIR, "config", "train_dcctn.yaml")
    dcctn_ckpt = "/home/jagat/Chittem/RASE-Challenge-team-quazo/src/logs/dcctn_learning_rate=0.0001_weight_decay=0.0001_betas=79791a4d_stft_loss_config=multi-e743a5b6/version_0/checkpoints/epoch=013-step=588-val/loss=98.89.ckpt"  # TODO: fill in
    cfg = safe_open_yaml(dcctn_cfg_path)
    dcctn_cls = getattr(models, cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = dcctn_cls.load_from_checkpoint(
        dcctn_ckpt, map_location=device, **cfg["model_params"]
    ).to(device)
    print("Loaded model:", type(model))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, device


def load_hifigan_config():
    cfg_path = os.path.join(THIS_DIR, "config.json")
    with open(cfg_path) as f:
        data = json.load(f)
    return AttrDict(data)


def list_task_pairs(dataset_root, tasks=("Task1", "Task2"), split="train"):
    """
    split: 'train' or 'dev'
    recorded filenames end with '_recorded_aligned.wav'
    clean filenames have the same id without that suffix.
    """
    rec_paths = []
    clean_paths = []
    for task in tasks:
        task_root = os.path.join(dataset_root, task)
        rec_dir = os.path.join(task_root, "Recorded", split)
        clean_dir = os.path.join(task_root, "Clean", split)

        task_rec = sorted(glob.glob(os.path.join(rec_dir, "*.wav")))
        for rp in task_rec:
            base = os.path.basename(rp)
            clean_name = base.replace("_recorded_aligned", "")
            rec_paths.append(rp)
            clean_paths.append(os.path.join(clean_dir, clean_name))
    assert len(rec_paths) == len(clean_paths)
    return rec_paths, clean_paths


def process_split(dcctn, dcctn_device, h, dataset_root, split, out_root):
    rec_paths, clean_paths = list_recorded_clean_pairs_all_tasks(dataset_root, split)
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    for rec_path, clean_path in zip(rec_paths, clean_paths):
        # base = os.path.splitext(os.path.basename(clean_path))[0]
        # out_path = os.path.join(out_dir, base + ".npy")
        # print(f"Processing {rec_path} -> {out_path}")
        wav_clean, sr_c = torchaudio.load(clean_path, normalize=False)
        wav_rec, sr_r = torchaudio.load(rec_path, normalize=False)
        assert sr_c == sr_r == h.sampling_rate

        clean = wav_clean.squeeze(0).to(torch.float32)
        rec = wav_rec.squeeze(0).to(torch.float32)

        scale_val = max(clean.abs().max().item(), 1e-8)
        rec_norm = rec / scale_val

        with torch.no_grad():
            y = dcctn(rec_norm.unsqueeze(0).to(dcctn_device))  # (1, T)
            y = y.squeeze(0).cpu() * scale_val                # denorm to match clean range

        y = y.unsqueeze(0)  # (1, T)
        mel_cond = mel_spectrogram(
            y,
            n_fft=h.n_fft,
            num_mels=h.num_mels,
            sampling_rate=h.sampling_rate,
            hop_size=h.hop_size,
            win_size=h.win_size,
            fmin=h.fmin,
            fmax=h.fmax,          # 0–1 kHz conditioning
            center=False,
        ).squeeze(0).numpy()      # (num_mels, frames)
        
        rel = os.path.relpath(clean_path, dataset_root)      # 'Task1/Clean/train/14-212-0024.wav'
        parts = rel.split(os.sep)
        task_name = parts[0]                                 # 'Task1'
        split = parts[2]   

        base = os.path.splitext(os.path.basename(clean_path))[0]
        out_dir = os.path.join(out_root, task_name, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, base + ".npy")
        np.save(out_path, mel_cond)
        print(f"Saved {out_path}")


def main():
    dataset_root = os.path.abspath(os.path.join(SRC_DIR, "..", "dataset"))
    out_root = os.path.join(THIS_DIR, "phase2_mels")
    print("dataset_root:", dataset_root)
    print("out_root:", out_root)

    h = load_hifigan_config()
    print("Loaded HiFi-GAN config")
    dcctn, dcctn_device = load_dcctn()
    print("Loaded DCCTN on", dcctn_device)

    for split in ["train", "val"]:
        process_split(dcctn, dcctn_device, h, dataset_root, split, out_root)


if __name__ == "__main__":
    main()
