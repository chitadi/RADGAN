#!/usr/bin/env python3
"""
Run RAD-GAN inference on a single audio file and save the output.

Saves noisy.wav, clean.wav (if provided), and enhanced.wav to the
output directory. Also prints output statistics.

Usage:
    python -m scripts.hear_output \
        --config src/config/train_radgan.yaml \
        --checkpoint path/to/model.ckpt \
        --audio path/to/recorded.wav \
        --output-dir outputs/audios

    # Optionally provide a clean reference for comparison:
    python -m scripts.hear_output \
        --config src/config/train_radgan.yaml \
        --checkpoint path/to/model.ckpt \
        --audio path/to/recorded.wav \
        --clean path/to/clean.wav \
        --output-dir outputs/audios
"""

import argparse
import os
import sys

import torch
import soundfile as sf

# Ensure src/ is on the path so `import models` and `from datamodule` work
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import models
from datamodule import WavPairDataset
from utils import safe_open_yaml


def main():
    parser = argparse.ArgumentParser(description="Run RAD-GAN inference on a single audio file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to recorded/noisy WAV file")
    parser.add_argument("--clean", type=str, default=None,
                        help="Path to clean reference WAV file (optional)")
    parser.add_argument("--output-dir", type=str, default="outputs/audios",
                        help="Directory to save output WAV files")
    parser.add_argument("--task", type=str, default="Task1",
                        help="Task name for the audio (Task1 or Task2)")
    parser.add_argument("--start-sec", type=float, default=0.0,
                        help="Start time in seconds for cropping (default: 0.0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = safe_open_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_cls = getattr(models, config["model"])
    model = model_cls.load_from_checkpoint(
        args.checkpoint, map_location=device, **config["model_params"]
    ).to(device)
    model.eval()

    # Build a single-sample dataset
    clean_path = args.clean if args.clean else args.audio  # dummy if no clean
    ds = WavPairDataset(
        [args.audio], [clean_path],
        task=args.task,
        length_sec=config["datamodule"]["length_sec"],
        random_crop=False,
        energy_crop=True,
        fixed_window=True,
        fixed_start_sec=args.start_sec,
    )

    sample = ds[0]
    x = sample["recorded"].unsqueeze(0).to(device)
    clean = sample["clean"].unsqueeze(0).to(device)
    scale = float(sample["scale_recorded"])
    fs = sample["fs"]

    with torch.no_grad():
        y = model(x).squeeze(0)

    print(f"Model output stats — min: {y.min().item():.4f}, max: {y.max().item():.4f}, std: {y.std().item():.4f}")

    # Denormalize to original scale
    noisy = (x.cpu().squeeze(0) * scale)
    clean = (clean.cpu().squeeze(0) * scale)
    enhanced = (y.cpu() * scale)

    # Save outputs
    noisy_path = os.path.join(args.output_dir, "noisy.wav")
    enhanced_path = os.path.join(args.output_dir, "enhanced.wav")

    sf.write(noisy_path, noisy.numpy(), fs, subtype="PCM_16")
    sf.write(enhanced_path, enhanced.numpy(), fs, subtype="PCM_16")
    print(f"Saved noisy audio to {noisy_path}")
    print(f"Saved enhanced audio to {enhanced_path}")

    if args.clean:
        clean_path = os.path.join(args.output_dir, "clean.wav")
        sf.write(clean_path, clean.numpy(), fs, subtype="PCM_16")
        print(f"Saved clean audio to {clean_path}")


if __name__ == "__main__":
    main()
