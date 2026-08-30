#!/usr/bin/env python3
"""
Plot time-domain waveforms and STFT spectrograms for a set of WAV files.

Generates two PNG files in the output directory:
  - time_domain.png  — overlaid or stacked waveforms
  - stft.png         — stacked spectrograms in dB

Usage:
    python -m scripts.plot_time_domain_and_stft \
        --files clean.wav noisy.wav enhanced.wav \
        --labels Clean Noisy Enhanced \
        --output-dir outputs/plots

    # Or with default labels (filenames):
    python -m scripts.plot_time_domain_and_stft \
        --files outputs/audios/clean.wav outputs/audios/noisy.wav outputs/audios/enhanced.wav \
        --output-dir outputs/plots
"""

import argparse
import os

import numpy as np
import torch
import torchaudio
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_wav_mono(path):
    """Read a WAV file and return mono audio as a float32 numpy array."""
    x, sr = torchaudio.load(path, normalize=False)
    x = x.squeeze(0).to(torch.float32)
    return x.cpu().numpy(), sr


def plot_time_domain(wavs, srs, labels, output_path):
    """Plot waveforms in time domain and save to output_path."""
    n = len(wavs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, (x, sr, label) in enumerate(zip(wavs, srs, labels)):
        t = np.arange(len(x)) / sr
        axes[i].plot(t, x, linewidth=0.8)
        axes[i].set_title(f"{label} (sr={sr} Hz)")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.4)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved time-domain plot to {output_path}")


def plot_stfts(wavs, srs, labels, output_path, n_fft=256, hop=128):
    """Plot STFT magnitude spectrograms and save to output_path."""
    n = len(wavs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    last_img = None
    for i, (x, sr, label) in enumerate(zip(wavs, srs, labels)):
        S = librosa.stft(x, n_fft=n_fft, hop_length=hop)
        Y = np.abs(S) ** 2
        Y_db = librosa.power_to_db(Y)
        last_img = librosa.display.specshow(
            Y_db, sr=sr, hop_length=hop, x_axis="time", y_axis="linear", ax=axes[i]
        )
        axes[i].set_title(f"STFT: {label}  (n_fft={n_fft}, hop={hop})")
        axes[i].set_ylabel("Hz")

    fig.colorbar(last_img, ax=axes, format="%+2.0f dB", pad=0.01)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved STFT plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot time-domain and STFT for WAV files")
    parser.add_argument("--files", nargs="+", required=True,
                        help="Paths to WAV files to plot")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each file (default: filenames)")
    parser.add_argument("--output-dir", type=str, default="outputs/plots",
                        help="Directory to save plots")
    parser.add_argument("--n-fft", type=int, default=256,
                        help="FFT size for STFT (default: 256)")
    parser.add_argument("--hop", type=int, default=128,
                        help="Hop size for STFT (default: 128)")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        parser.error("--labels must have the same count as --files")

    os.makedirs(args.output_dir, exist_ok=True)

    labels = args.labels if args.labels else [os.path.basename(f) for f in args.files]

    wavs, srs = [], []
    for f in args.files:
        x, sr = read_wav_mono(f)
        wavs.append(x)
        srs.append(sr)

    time_path = os.path.join(args.output_dir, "time_domain.png")
    stft_path = os.path.join(args.output_dir, "stft.png")

    plot_time_domain(wavs, srs, labels, time_path)
    plot_stfts(wavs, srs, labels, stft_path, n_fft=args.n_fft, hop=args.hop)


if __name__ == "__main__":
    main()
