import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torchaudio
import matplotlib.pyplot as plt
import torch



def read_wav_mono(path):
    """Read WAV file and return mono audio and sample rate."""
    # x, sr = sf.read(path, always_2d=True)
    # x = x.mean(axis=1)  # convert to mono
    x, sr = torchaudio.load(path, normalize=False)
    x    = x.squeeze(0).to(torch.float32)
    # x = x.cpu().numpy()

    return x.cpu().numpy(), sr


def plot_time_domain(wavs, srs, labels, normalize=True):
    """Plot waveforms in time domain."""
    n = len(wavs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, (x, sr, label) in enumerate(zip(wavs, srs, labels)):
        
        # if normalize:
        #     x = x / np.max(np.abs(x) + 1e-12)
        t = np.arange(len(x)) / sr
        axes[i].plot(t, x, linewidth=0.8)
        axes[i].set_title(f"{label} (sr={sr} Hz)")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.4)

    axes[-1].set_xlabel("Time (s)")
    # plt.tight_layout()
    plt.savefig("time_plot.png")


def plot_stfts(wavs, srs, labels, n_fft=256, hop=128, window="hann"):
    """Plot STFT magnitude spectrograms."""
    n = len(wavs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, (x, sr, label) in enumerate(zip(wavs, srs, labels)):

        #method 1 (preferred)
        # S_scale = librosa.stft(x, n_fft=n_fft, hop_length=hop)
        # Y_scale = np.abs(S_scale) 
        # Y_log_scale = 20*np.log10(Y_scale)
        # #method 2
        S_scale = librosa.stft(x, n_fft=n_fft, hop_length=hop)
        Y_scale = np.abs(S_scale) **2
        Y_log_scale = librosa.power_to_db(Y_scale)
        last_img = librosa.display.specshow(
            Y_log_scale, sr=sr, hop_length=hop, x_axis="time", y_axis="linear", ax=axes[i]
        )
        axes[i].set_title(f"STFT: {label}  (n_fft={n_fft}, hop={hop}, win={window})")
        axes[i].set_ylabel("Hz")
        # plt.colorbar(format="%+2.f")

        # f, t, Z = stft(x, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop)
        # mag = np.abs(Z)
        # db = 20 * np.log10(mag + 1e-12)
        # im = axes[i].pcolormesh(t, f, db, shading="gouraud")
        # axes[i].set_title(f"STFT: {label}")
        # axes[i].set_ylabel("Freq (Hz)")
        # fig.colorbar(im, ax=axes[i], label="Magnitude (dB)")
    fig.colorbar(last_img, ax=axes, format="%+2.0f dB", pad=0.01)

    axes[-1].set_xlabel("Time (s)")
    # plt.tight_layout()
    plt.savefig("stft_plots.png")


if __name__ == "__main__":
    wav_files = [
        "clean.wav",
        "noisy.wav",
        "enhanced.wav",
    ]

    # Load files
    wavs, srs, labels = [], [], []
    for f in wav_files:
        x, sr = read_wav_mono(f)
        wavs.append(x)
        srs.append(sr)
        labels.append(f)

    # Plot
    plot_time_domain(wavs, srs, labels)
    plot_stfts(wavs, srs, labels)
