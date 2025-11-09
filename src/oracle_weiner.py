import numpy as np
import scipy.signal as sg
from numpy.fft import rfft, irfft
import torch
import torchaudio
import scipy.io.wavfile as wav
# ------------------------ Utility functions ------------------------

def _to_float(x: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in [-1,1]."""
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float32) / (np.iinfo(x.dtype).max + 1.0)
    return x.astype(np.float32)


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Ensure shape (N, C)."""
    return x[:, None] if x.ndim == 1 else x


def _frame_params(sr: int, frame_ms: float, hop_ratio: float):
    frame = int(round(frame_ms * 1e-3 * sr))
    hop = max(1, int(round(hop_ratio * frame)))
    win = sg.windows.hann(frame, sym=False).astype(np.float32)
    Ew = np.sum(win ** 2)
    return frame, hop, win, Ew


def _ola_buffers(N: int, C: int):
    return np.zeros((N, C), dtype=np.float32), np.zeros((N, C), dtype=np.float32)

def load_waveform(file_path):
    waveform, sr = torchaudio.load(file_path, normalize=False)
    waveform = waveform.squeeze(0).unsqueeze(1)
    return waveform.to(torch.float32), sr

def oracle_wiener(y, x, fs, n_fft=1024, frame_ms=20, hop_ratio=0.5):
    """Ideal Wiener filter using clean signal PSD."""
    d = y - x
    N, C = y.shape
    frame, hop, win, Ew = _frame_params(fs, frame_ms, hop_ratio)
    frames = np.arange(0, N - frame + 1, hop)
    out, den = _ola_buffers(N, C)

    for c in range(C):
        for start in frames:
            xseg = x[start:start+frame, c] * win
            dseg = d[start:start+frame, c] * win
            X = rfft(xseg, n=n_fft)
            D = rfft(dseg, n=n_fft)

            Phi_x = np.abs(X) ** 2 / (Ew + 1e-12)
            Phi_d = np.abs(D) ** 2 / (Ew + 1e-12)
            G = Phi_x / (Phi_x + Phi_d + 1e-12)

            Y = rfft(y[start:start+frame, c] * win, n=n_fft)
            S_hat = G * Y

            s = irfft(S_hat, n=n_fft).real[:frame]
            out[start:start+frame, c] += s
            den[start:start+frame, c] += win ** 2

    out /= (den + 1e-12)
    return out
def main():
    # ======= USER SETTINGS =======
    clean_audio_file = "clean_task2.wav"
    recorded_audio_file = "recorded_task2.wav"

    out_path_classic = "enhanced_classic.wav"
    out_path_oracle = "enhanced_oracle.wav"
    # =============================

    clean, sr_clean = load_waveform(clean_audio_file)
    noisy, sr_noisy = load_waveform(recorded_audio_file)

    clean = clean.float()
    noisy = noisy.float()
    print(clean.shape)
    scale = torch.max(torch.abs(clean))
    clean = clean / scale
    noisy = noisy / scale

    assert sr_clean == sr_noisy

    # --- 2. Oracle Wiener ---
    enhanced_oracle = oracle_wiener(noisy, clean, sr_clean)
    # wav.write(out_path_oracle, sr_clean, enhanced_oracle.squeeze().astype(np.float32))
    # print(f"Saved oracle Wiener: {out_path_oracle}")


if __name__ == "__main__":
    main()
