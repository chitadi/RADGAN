import numpy as np
import librosa
import scipy.io.wavfile as wav

def enhance_low_harmonics_spectral(
    y,
    sr,
    n_fft=512,
    hop_length=128,
    f0_min=70.0,
    f0_max=300.0,
    max_freq=1000.0,
    num_harmonics=4,
    bw_hz=40.0,
    alpha=0.5,
):
    y = y.astype(np.float32)

    S = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    mag = np.abs(S)
    phase = np.angle(S)
    F, T = mag.shape

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    df = freqs[1] - freqs[0]

    max_bin = np.searchsorted(freqs, max_freq)

    f0_track = librosa.yin(
        y,
        fmin=f0_min,
        fmax=f0_max,
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length,
    )  
    if len(f0_track) < T:
        f0_track = np.pad(f0_track, (0, T - len(f0_track)), mode="edge")
    elif len(f0_track) > T:
        f0_track = f0_track[:T]

    f0_track = np.nan_to_num(f0_track, nan=0.0)
    f0_track[(f0_track < f0_min) | (f0_track > f0_max)] = 0.0

    mag_enh = mag.copy()

    for t in range(T):
        f0 = f0_track[t]
        if f0 <= 0.0:
            continue  

        comb = np.zeros(F, dtype=np.float32)

        for m in range(1, num_harmonics + 1):
            fc = m * f0
            if fc >= max_freq:
                break
            gauss = np.exp(-0.5 * ((freqs - fc) / bw_hz) ** 2)
            comb = np.maximum(comb, gauss)  

        if comb[:max_bin].max() > 0:
            comb[:max_bin] /= comb[:max_bin].max()

        mask = np.ones(F, dtype=np.float32)
        mask[:max_bin] = alpha + (1.0 - alpha) * comb[:max_bin]

        mag_enh[:, t] *= mask

    S_enh = mag_enh * np.exp(1j * phase)
    y_enh = librosa.istft(
        S_enh,
        hop_length=hop_length,
        window="hann",
        center=True,
    )

    return y_enh.astype(np.float32), S, S_enh, f0_track.astype(np.float32)
