from pathlib import Path
import numpy as np
import torchaudio
import pywt

WAVELET     = "coif5"  
LEVEL       = 7    
CYCLE_SPINS = 0        # 0 disables cycle-spinning; try 4, 8, or 16 for -5 dB
PLOT        = True     # quick before/after visualization


def read_wav_mono(path: Path):
    y, sr = torchaudio.load(str(path), normalize=False)  
    y = y.squeeze(0).float().numpy()  
    return y, sr


def soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    if t <= 0:
        return x
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

# sigma_n ≈ median(|d1|)/0.6745
def estimate_sigma_noise(detail_finest: np.ndarray) -> float:
    mad = np.median(np.abs(detail_finest - np.median(detail_finest)))
    sigma = mad / 0.6745 if mad > 0 else np.std(detail_finest)
    return float(max(sigma, 1e-12))


def bayesshrink_denoise_1d(x: np.ndarray, wavelet="db8", level=None) -> np.ndarray:
    # Wavelet decomposition
    coeffs = pywt.wavedec(x, wavelet=wavelet, mode="symmetric", level=level)
    cA, cD_list = coeffs[0], coeffs[1:]

    # Noise sigma from the finest detail band (last in list)
    sigma_n = estimate_sigma_noise(cD_list[-1])

    # Threshold each detail subband separately
    new_details = []
    for d in cD_list:
        var_y = np.var(d)
        sigma_x = np.sqrt(max(var_y - sigma_n**2, 0.0))
        # If sigma_x == 0 (band mostly noise), heavy shrink
        T = (sigma_n**2 / (sigma_x + 1e-12)) if sigma_x > 0 else np.max(np.abs(d))
        new_details.append(soft_threshold(d, T))

    # Reconstruct
    denoised = pywt.waverec([cA] + new_details, wavelet=wavelet, mode="symmetric")
    if len(denoised) != len(x):
        denoised = denoised[:len(x)]
    return denoised


def bayesshrink_cycle_spinning(x: np.ndarray, wavelet="db8", level=None, spins=8) -> np.ndarray:
    if spins <= 0:
        return bayesshrink_denoise_1d(x, wavelet, level)
    N = len(x)
    acc = np.zeros_like(x, dtype=np.float64)
    for s in range(spins):
        # circular shift
        x_shift = np.roll(x, s)
        y_shift = bayesshrink_denoise_1d(x_shift, wavelet, level)
        # inverse shift
        y_unshift = np.roll(y_shift, -s)
        acc += y_unshift
    y_avg = acc / float(spins)
    return y_avg.astype(x.dtype, copy=False)


def pick_level(x_len: int, wavelet: str) -> int:
    w = pywt.Wavelet(wavelet)
    return pywt.dwt_max_level(x_len, w.dec_len)


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    eps = 1e-12
    e = clean - test
    return 10.0 * np.log10((np.sum(clean**2) + eps) / (np.sum(e**2) + eps))


def denoise_wavelet_bayeshrink(input_wav: str):
    noisy, sr = read_wav_mono(Path(input_wav))
    lvl = LEVEL

    denoised = bayesshrink_cycle_spinning(
        noisy.astype(np.float32, copy=False),
        wavelet=WAVELET,
        level=lvl,
        spins=CYCLE_SPINS
    )
    return denoised