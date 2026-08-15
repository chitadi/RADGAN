#!/usr/bin/env python3
from .wiener_filter import Wiener
from .enhance_harmonics import enhance_low_harmonics_spectral

NOISE_BEGIN = 0.0
NOISE_END   = 0.2

def preprocessing_weiner_with_harmonics(y_rec,sr_rec):
    # y_rec, sr_rec = load_mono(RECORDED_WAV_PATH, target_sr=None)

    y_lowharm, S_orig, S_enh, f0_track = enhance_low_harmonics_spectral(
        y_rec,
        sr_rec,
        n_fft=512,
        hop_length=128,
        f0_min=70.0,
        f0_max=300.0,
        max_freq=1000.0,
        num_harmonics=4,
        bw_hz=40.0,
        alpha=0.5,
    )

    wnr = Wiener((y_lowharm,sr_rec), NOISE_BEGIN, NOISE_END)
    enhanced_signal = wnr.wiener_two_step(save_path=None)

    y_enh =enhanced_signal.squeeze(1)

    return y_enh
