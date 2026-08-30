#!/usr/bin/env python3
from scipy.fftpack import fft, ifft
import scipy.io.wavfile as wav
import scipy.signal as sg
import numpy as np
import matplotlib.pyplot as plt

MAX_WAV_VALUE = 32768.0

def halfwave_rectification(array):
    """
    Function that computes the half wave rectification with a threshold of 0.
    
    Input :
        array : 1D np.array, Temporal frame
    Output :
        halfwave : 1D np.array, Half wave temporal rectification
        
    """
    return np.maximum(array, 0.0)

    


class Wiener:
    """
    Class made for wiener filtering based on the article "Improved Signal-to-Noise Ratio Estimation for Speech
    Enhancement".

    Reference :
        Cyril Plapous, Claude Marro, Pascal Scalart. Improved Signal-to-Noise Ratio Estimation for Speech
        Enhancement. IEEE Transactions on Audio, Speech and Language Processing, Institute of Electrical
        and Electronics Engineers, 2006.
        
    """

    def __init__(self, WAV_SOURCE, *T_NOISE):
        self.WAV_FILE = WAV_SOURCE  # keep for backward compat

        # Normalise T_NOISE shape: either (start, end) or ((start, end),)
        if len(T_NOISE) == 1 and isinstance(T_NOISE[0], (tuple, list, np.ndarray)):
            self.T_NOISE = tuple(T_NOISE[0])
        else:
            self.T_NOISE = tuple(T_NOISE)

        if isinstance(WAV_SOURCE, str):
            # Original behaviour: read "<basename>.wav" from disk
            self.FS, self.x = wav.read(self.WAV_FILE + ".wav")
        elif isinstance(WAV_SOURCE, tuple) and len(WAV_SOURCE) == 2:
            # New behaviour: in-memory signal
            sig, fs = WAV_SOURCE
            self.FS = int(fs)
            self.x = sig
            # self.x = sig * MAX_WAV_VALUE
        else:
            raise TypeError(
                "WAV_SOURCE must be either a basename string or a (signal, fs) tuple."
            )

        # Work in float, but KEEP the original PCM scale (no normalization to [-1,1])
        if self.x.ndim == 1:
            self.x = self.x[:, None]  # (N,) -> (N,1)
        self.x = self.x.astype(np.float64)


        # Analysis params
        self.FRAME = int(0.032 * self.FS)          # 32 ms as per comment
        self.SHIFT = 0.25                           # 50% overlap
        self.OFFSET = int(self.SHIFT * self.FRAME) # hop in samples

        # Make sure NFFT >= frame length (avoid truncation)
        self.NFFT = 1 << int(np.ceil(np.log2(self.FRAME)))

        # Hann window and Ew = sum(w^2) (used for SNR estimate only)
        self.WINDOW = sg.windows.hann(self.FRAME, sym=False)
        self.EW = np.sum(self.WINDOW**2)

        self.channels = np.arange(self.x.shape[1])
        length = self.x.shape[0]
        self.frames = np.arange((length - self.FRAME) // self.OFFSET + 1)

        # Noise PSD
        self.Sbb = self.welchs_periodogram()

    @staticmethod
    def a_posteriori_gain(SNR):
        """
        Function that computes the a posteriori gain G of Wiener filtering.
        
            Input :
                SNR : 1D np.array, Signal to Noise Ratio
            Output :
                G : 1D np.array, gain G of Wiener filtering
                
        """
        G = (SNR - 1)/SNR
        return G

    @staticmethod
    def a_priori_gain(SNR):
        """
        Function that computes the a priori gain G of Wiener filtering.
        
            Input :
                SNR : 1D np.array, Signal to Noise Ratio
            Output :
                G : 1D np.array, gain G of Wiener filtering
                
        """
        G = SNR/(SNR + 1)
        return G

    def welchs_periodogram(self):
        """
        Estimation of the Power Spectral Density (Sbb) of the stationnary noise
        with Welch's periodogram given prior knowledge of n_noise points where
        speech is absent.
        
            Output :
                Sbb : 1D np.array, Power Spectral Density of stationnary noise
                
        """
        # Initialising Sbb
        Sbb = np.zeros((self.NFFT, self.channels.size))

        self.N_NOISE = int(self.T_NOISE[0]*self.FS), int(self.T_NOISE[1]*self.FS)
        # Number of frames used for the noise
        noise_frames = np.arange(((self.N_NOISE[1] -  self.N_NOISE[0])-self.FRAME) // self.OFFSET + 1)
        for channel in self.channels:
            for frame in noise_frames:
                i_min, i_max = frame*self.OFFSET + self.N_NOISE[0], frame*self.OFFSET + self.FRAME + self.N_NOISE[0]
                x_framed = self.x[i_min:i_max, channel]*self.WINDOW
                X_framed = fft(x_framed, self.NFFT)
                Sbb[:, channel] = frame * Sbb[:, channel] / (frame + 1) + np.abs(X_framed)**2 / (frame + 1)
        return Sbb

    def moving_average(self):
        # Initialising Sbb
        Sbb = np.zeros((self.NFFT, self.channels.size))
        # Number of frames used for the noise
        noise_frames = np.arange((self.N_NOISE - self.FRAME) + 1)
        for channel in self.channels:
            for frame in noise_frames:
                x_framed = self.x[frame:frame + self.FRAME, channel]*self.WINDOW
                X_framed = fft(x_framed, self.NFFT)
                Sbb[:, channel] += np.abs(X_framed)**2
        return Sbb/noise_frames.size

    def wiener(self, return_int16: bool = False, save_path: str | None = None):
        """
        Return the Wiener-filtered waveform in the SAME amplitude scale as input.
        No peak normalization. OLA normalization fixes window overlap energy.
        """
        eps = 1e-12
        s_est = np.zeros_like(self.x, dtype=np.float64)
        ola_norm = np.zeros_like(self.x, dtype=np.float64)

        for channel in self.channels:
            for frame in self.frames:
                i_min = frame * self.OFFSET
                i_max = i_min + self.FRAME
                x_framed = self.x[i_min:i_max, channel] * self.WINDOW

                X = fft(x_framed, self.NFFT)

                # a-posteriori SNR (scaled with Ew to account for window energy)
                SNR_post = (np.abs(X)**2 / (self.EW + eps)) / (self.Sbb[:, channel] + eps)

                # a-priori gain
                G = Wiener.a_priori_gain(SNR_post)
                S = G * X

                # iFFT and OLA with window; compensate later by sum(w^2)
                s_time = np.real(ifft(S))[:self.FRAME]
                s_est[i_min:i_max, channel] += s_time * self.WINDOW
                ola_norm[i_min:i_max, channel] += (self.WINDOW**2)
        # Avoid division by zero at tails
        s_est = s_est / np.maximum(ola_norm, eps)

        if save_path is not None:
            out = s_est.copy()
            out_clip = np.clip(out, -32768, 32767).astype(np.int16)
            wav.write(save_path, self.FS, out_clip)

        if return_int16:
            return np.clip(s_est, -32768, 32767).astype(np.int16)
        return s_est



    def wiener_two_step(self, return_int16: bool = False, save_path: str | None = None):
        """
        Two-step NR (Wiener → decision-directed → TSNR), unnormalized waveform.
        """
        eps = 1e-12
        beta = 0.98

        s_est = np.zeros_like(self.x, dtype=np.float64)
        ola_norm = np.zeros_like(self.x, dtype=np.float64)

        # Store previous S for decision-directed (init zeros)
        S_prev = np.zeros((self.NFFT,), dtype=np.complex128)

        for channel in self.channels:
            S_prev[:] = 0.0j
            for frame in self.frames:
                i_min = frame * self.OFFSET
                i_max = i_min + self.FRAME
                x_framed = self.x[i_min:i_max, channel] * self.WINDOW

                X = fft(x_framed, self.NFFT)

                # Step 1: Wiener
                SNR_post = (np.abs(X)**2 / (self.EW + eps)) / (self.Sbb[:, channel] + eps)
                G = Wiener.a_priori_gain(SNR_post)
                S_w = G * X

                # Step 2: Decision-directed a-priori SNR
                SNR_dd_prio = beta * (np.abs(S_prev)**2) / (self.Sbb[:, channel] + eps) \
                            + (1 - beta) * halfwave_rectification(SNR_post - 1.0)
                G_dd = Wiener.a_priori_gain(SNR_dd_prio)
                S_dd = G_dd * X

                # Step 3: TSNR
                SNR_tsnr_prio = (np.abs(S_dd)**2) / (self.Sbb[:, channel] + eps)
                G_tsnr = Wiener.a_priori_gain(SNR_tsnr_prio)
                S_tsnr = G_tsnr * X

                # iFFT + OLA with window; compensate by sum(w^2)
                s_time = np.real(ifft(S_tsnr))[:self.FRAME]
                s_est[i_min:i_max, channel] += s_time * self.WINDOW
                ola_norm[i_min:i_max, channel] += (self.WINDOW**2)

                # Update previous spectrum for DD
                S_prev = S_w
        

        # s_est = s_est / np.maximum(ola_norm, eps)
        den = np.maximum(ola_norm, eps)
        s_est = s_est / den

        # kill samples where normalization is unreliable (very small overlap)
        thr = 1e-3 * np.max(den)
        edge_mask = den < thr
        s_est[edge_mask] = 0.0
        # s_est = s_est / MAX_WAV_VALUE
        
        if save_path is not None:
            out = s_est.copy()
            out_clip = np.clip(out, -32768, 32767).astype(np.int16)
            wav.write(save_path, self.FS, out_clip)

        if return_int16:
            return np.clip(s_est, -32768, 32767).astype(np.int16)
        return s_est

