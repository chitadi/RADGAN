import torch
import torchaudio.functional as aF

def amp_pha_stft(audio, n_fft, hop_size, win_size, center=True):

    hann_window = torch.hann_window(win_size).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    log_amp = torch.log(torch.abs(stft_spec)+1e-4)
    pha = torch.angle(stft_spec)

    com = torch.stack((torch.exp(log_amp)*torch.cos(pha), 
                       torch.exp(log_amp)*torch.sin(pha)), dim=-1)

    return log_amp, pha, com


def amp_pha_istft(log_amp, pha, n_fft, hop_size, win_size, center=True):
    
    amp = torch.exp(log_amp)
    com = torch.complex(amp*torch.cos(pha), amp*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    audio = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return audio


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    return training_indexes, validation_indexes