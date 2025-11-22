
import torch
import torchaudio
from weiner_with_spectral_preprocessing import preprocessing_weiner_with_harmonics
import torchaudio.functional as aF

LR_SAMPLING_RATE = 2000
HR_SAMPLING_RATE = 8000

def preprocess_train(recorded_audio,clean_audio,orig_sampling_rate):
    
    #step 1
    recorded_audio = torch.from_numpy(recorded_audio).unsqueeze(0)
    clean_audio = torch.from_numpy(clean_audio).unsqueeze(0)

    #step 2
    audio_hr = clean_audio
    audio_lr = aF.resample(recorded_audio, orig_freq=orig_sampling_rate, new_freq=LR_SAMPLING_RATE)
    audio_lr = aF.resample(audio_lr, orig_freq=LR_SAMPLING_RATE, new_freq=HR_SAMPLING_RATE)
    audio_lr=audio_lr.squeeze()
    audio_hr=audio_hr.squeeze()

    #step 3
    
    audio_lr = audio_lr.cpu().numpy()
    # audio_lr = preprocessing_weiner_with_harmonics(audio_lr,HR_SAMPLING_RATE)
    audio_lr = torch.from_numpy(audio_lr).to(torch.float32)
    
    return audio_hr,audio_lr