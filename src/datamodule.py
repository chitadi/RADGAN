

import os
import soundfile as sf
import torch
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
import pytorch_lightning as pl
from loguru import logger
from typing import Dict
from tqdm import tqdm
import numpy as np
import librosa
import torchaudio
from utils import safe_open_yaml
import math
from loguru import logger
from dwt import unify

DATASET_FOLDER = "/data/"

class WavPairDataset(Dataset):
    def __init__(self, recorded_wav_filepaths,  clean_wav_filepaths, task, length_sec, wavelet, level, tag):
        self.recorded_wav_filepaths = recorded_wav_filepaths
        self.clean_wav_filepaths = clean_wav_filepaths
        self.task = task
        self.length_sec = length_sec
        # specifically for wavelets
        self.wavelet = wavelet
        self.level = level
        self.tag = tag
        assert len(self.recorded_wav_filepaths) == len(self.clean_wav_filepaths) 
        assert len(self.recorded_wav_filepaths) > 0
    def __getitem__(self, idx):
        

        recorded, recorded_fs = sf.read(self.recorded_wav_filepaths[idx], dtype=np.float32)
        clean, clean_fs       = sf.read(self.clean_wav_filepaths[idx], dtype=np.float32)


        assert recorded_fs == clean_fs
        fs = clean_fs


        # temporarily cut off to align later
        recorded = recorded[:len(clean)]
        # print(f"After alignment - Recorded: {recorded.shape}, Clean: {clean.shape}")
        assert len(recorded) == len(clean)

        sample_length = self.length_sec * fs
        start_time = 1

        recorded_padded = np.zeros(sample_length, dtype=np.float32)
        clean_padded    = np.zeros(sample_length, dtype=np.float32)

        # if len(recorded) > sample_length:
        # if len(recorded) > sample_length + start_time * fs:
            # recorded_padded = recorded[:sample_length]
        recorded_padded = recorded[start_time * fs : sample_length + start_time * fs]
        # else:
        #     # recorded_padded[:len(recorded)] = recorded
        #     recorded_padded[start_time * fs : len(recorded)] = recorded


        # if len(clean) > sample_length:
        # if len(clean) > sample_length + start_time * fs:
            # clean_padded = clean[:sample_lenth]
        clean_padded = clean[start_time * fs : sample_length + start_time * fs]
        # else:
        #     # clean_padded = clean[:len(clean)]
        #     clean_padded[start_time * fs : len(clean)] = clean

        #  call the dwt function here and then add it to this dictionary for returning
        features, scale = unify(recorded_padded, wavelet=self.wavelet, level=self.level, tag=self.tag)
        clean_padded = clean_padded.astype(np.float32) / scale
        features = torch.from_numpy(features).float()
        target   = torch.from_numpy(clean_padded).float().unsqueeze(0)
        return {"recorded": features, "clean": target, "fs": fs, 
            "task": self.task
        }

    def __len__(self):
        return len(self.recorded_wav_filepaths)

class DataModule(pl.LightningDataModule):
    def __init__(self, 
        batch_size,
        length_sec, 
        wavelet="db1", 
        level=3, 
        tag="interpolation"
        ):
        super().__init__()

        # for wavelets
        self.wavelet = wavelet
        self.level = level
        self.tag = tag
        # normal processing
        self.batch_size = batch_size
        self.length_sec = length_sec
        # self.pairs = self.get_file_paths()
        self.dataset = {"train": [], "val": [], "test": []}
    # def print_summary(self):

    #     for mode in self.modes:
    #         # print(f"{mode}: {len(self.speech_data_points[mode])} speech, and {len(self.noise_data_points[mode])} noise data_points")
    #         print(f"{mode}: {len(self.speech_data_points[mode])} speech, and {len(self.ir_data_points[mode])} impulse response data_points")

    def setup(self, stage=None):
        logger.info(f"Loading dataset for {stage}")

        if stage == "fit":
            self.load_train_dev()
        
    def load_train_dev(self):
        
        assert os.path.exists(DATASET_FOLDER)

        task_folders = os.listdir(DATASET_FOLDER)
        logger.info(f"The available tasks are {task_folders}")
        logger.info(f"Loading dataset")
        
        
        for task in task_folders:
            
            logger.info(f"{task} contains:")
            folder_path = os.path.join(DATASET_FOLDER, task)
            clean_folder = os.path.join(folder_path, "Clean")
            recorded_folder = os.path.join(folder_path, "Recorded")

            for mode in os.listdir(recorded_folder):
                
                
                folder_path = os.path.join(recorded_folder, mode)
                files = os.listdir(folder_path)
                
                # filter all recorded files with ext ".wav" 
                is_wav = lambda filename: os.path.splitext(filename)[-1] == ".wav" 
                wav_filenames = [_file for _file in files if is_wav(_file)]  
                recorded_wav_filepaths = [os.path.join(recorded_folder, mode, _file) for _file in wav_filenames]

                file_exists = [os.path.exists(filepath) for filepath in recorded_wav_filepaths]
                assert all(file_exists)

                # find the pairs in clean folder 
                clean_wav_filenames = [_file.replace("_recorded_aligned", "") for _file in wav_filenames]  
                clean_wav_filepaths = [os.path.join(clean_folder, mode, _file) for _file in clean_wav_filenames]
                file_exists = [os.path.exists(filepath) for filepath in clean_wav_filepaths]

                assert all(file_exists)
                assert len(recorded_wav_filepaths) == len(clean_wav_filepaths)
                
                logger.info(f"{mode}: {len(recorded_wav_filepaths)} wav files.")
                
                self.dataset[mode] += WavPairDataset(recorded_wav_filepaths, 
                    clean_wav_filepaths, 
                    task=task, 
                    length_sec=self.length_sec,
                    wavelet=self.wavelet,
                    level=self.level,
                    tag=self.tag,)

                # if mode == "train":
                #     self.dataset["train"] 
                # elif mode == "val":
                #     self.dataset["val"] = {}
                # get the corresponding clean ones
        # for mode in os.listdir(recorded_folder):
            # self.dataset[mode] = ConcatDataset(self.dataset[mode])
    

        
    def data_loader(self, mode):

        # shuffle = True if mode == "train" or mode == "val" else False
        shuffle = True if mode == "train" else False
        
        return torch.utils.data.DataLoader(
            self.dataset[mode], 
            batch_size=self.batch_size, 
            shuffle=shuffle)

    def train_dataloader(self):
        return self.data_loader("train")
    def val_dataloader(self):
        return self.data_loader("val")
    def test_dataloader(self):
        return self.data_loader("test")

