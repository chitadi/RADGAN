

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

DATASET_FOLDER = "/data/"

class WavPairDataset(Dataset):
    def __init__(self, recorded_wav_filepaths,  clean_wav_filepaths, task, length_sec):
        self.recorded_wav_filepaths = recorded_wav_filepaths
        self.clean_wav_filepaths = clean_wav_filepaths
        self.task = task
        self.length_sec = length_sec
        assert len(self.recorded_wav_filepaths) == len(self.clean_wav_filepaths) 
        assert len(self.recorded_wav_filepaths) > 0
    def __getitem__(self, idx):
        

        recorded, recorded_fs = sf.read(self.recorded_wav_filepaths[idx], dtype=np.float32)
        clean, clean_fs       = sf.read(self.clean_wav_filepaths[idx], dtype=np.float32)


        assert recorded_fs == clean_fs
        fs = clean_fs


        # temporarily cut off to align later
        recorded = recorded[:len(clean)]

        sample_length = self.length_sec * fs

        recorded_padded = np.zeros(sample_length, dtype=np.float32)
        clean_padded    = np.zeros(sample_length, dtype=np.float32)
        
        if len(recorded) > sample_length:
            recorded_padded = recorded[:sample_length]
        else:
            recorded_padded[:len(recorded)] = recorded


        if len(clean) > sample_length:
            clean_padded = clean[:sample_length]
        else:
            clean_padded[:len(clean)] = clean

        # converting to a tensor for new pipeline
        recorded_tensor = torch.from_numpy(recorded_padded)
        clean_tensor = torch.from_numpy(clean_padded)

        # can change this to 1e-10 as well
        scale_val = max(recorded_tensor.abs().max().item(), 1e-8)
        scale = torch.tensor(scale_val, dtype=recorded_tensor.dtype)

        recorded_tensor = recorded_tensor / scale
        clean_tensor = clean_tensor / scale

        return {
            "recorded": recorded_tensor,
            "clean": clean_tensor,
            "scale": scale,
            "fs": fs,
            "task": self.task,
        }
        
        # return {"recorded": recorded_padded, "clean": clean_padded, "fs": fs, 
        #     "task": self.task
        # }

    def __len__(self):
        return len(self.recorded_wav_filepaths)

class DataModule(pl.LightningDataModule):
    def __init__(self, 
        batch_size,
        length_sec
        ):
        super().__init__()


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

        # sorting everything for reproducable results
        task_folders = sorted(os.listdir(DATASET_FOLDER))
        logger.info(f"The available tasks are {task_folders}")
        logger.info(f"Loading dataset")
        
        
        for task in task_folders:
            
            logger.info(f"{task} contains:")
            folder_path = os.path.join(DATASET_FOLDER, task)
            clean_folder = os.path.join(folder_path, "Clean")
            recorded_folder = os.path.join(folder_path, "Recorded")

            for mode in sorted(os.listdir(recorded_folder)):
                
                
                folder_path = os.path.join(recorded_folder, mode)
                files = sorted(os.listdir(folder_path))
                
                # filter all recorded files with ext ".wav" 
                is_wav = lambda filename: os.path.splitext(filename)[-1] == ".wav" 
                wav_filenames = sorted(_file for _file in files if is_wav(_file))  
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
                    length_sec=self.length_sec)

                # if mode == "train":
                #     self.dataset["train"] 
                # elif mode == "val":
                #     self.dataset["val"] = {}
                # get the corresponding clean ones
        # for mode in os.listdir(recorded_folder):
            # self.dataset[mode] = ConcatDataset(self.dataset[mode])
    

        
    def data_loader(self, mode):

        shuffle = True if mode == "train" or mode == "val" else False
        
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
