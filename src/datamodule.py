

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
import noise_reduction as nr
import scipy.io.wavfile as wav

# DATASET_FOLDER = "/data/"
DATASET_FOLDER = os.path.join(os.path.dirname(__file__), "..", "dataset")
class WavPairDataset(Dataset):
    def __init__(self, recorded_wav_filepaths,  clean_wav_filepaths, task, length_sec, 
                 amp_norm="rms", percentile=95, eps=1e-8,
                 energy_crop=False, energy_threshold=0.05, random_crop=False, 
                 crop_jitter=0.5, fixed_window=True, fixed_start_sec=0.0):
        self.recorded_wav_filepaths = recorded_wav_filepaths
        self.clean_wav_filepaths = clean_wav_filepaths
        self.task = task
        self.length_sec = length_sec
        self.amp_norm = amp_norm
        self.percentile = percentile
        self.eps = eps
        self.energy_crop = energy_crop
        self.energy_threshold = energy_threshold
        self.random_crop = random_crop
        self.crop_jitter = crop_jitter
        self.fixed_window = fixed_window
        self.fixed_start_sec = fixed_start_sec
        assert len(self.recorded_wav_filepaths) == len(self.clean_wav_filepaths) 
        assert len(self.recorded_wav_filepaths) > 0
    
    def _compute_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        flat = tensor.abs().flatten()
        if self.amp_norm == "percentile":
            k = max(1, int(math.ceil(len(flat) * self.percentile / 100.0)))
            scale = flat.kthvalue(k).values
        else:  # default RMS
            scale = torch.sqrt(torch.mean(tensor.pow(2)) + self.eps)
        return scale.clamp_min(self.eps)
    
    def _active_center(self, x):
        # Torch path
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return 0
            peak = float(x.abs().max())
            if not np.isfinite(peak) or peak < self.eps:
                return 0
            thr = self.energy_threshold * peak
            idx = (x.abs() >= thr).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                return 0
            return int((idx[0] + idx[-1]).item() // 2)
        # NumPy path
        x = np.asarray(x)
        if x.size == 0:
            return 0
        peak = float(np.abs(x).max())
        if not np.isfinite(peak) or peak < self.eps:
            return 0
        thr = self.energy_threshold * peak
        idx = np.flatnonzero(np.abs(x) >= thr)
        if idx.size == 0:
            return 0
        return int((idx[0] + idx[-1]) // 2)
    
    def _extract_window(self, recorded, clean, sample_length):
        if len(recorded) <= sample_length:
            return recorded[:sample_length], clean[:sample_length]
        
        # change knobs in init
        if getattr(self, "fixed_window", False):
            start = int(round(self.fixed_start_sec * self.fs))
            max_start = max(len(recorded) - sample_length, 0)
            start = max(0, min(start, max_start))
            end = start + sample_length
            return recorded[start:end], clean[start:end]

        center = self._active_center(recorded) if self.energy_crop else len(recorded) // 2

        if self.random_crop:
            max_start = max(len(recorded) - sample_length, 0)
            jitter = int(self.crop_jitter * self.fs) if self.crop_jitter else sample_length // 2
            jitter = max(1, jitter)

            low = max(0, min(max_start, center - jitter))
            high = max(low, min(max_start, center + jitter))

            if high == low:
                start = low
            else:
                worker_info = torch.utils.data.get_worker_info()
                rng = np.random.default_rng() if worker_info is None else np.random.default_rng(worker_info.seed)
                start = int(rng.integers(low, high + 1))
        else:
            start = max(0, min(center - sample_length // 2, len(recorded) - sample_length))

        end = start + sample_length
        return recorded[start:end], clean[start:end]
    
    def __getitem__(self, idx):
        

        # recorded, recorded_fs = sf.read(self.recorded_wav_filepaths[idx], dtype=np.float32)
        # clean, clean_fs       = sf.read(self.clean_wav_filepaths[idx], dtype=np.float32)
        
        recorded, recorded_fs = torchaudio.load(self.recorded_wav_filepaths[idx], normalize=False)
        clean, clean_fs       = torchaudio.load(self.clean_wav_filepaths[idx], normalize=False)

        # recorded = recorded.squeeze(-1).to(torch.float32)
        clean    = clean.squeeze(0).to(torch.float32)

        noise_begin, noise_end = 0.0, 0.75
        w = nr.Wiener(self.recorded_wav_filepaths[idx], noise_begin, noise_end)
        rec_np = w.wiener_two_step(save_path=None) # (N, 1), float64
        recorded = torch.from_numpy(rec_np.squeeze(-1).astype(np.float32)) # (N,)

        assert recorded_fs == clean_fs
        fs = clean_fs

        if getattr(self, "fs", None) is None:
            self.fs = fs  # cache once for helper methods that need it

        # temporarily cut off to align later
        recorded = recorded[:len(clean)]

        sample_length = int(self.length_sec * fs)
        recorded, clean = self._extract_window(recorded, clean, sample_length)

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
        scale_recorded = torch.tensor(scale_val, dtype=recorded_tensor.dtype)
        # scale_clean = torch.tensor(max(clean_tensor.abs().max().item(), 1e-8), dtype=clean_tensor.dtype)
        # scale = torch.tensor(scale_val, dtype=recorded_tensor.dtype)
        scale_clean = scale_recorded

        recorded_tensor = recorded_tensor / scale_recorded
        clean_tensor = clean_tensor / scale_recorded
        
        return {
            "recorded": recorded_tensor,
            "clean": clean_tensor,
            "scale_recorded": scale_recorded,
            "scale_clean": scale_recorded,
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
        length_sec,
        num_workers=0,
        pin_memory=False,
        crop_fixed_on=None, 
        fixed_start_sec=0.0
        ):
        super().__init__()


        self.batch_size = batch_size
        self.length_sec = length_sec
        # self.pairs = self.get_file_paths()
        self.dataset = {"train": [], "val": [], "test": []}
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.crop_fixed_on = set(crop_fixed_on or []) # e.g., {"train"}, {"train","val"}
        self.fixed_start_sec = float(fixed_start_sec)
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
        
        # start fresh every setup call so append() always works
        self.dataset = {"train": [], "val": [], "test": []}

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
                
                # self.dataset[mode] += WavPairDataset(recorded_wav_filepaths, 
                #     clean_wav_filepaths, 
                #     task=task, 
                #     length_sec=self.length_sec)
                
                fixed = (mode in self.crop_fixed_on) 
                self.dataset[mode].append(
                    WavPairDataset(recorded_wav_filepaths, 
                    clean_wav_filepaths, 
                    task=task, 
                    length_sec=self.length_sec,
                    random_crop=(mode == "train") and not fixed,
                    energy_crop=(mode != "train") and not fixed,
                    fixed_window=fixed,
                    fixed_start_sec=self.fixed_start_sec,)
                )
                
        # after all tasks/modes have been loaded
        for mode in ("train", "val"):
            datasets = self.dataset[mode]
            if len(datasets) == 0:
                raise RuntimeError(f"No {mode} data found.")
            if len(datasets) == 1:
                self.dataset[mode] = datasets[0]
            else:
                self.dataset[mode] = ConcatDataset(datasets)


                # if mode == "train":
                #     self.dataset["train"] 
                # elif mode == "val":
                #     self.dataset["val"] = {}
                # get the corresponding clean ones
        # for mode in os.listdir(recorded_folder):
            # self.dataset[mode] = ConcatDataset(self.dataset[mode])
    

        
    def data_loader(self, mode):

        shuffle = True if mode == "train" else False
        
        return torch.utils.data.DataLoader(
            self.dataset[mode], 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,)

    def train_dataloader(self):
        return self.data_loader("train")
    def val_dataloader(self):
        return self.data_loader("val")
    def test_dataloader(self):
        return self.data_loader("test")
