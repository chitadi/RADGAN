import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import json

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.distributed import init_process_group

from env import AttrDict, build_env
from mel_dataset import MelDataset, mel_spectrogram, get_dataset_filelist
from network import Generator
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from auraloss.freq import MultiResolutionSTFTLoss

torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{}'.format(rank))

    generator = Generator(h).to(device)

    # Multi-resolution STFT loss (4.3)
    mrstft = MultiResolutionSTFTLoss(
        fft_sizes=[256, 512, 1024],
        hop_sizes=[64, 128, 256],
        win_lengths=[256, 512, 1024],
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
    ).to(device)
    mrstft_weight = float(getattr(h, "mrstft_weight", 1.0))

    # HF-weight mask for mel loss (4.2)
    hf_weights = None
    hf_mel_weight = float(getattr(h, "hf_mel_weight", 1.0))
    if hf_mel_weight > 1.0:
        cutoff = float(getattr(h, "hf_cutoff_hz", 1000.0))
        fmax_loss = h.fmax_for_loss if h.fmax_for_loss is not None else h.fmax
        mel_freqs = librosa.mel_frequencies(
            n_mels=h.num_mels, fmin=h.fmin, fmax=fmax_loss
        )
        w = np.ones_like(mel_freqs, dtype=np.float32)
        w[mel_freqs >= cutoff] = hf_mel_weight
        hf_weights = torch.from_numpy(w)[None, :, None].to(device)  # (1, n_mels, 1)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
    else:
        cp_g = None

    steps = 0
    last_epoch = -1
    if cp_g is not None:
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    def keep_task(clean_paths, task_name="Task1"):
        return [p for p in clean_paths if f"{os.sep}{task_name}{os.sep}" in p]

    training_filelist = keep_task(training_filelist, task_name="Task1")
    validation_filelist = keep_task(validation_filelist, task_name="Task1")

    if not training_filelist or not validation_filelist:
        raise RuntimeError("No training/validation files left after filtering to Task1")

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            # HF-weighted mel loss (4.2)
            if hf_weights is not None:
                diff = (y_mel - y_g_hat_mel).abs()
                loss_mel = (hf_weights * diff).mean() * 45.0
            else:
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45.0

            # MR-STFT waveform loss (4.3)
            loss_mrstft = mrstft(y_g_hat, y)

            loss_gen_all = loss_mel + mrstft_weight * loss_mrstft

            optim_g.zero_grad()
            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % a.validation_interval == 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print("Initializing RAD-GAN Phase 1 pretraining..")

    base_dir = os.path.dirname(__file__)

    a = AttrDict(
        group_name=None,
        input_wavs_dir=os.path.join(base_dir, "..", "..", "..", "dataset", "Task1", "Clean"),
        input_mels_dir="",              # unused in our setup
        input_training_file="",         # ignored by get_dataset_filelist
        input_validation_file="",       # ignored by get_dataset_filelist
        checkpoint_path=os.path.join(base_dir, "checkpoints_pretrain"),
        config=os.path.join(base_dir, "config.json"),
        training_epochs=200,
        stdout_interval=5,
        checkpoint_interval=1000,
        summary_interval=100,
        validation_interval=1000,
        fine_tuning=False,
    )

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, "config.json", a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = 1
    else:
        h.num_gpus = 0

    train(0, a, h)


if __name__ == "__main__":
    main()
