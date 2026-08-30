import os
import argparse
import json

import numpy as np
import torch
from scipy.io.wavfile import write

from .env import AttrDict
from .mel_dataset import MAX_WAV_VALUE
from .network import Generator
from .utils import scan_checkpoint, load_checkpoint

h = None
device = None


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    filelist = os.listdir(a.input_mels_dir)
    os.makedirs(a.output_dir, exist_ok=True)

    with torch.no_grad():
        for i, filename in enumerate(filelist):
            x = np.load(os.path.join(a.input_mels_dir, filename))
            x = torch.FloatTensor(x).to(device)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated_e2e.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
