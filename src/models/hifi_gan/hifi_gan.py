import torch
import json
import os
import numpy as np
import librosa
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from typing import Any, Dict, Optional, Tuple
# from utils import init_weights, get_padding
from models.hifi_gan.utils import init_weights, get_padding
from models.hifi_gan.mel_dataset import mel_spectrogram, MAX_WAV_VALUE
from models.hifi_gan.env import AttrDict
from enhance_lower_harmonics_for_model import enhance_low_harmonics_spectral
from noise_reduction_for_gan import Wiener

try:
    from ..base_model import BaseModel
except ImportError:
    from base_model import BaseModel


LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

class MelDiscriminator2D(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1,   32, (3, 3), (1, 1), padding=(1, 1))),
            norm_f(Conv2d(32,  64, (3, 3), (2, 2), padding=(1, 1))),
            norm_f(Conv2d(64, 128, (3, 3), (2, 2), padding=(1, 1))),
            norm_f(Conv2d(128,256, (3, 3), (2, 2), padding=(1, 1))),
        ])
        self.conv_post = norm_f(Conv2d(256, 1, (3, 3), (1, 1), padding=(1, 1)))

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiMelDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MelDiscriminator2D(use_spectral_norm=True),
            MelDiscriminator2D(),
        ])

    def forward(self, y_mel, y_hat_mel):
        # y_mel, y_hat_mel: (B, 1, n_mels, T)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y_mel)
            y_d_g, fmap_g = d(y_hat_mel)
            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    

class HiFiGAN(BaseModel):
    def __init__(self, 
        window_length=512,
        hop_length = 128,
        num_kernels=32,
        # num_heads=8,
        kernel_size=(3,3),
        learning_rate=1e-3,
        verbose=False):

        
        super().__init__()
        self.register_buffer("window", torch.hamming_window(window_length))
        self.loss_function = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.stft_params = dict(
            n_fft=window_length, 
            hop_length=hop_length,
            win_length=window_length,
            # window=self.window, # in the registered_buffer
            center=True,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        self.istft_params = self.stft_params.copy()
        self.istft_params["return_complex"]=False

        self.downsample_blocks = torch.nn.ModuleList()
        self.residual_blocks = torch.nn.Sequential()
        self.upsample_blocks = torch.nn.ModuleList()

        out_kernels_list = [32, 64, 128]
        in_kernels = 1
        kernel_size_list = [(3, 3), (3, 3), (3, 3)]
        padding_list = [1, 1, 1]
        for i, (out_kernels, kernel_size, padding) in enumerate(zip(out_kernels_list, kernel_size_list,padding_list)):
            # # in_kernels = 
            # out_kernels = out_kernels_list[i]
            
            if i ==0:
                self.downsample_blocks.append(DownsampleBlock(in_kernels, out_kernels, kernel_size, stride=2, padding=padding))
            else:
                self.downsample_blocks.append(DownsampleBlock(in_kernels, out_kernels, kernel_size, stride=2, padding=padding))
            
            in_kernels= out_kernels
        

        for i in range(5):
            self.residual_blocks.append(ResidualBlock(out_kernels, out_kernels, kernel_size))
        
        # out_kernels = in_kernels // 2
        # in_kernels = out_kernels *2 # cause of concatenate
        # in_kernels = [12] 
        # out_kernels_list = out_kernels_list[::-1] # reverse order
        out_kernels_list = [64, 32]
        kernel_size_list = [(3, 3), (3, 3)]
        output_padding_list = [0, 1]
        stride_list=[4, 2]
        in_kernels = out_kernels
        for i, (out_kernels, kernel_size, stride, output_padding) in enumerate(zip(out_kernels_list, kernel_size_list, stride_list, output_padding_list)):
                # in_ = in_kernels[i] * 2 
            if i == 1:
                self.upsample_blocks.append(UpsampleBlock(in_kernels, out_kernels, kernel_size, stride=stride, output_padding=output_padding))
            else:
                self.upsample_blocks.append(UpsampleBlock(in_kernels, out_kernels, kernel_size, stride=stride, output_padding=output_padding))
            in_kernels = out_kernels

        self.last_layer = torch.nn.Conv2d(in_kernels, 1, kernel_size, padding="same")
        
            
    def forward(self, x):
        length = x.shape[-1]

        x = self.stft(x) # B, F, T

        # print(x.dtype)
        dc = x[:, 0:1]
        x = x[:, 1:] 
        # dc = x[:, 0:1]
        # x = x[:, 1:] 

        # assert x.shape[1:3] == (128, 128), x.shape

        real = x.real
        imag = x.imag
        phase = x.angle()
        mag = x.abs()
        x = mag
        x = x.unsqueeze(1) # B, 1, F, T
        # for 
        B, _, F, T = x.shape
        
        out = []
        # windowing operation 
        for _x in torch.split(x, F, dim=-1):

            if _x.shape[-1] == F:
                _x = self.forward_in_stft(_x) 
            else: # last split have to be padded
                pad = F - _x.shape[-1]
                _x = torch.nn.functional.pad(_x, (0,pad))
                _x = self.forward_in_stft(_x)
                _x = _x[..., :-pad]
            # print(_x.shape)
            out.append(_x)
        
        x = torch.cat(out, dim=-1)
        x = x.squeeze(1)
        
        # x = torch.stack([ x * torch.cos(phase),  x *torch.sin(phase)], dim=-1)

        x_complex = x * torch.cos(phase) + 1j * x * torch.sin(phase)

        x = torch.cat([dc, x_complex], dim=1)
        # print(self.stft_params)
        # print(x.shape)
        x_stft = x # B, F, T
        # x = x[..., 0] + 1j * x[..., 1]
        
        x = torch.istft(x_stft, length=length, window=self.window, **self.istft_params)
    
        # return x_stft, x
        return x

    def forward_in_stft(self, x):


        # skip_connections = []
        for i, block in enumerate(self.downsample_blocks):
            
            x = block(x)
            if self.verbose:
                print(f"Downsample {i} ", x.shape)
            # skip_connections.append(x)

        for i, block in enumerate(self.residual_blocks):
            x  = block(x)


        for i, block in enumerate(self.upsample_blocks):

            x = block(x)
            if self.verbose:
                print(f"Upsample {i} ", x.shape)
        
        x = self.last_layer(x)
        # x = self.upsample_blocks(x)

        return x
    def loss_function(self, x_hat, x):
        
        # mag_hat = torch.sqrt(x_hat[..., 0]**2 + x_hat[..., 1]**2)
        # mag = torch.sqrt(x[..., 0]**2 + x[..., 1]**2)
        x_hat = self.stft(x_hat)
        x = self.stft(x)
        mag_hat = x_hat.abs()
        mag = x.abs()
        # print(mag.shape)
        # print(mag_hat.shape)

        mag = mag[:, 1:]
        mag_hat = mag_hat[:, 1:]
        

        l2_loss = torch.linalg.norm(torch.log(1+ mag_hat) - torch.log(1+ mag), ord="fro", dim=(-2, -1))
        # print(l2_loss)
        # torch.mean()
        loss = torch.mean(l2_loss)
        # loss = torch.mean((mag_hat - mag)**2)
        # print(loss)
        return loss

    def stft(self, x):
        return torch.stft(x,window=self.window, **self.stft_params)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    
def main():

    model = HiFiGAN(verbose=True)
    batch_size = 32

    sample = torch.zeros(batch_size, 8000 * 10) 
    output = model.forward(sample)
    assert output.shape == sample.shape

if __name__ == "__main__":
    main()