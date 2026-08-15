

if __name__ == "__main__":
    from base_model import BaseModel
else:
    from .base_model import BaseModel

import torch

class DownsampleBlock(torch.nn.Module):
    def __init__(self, in_kernels, out_kernels, kernel_size, stride, padding):
        super().__init__()
        # self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2d = torch.nn.Conv2d(in_kernels, out_kernels, kernel_size, stride=stride, padding=padding)
        self.bn2d = torch.nn.InstanceNorm2d(out_kernels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x = self.pool(x)
        x = self.conv2d(x)
        x = self.bn2d(x)
        x = self.relu(x)
        
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_kernels, out_kernels, kernel_size):
        super(ResidualBlock, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_kernels, out_kernels, kernel_size, padding="same"),
            torch.nn.InstanceNorm2d(out_kernels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_kernels, out_kernels, kernel_size, padding="same"),
            torch.nn.InstanceNorm2d(out_kernels),
        )

    def forward(self, x):
        identity = x
        x = self.model.forward(x)
        return x + identity
        
class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_kernels, out_kernels, kernel_size, stride, output_padding=0):
        super().__init__()

        self.deconv2d = torch.nn.ConvTranspose2d(in_kernels, out_kernels, kernel_size, stride=stride,padding=0, output_padding=output_padding)
        self.bn2d = torch.nn.InstanceNorm2d(out_kernels)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        
        # x = torch.cat([x, conn], dim=1)

        x = self.deconv2d(x)
        x = self.bn2d(x)
        x = self.relu(x)
        
        
        return x
    

class WaveVoiceNet(BaseModel):
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

    model = WaveVoiceNet(verbose=True)
    batch_size = 32

    sample = torch.zeros(batch_size, 8000 * 10) 
    output = model.forward(sample)
    assert output.shape == sample.shape

if __name__ == "__main__":
    main()