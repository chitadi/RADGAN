import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolutional Block for the Encoder (Downsampling) Stage.
    
    Each block consists of:
    - Two 1D Convolutional layers with ReLU activation
    - One MaxPooling layer for downsampling
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (neurons per layer in the block)
        kernel_size (int): Size of the convolutional kernel (default: 3)
        padding (int): Padding size (default: 1 for same padding)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # Second convolution layer (same number of output channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)
        
        # MaxPooling layer with size 2 for downsampling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """
        Forward pass through the convolutional block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length)
        
        Returns:
            tuple: (pooled_output, skip_connection)
                - pooled_output: Downsampled output for next block
                - skip_connection: Output before pooling for U-Net skip connections
        """
        # First convolution + ReLU
        x = self.conv1(x)
        x = self.relu(x)
        
        # Second convolution + ReLU
        x = self.conv2(x)
        x = self.relu(x)
        
        # Store this for skip connections before pooling
        skip = x
        
        # Apply pooling for downsampling
        pooled = self.pool(x)
        
        return pooled, skip


class BottleneckBlock(nn.Module):
    """
    Bottleneck Block - The deepest part of the U-Net.
    
    This block consists of:
    - Two 1D Convolutional layers with ReLU activation
    - NO pooling layer (maintains feature map size)
    - Dropout layer (50%) to prevent overfitting
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (1024 neurons)
        kernel_size (int): Size of the convolutional kernel (default: 3)
        padding (int): Padding size (default: 1)
        dropout_rate (float): Dropout rate (default: 0.5 for 50%)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.5):
        super(BottleneckBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Dropout layer (50% dropout as specified in paper)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        """
        Forward pass through the bottleneck block.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor with dropout applied
        """
        # First convolution + ReLU
        x = self.conv1(x)
        x = self.relu(x)
        
        # Second convolution + ReLU
        x = self.conv2(x)
        x = self.relu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class DeconvBlock(nn.Module):
    """
    Deconvolutional Block for the Decoder (Upsampling) Stage.
    
    Each block consists of:
    - One ConvTranspose1d layer for upsampling
    - Concatenation with skip connection from encoder
    - Two 1D Convolutional layers with ReLU activation
    
    Args:
        in_channels (int): Number of input channels from previous deconv block
        out_channels (int): Number of output channels (neurons per layer in the block)
        kernel_size (int): Size of the convolutional kernel (default: 3)
        padding (int): Padding size (default: 1)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeconvBlock, self).__init__()
        
        # Transposed convolution for upsampling (2x upsampling)
        self.upconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )
        
        # First convolution after concatenation
        # Input channels = out_channels (from upconv) + out_channels (from skip connection)
        self.conv1 = nn.Conv1d(
            in_channels=out_channels * 2,  # Concatenated channels
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, skip_connection):
        """
        Forward pass through the deconvolutional block.
        
        Args:
            x (torch.Tensor): Input tensor from previous deconv block
            skip_connection (torch.Tensor): Skip connection from corresponding encoder block
        
        Returns:
            torch.Tensor: Upsampled and processed output
        """
        # Upsample using transposed convolution
        x = self.upconv(x)
        
        # Concatenate with skip connection along channel dimension
        # This preserves features that were lost during encoding
        x = torch.cat([x, skip_connection], dim=1)
        
        # First convolution + ReLU
        x = self.conv1(x)
        x = self.relu(x)
        
        # Second convolution + ReLU
        x = self.conv2(x)
        x = self.relu(x)
        
        return x


class AudioUNet(nn.Module):
    """
    1D U-Net Convolutional/Deconvolutional Deep Autoencoder for Audio Restoration.
    
    Architecture Overview:
    - Input: Raw audio waveform (batch_size, 1, 32000)
    - Encoder: 4 convolutional blocks with progressive channel increase (64->128->256->512)
    - Bottleneck: 1024 channels with 50% dropout
    - Decoder: 4 deconvolutional blocks with progressive channel decrease (512->256->128->64)
    - Output: Restored audio waveform (batch_size, 1, 32000)
    
    The model uses skip connections between encoder and decoder blocks to preserve
    fine-grained features that might be lost during the encoding process.
    
    Args:
        in_channels (int): Number of input channels (default: 1 for mono audio)
        out_channels (int): Number of output channels (default: 1 for mono audio)
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(AudioUNet, self).__init__()
        
        # ==================== ENCODER (CONVOLUTIONAL STAGE) ====================
        # Progressive dimensionality reduction with feature extraction
        
        # Block 1: 32000 -> 16000, channels: 1 -> 64
        self.enc1 = ConvBlock(in_channels, 64)
        
        # Block 2: 16000 -> 8000, channels: 64 -> 128
        self.enc2 = ConvBlock(64, 128)
        
        # Block 3: 8000 -> 4000, channels: 128 -> 256
        self.enc3 = ConvBlock(128, 256)
        
        # Block 4: 4000 -> 2000, channels: 256 -> 512
        self.enc4 = ConvBlock(256, 512)
        
        # ==================== BOTTLENECK ====================
        # Deepest layer with maximum compression: 2000 length, 1024 channels
        self.bottleneck = BottleneckBlock(512, 1024, dropout_rate=0.5)
        
        # ==================== DECODER (DECONVOLUTIONAL STAGE) ====================
        # Progressive dimensionality expansion with feature reconstruction
        
        # Block 1: 2000 -> 4000, channels: 1024 -> 512
        self.dec1 = DeconvBlock(1024, 512)
        
        # Block 2: 4000 -> 8000, channels: 512 -> 256
        self.dec2 = DeconvBlock(512, 256)
        
        # Block 3: 8000 -> 16000, channels: 256 -> 128
        self.dec3 = DeconvBlock(256, 128)
        
        # Block 4: 16000 -> 32000, channels: 128 -> 64
        self.dec4 = DeconvBlock(128, 64)
        
        # ==================== OUTPUT LAYER ====================
        # Final convolution to get output with correct number of channels
        # Uses tanh activation to get values in range [-1.0, 1.0] (audio waveform range)
        self.final_conv = nn.Conv1d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1,  # 1x1 convolution for channel adjustment
            stride=1,
            padding=0
        )
        
        # TODO: Consider adding batch normalization if training is unstable
    
    def forward(self, x):
        """
        Forward pass through the entire U-Net architecture.
        
        Data Flow:
        1. Input audio (32000 samples) passes through 4 encoder blocks
        2. Each encoder block extracts features and downsamples by 2x
        3. Bottleneck compresses to essential features (2000 samples, 1024 channels)
        4. Each decoder block upsamples by 2x and concatenates with skip connections
        5. Output layer produces restored audio (32000 samples)
        
        Args:
            x (torch.Tensor): Input audio tensor of shape (batch_size, 1, 32000)
        
        Returns:
            torch.Tensor: Restored audio tensor of shape (batch_size, 1, 32000)
        """
        # ==================== ENCODER PATH ====================
        # Each encoder block returns (downsampled_output, skip_connection)
        
        # Encoder Block 1: 32000 -> 16000
        x1, skip1 = self.enc1(x)  # skip1: (batch, 64, 32000)
        
        # Encoder Block 2: 16000 -> 8000
        x2, skip2 = self.enc2(x1)  # skip2: (batch, 128, 16000)
        
        # Encoder Block 3: 8000 -> 4000
        x3, skip3 = self.enc3(x2)  # skip3: (batch, 256, 8000)
        
        # Encoder Block 4: 4000 -> 2000
        x4, skip4 = self.enc4(x3)  # skip4: (batch, 512, 4000)
        
        # ==================== BOTTLENECK ====================
        # Compressed representation: (batch, 1024, 2000)
        bottleneck_out = self.bottleneck(x4)
        
        # ==================== DECODER PATH ====================
        # Each decoder block upsamples and concatenates with corresponding skip connection
        
        # Decoder Block 1: 2000 -> 4000
        # Concatenates with skip4 from encoder block 4
        dec1_out = self.dec1(bottleneck_out, skip4)
        
        # Decoder Block 2: 4000 -> 8000
        # Concatenates with skip3 from encoder block 3
        dec2_out = self.dec2(dec1_out, skip3)
        
        # Decoder Block 3: 8000 -> 16000
        # Concatenates with skip2 from encoder block 2
        dec3_out = self.dec3(dec2_out, skip2)
        
        # Decoder Block 4: 16000 -> 32000
        # Concatenates with skip1 from encoder block 1
        dec4_out = self.dec4(dec3_out, skip1)
        
        # ==================== OUTPUT LAYER ====================
        # Final convolution to get single channel output
        output = self.final_conv(dec4_out)
        
        # Apply tanh activation to constrain output to [-1.0, 1.0] range
        output = torch.tanh(output)
        
        return output