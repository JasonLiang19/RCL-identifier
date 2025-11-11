"""
PyTorch model architectures for RCL prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CNNModel(nn.Module):
    """
    1D CNN model for RCL prediction.
    Improved over the old version with residual connections and better architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        channels: list = [256, 128, 64, 32],
        kernel_sizes: list = [7, 5, 5, 3],
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        """
        Args:
            input_dim: Dimension of input encoding
            channels: List of channel sizes for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout: Dropout rate
            num_classes: Number of output classes (2 for binary RCL/non-RCL)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build convolutional layers
        layers = []
        in_channels = input_dim
        
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            layers.append(ConvBlock(
                in_channels, out_channels, kernel_size, dropout
            ))
            in_channels = out_channels
            
        self.conv_layers = nn.ModuleList(layers)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv1d(channels[-1], num_classes, kernel_size=1),
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, max_length, input_dim)
            
        Returns:
            Output tensor of shape (batch, max_length, num_classes)
        """
        # Transpose for Conv1d: (batch, channels, length)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Output
        x = self.output(x)
        
        # Transpose back: (batch, length, num_classes)
        x = x.transpose(1, 2)
        
        return x


class ConvBlock(nn.Module):
    """Convolutional block with batch norm, activation, and dropout."""
    
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResidualConvBlock(nn.Module):
    """Convolutional block with residual connection."""
    
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class UNetModel(nn.Module):
    """
    U-Net architecture for RCL prediction with attention gates.
    """
    
    def __init__(
        self,
        input_dim: int,
        base_filters: int = 32,
        depth: int = 4,
        dropout: float = 0.25,
        attention: bool = True,
        num_classes: int = 2
    ):
        """
        Args:
            input_dim: Dimension of input encoding
            base_filters: Number of filters in first layer
            depth: Number of down/up sampling steps
            dropout: Dropout rate
            attention: Whether to use attention gates
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.depth = depth
        self.attention = attention
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, base_filters, kernel_size=1)
        
        # Encoder (contracting path)
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        in_ch = base_filters
        for i in range(depth):
            out_ch = base_filters * (2 ** min(i, 2))  # Cap at 4x base_filters
            self.encoder_blocks.append(
                DoubleConv(in_ch, out_ch, dropout)
            )
            self.downsample.append(nn.AvgPool1d(2))
            in_ch = out_ch
            
        # Bottleneck
        self.bottleneck = DoubleConv(in_ch, in_ch * 2, dropout)
        
        # Decoder (expanding path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if attention else None
        
        # After bottleneck, we have in_ch * 2 channels
        decoder_in_ch = in_ch * 2
        
        for i in range(depth):
            # Skip connection comes from encoder (in reverse order)
            encoder_idx = depth - i - 1
            skip_ch = base_filters * (2 ** min(encoder_idx, 2))
            
            # Output channels for this decoder level
            out_ch = base_filters * (2 ** max(encoder_idx - 1, 0)) if encoder_idx > 0 else base_filters
            
            self.upsample.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))
            
            if attention:
                # AttentionGate(gate_channels, skip_channels)
                # gate is from decoder (decoder_in_ch), skip is from encoder (skip_ch)
                self.attention_gates.append(
                    AttentionGate(decoder_in_ch, skip_ch)
                )
            
            # After concatenation: decoder_in_ch + skip_ch
            self.decoder_blocks.append(
                DoubleConv(decoder_in_ch + skip_ch, out_ch, dropout)
            )
            
            # Next decoder input is the output of this block
            decoder_in_ch = out_ch
            
        # Output
        self.output = nn.Sequential(
            nn.Conv1d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_filters, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, max_length, input_dim)
            
        Returns:
            Output tensor of shape (batch, max_length, num_classes)
        """
        # Transpose for Conv1d: (batch, channels, length)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Encoder
        encoder_features = []
        for encode, downsample in zip(self.encoder_blocks, self.downsample):
            x = encode(x)
            encoder_features.append(x)
            x = downsample(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, (upsample, decode) in enumerate(zip(self.upsample, self.decoder_blocks)):
            x = upsample(x)
            
            # Get skip connection
            skip = encoder_features[-(i+1)]
            
            # Match sizes if needed
            if x.size(2) != skip.size(2):
                x = F.interpolate(x, size=skip.size(2), mode='linear', align_corners=True)
            
            # Apply attention gate
            if self.attention:
                skip = self.attention_gates[i](skip, x)
            
            # Concatenate
            x = torch.cat([skip, x], dim=1)
            x = decode(x)
            
        # Output
        x = self.output(x)
        
        # Transpose back: (batch, length, num_classes)
        x = x.transpose(1, 2)
        
        return x


class DoubleConv(nn.Module):
    """Double convolution block (Conv -> BN -> ReLU -> Conv -> BN -> ReLU)."""
    
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class AttentionGate(nn.Module):
    """Attention gate for U-Net skip connections."""
    
    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        
        if inter_channels is None:
            inter_channels = skip_channels // 2
            
        self.W_gate = nn.Conv1d(gate_channels, inter_channels, kernel_size=1)
        self.W_skip = nn.Conv1d(skip_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv1d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, skip, gate):
        """
        Args:
            skip: Skip connection from encoder
            gate: Gating signal from decoder
        """
        # Match sizes
        if gate.size(2) != skip.size(2):
            gate = F.interpolate(gate, size=skip.size(2), mode='linear', align_corners=True)
            
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        
        attention = self.relu(g + s)
        attention = self.psi(attention)
        attention = self.sigmoid(attention)
        
        return skip * attention


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM model for RCL prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        """
        Args:
            input_dim: Dimension of input encoding
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, max_length, input_dim)
            
        Returns:
            Output tensor of shape (batch, max_length, num_classes)
        """
        # LSTM expects (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Apply fully connected layers
        out = self.fc(lstm_out)
        
        return out


def get_model(
    model_type: str,
    input_dim: int,
    config: dict,
    num_classes: int = 2
) -> nn.Module:
    """
    Factory function to get the appropriate model.
    
    Args:
        model_type: Type of model (cnn, unet, lstm)
        input_dim: Dimension of input encoding
        config: Model configuration dictionary
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    model_type = model_type.lower()
    
    if model_type == "cnn":
        return CNNModel(
            input_dim=input_dim,
            channels=config.get('channels', [256, 128, 64, 32]),
            kernel_sizes=config.get('kernel_sizes', [7, 5, 5, 3]),
            dropout=config.get('dropout', 0.3),
            num_classes=num_classes
        )
    elif model_type == "unet":
        return UNetModel(
            input_dim=input_dim,
            base_filters=config.get('base_filters', 32),
            depth=config.get('depth', 4),
            dropout=config.get('dropout', 0.25),
            attention=config.get('attention', True),
            num_classes=num_classes
        )
    elif model_type == "lstm":
        return LSTMModel(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.3),
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
