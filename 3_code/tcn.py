# Torch Temporal Convolutional Network (TCN) implementation
# Based on: Bai, Kolter, Koltun (2018) "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Crops the last 'chomp_size' timesteps to preserve causality when padding is used."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A single residual TCN block: Dilated causal Conv1d -> ReLU -> Dropout -> Conv1d -> ReLU -> Dropout + Residual."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal

        conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        if use_weight_norm:
            conv1 = weight_norm(conv1)
            conv2 = weight_norm(conv2)

        self.net = nn.Sequential(
            conv1,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            conv2,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample: Optional[nn.Module] = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.activation = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation."""
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: int,
        dropout: float = 0.0,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        num_levels = len(channels)
        current_in = in_channels
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    current_in,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                )
            )
            current_in = channels[i]
        self.network = nn.Sequential(*layers)
        self.out_channels = channels[-1] if len(channels) > 0 else in_channels

    def forward(self, x_b_l_c: torch.Tensor) -> torch.Tensor:
        # Accept [B, L, C] -> transpose to [B, C, L] for Conv1d
        x = x_b_l_c.transpose(1, 2)
        y = self.network(x)  # [B, C_out, L]
        return y.transpose(1, 2)  # back to [B, L, C_out]


class TCNRegressor(nn.Module):
    """
    Wrapper for sequence regression.
    - If return_sequence=False (default): returns [B, out_dim] using last timestep or global pooling
    - If return_sequence=True: returns [B, L, out_dim]
    """
    def __init__(
        self,
        input_dim: int,
        channels: List[int],
        kernel_size: int,
        dropout: float = 0.0,
        out_dim: int = 1,
        return_sequence: bool = False,
        use_global_pool: bool = False,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            in_channels=input_dim,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
        )
        self.return_sequence = return_sequence
        self.use_global_pool = use_global_pool
        self.head = nn.Linear(self.tcn.out_channels, out_dim)

    def forward(self, x_b_l_c: torch.Tensor) -> torch.Tensor:
        feats = self.tcn(x_b_l_c)  # [B, L, C_tcn]
        if self.return_sequence:
            return self.head(feats)  # [B, L, out_dim]
        if self.use_global_pool:
            pooled = feats.mean(dim=1)  # [B, C_tcn]
        else:
            pooled = feats[:, -1, :]  # last timestep
        return self.head(pooled)  # [B, out_dim]