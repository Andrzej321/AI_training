import torch, torch.nn as nn
import os, glob, pandas as pd
from torch.utils.data import Dataset
import numpy as np

from typing import List, Optional

from torch.nn.utils import weight_norm  # for TCN weight normalization

class SpeedEstimatorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=2):
        super(SpeedEstimatorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Add sequence dimension if it's missing
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through RNN
        out, _ = self.rnn(x, h0)

        # Take the output from the last timestep
        out = self.fc(out[:, -1, :])  # Now this will work correctly

        return out

class SpeedEstimatorRNNModified(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=2):
        super(SpeedEstimatorRNNModified, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # RNN returns (output, hidden_state)
        out = self.fc(out)  # Apply the fully connected layer to all time steps
        return out

class SpeedEstimatorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=2):  # Changed default output_size to 2
        super(SpeedEstimatorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Now results 2 values: longitudinal and lateral velocity

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # out will now have shape (batch_size, 2)

        return out

class SpeedEstimatorLSTMModified(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=2):
        super(SpeedEstimatorLSTMModified, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)  # out will have shape (batch_size, sequence_length, 2)
        return out

class SpeedEstimatorGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=2):  # Changed default output_size to 2
        super(SpeedEstimatorGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Now results 2 values: longitudinal and lateral velocity

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # out will now have shape (batch_size, 2)

        out = torch.sigmoid(out) * 100

        return out

class SpeedEstimatorGRUModified(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=2):
        super(SpeedEstimatorGRUModified, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)  # out will have shape (batch_size, sequence_length, 2)
        return out

class VehicleSpeedDatasetLongLat(Dataset):
    """
    A custom PyTorch Dataset for loading vehicle CAN signal 1_data and speed values
    from multiple `.csv` files for RNN-based speed prediction.
    """

    def __init__(self, data_path, extension="*.csv", seq_length=100, step_size=10):
        """
        Initialize the dataset object.

        Args:
            data_path (str): Path to the directory containing the `.csv` files.
            extension (str): The file pattern to search for (e.g., "*.csv").
            seq_length (int): The number of timesteps in each sequence.
            step_size (int): The step size for the sliding window (default is 10).
        """
        self.csv_files = glob.glob(os.path.join(data_path, extension))
        if len(self.csv_files) == 0:
            raise RuntimeError(f"No files found in directory '{data_path}' with extension '{extension}'.")

        self.seq_length = seq_length
        self.step_size = step_size
        self.data = []

        # Pre-compute the number of valid sequences across all files
        for file in self.csv_files:
            df = pd.read_csv(file)

            if len(df) < self.seq_length:
                print(
                    f"Skipping file {file} because it has fewer rows ({len(df)}) than seq_length ({self.seq_length})!")
                continue

            file_sequences = 0  # Counter for this file's sequences
            for i in range(0, len(df) - self.seq_length + 1, self.step_size):
                self.data.append((file, i))
                file_sequences += 1

    def __len__(self):
        """
        Return the total number of sequences across all files.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single sequence and its target value.

        Args:
            idx (int): The index of the sequence.

        Returns:
            can_signals (torch.Tensor): A tensor of CAN signal sequences.
            speed_value (torch.Tensor): A tensor of the target value (speed).
        """
        # Get the file and start index for this sequence
        file, start_idx = self.data[idx]

        # Read the file
        try:
            df = pd.read_csv(file)
        except Exception as e:
            raise RuntimeError(f"Error reading file {file}: {e}")

        # Check for both velocity columns
        if 'veh_u' not in df.columns or 'veh_v' not in df.columns:
            raise ValueError("Expected columns 'veh_u' and 'veh_v' not found in the file!")

        # Extract CAN signals and speed values
        can_signals = df.drop(columns=['veh_u', 'veh_v', 'time_series']).values
        speed_values_u = df['veh_u'].values
        speed_values_v = df['veh_v'].values

        # Extract the sequence
        can_sequence = can_signals[start_idx:start_idx + self.seq_length, :]
        speed_target_u = speed_values_u[start_idx + self.seq_length - 1]
        speed_target_v = speed_values_v[start_idx + self.seq_length - 1]

        # Convert to PyTorch tensors
        can_sequence = torch.tensor(can_sequence, dtype=torch.float32)
        speed_target = torch.tensor([speed_target_u, speed_target_v], dtype=torch.float32)

        return can_sequence, speed_target.unsqueeze(0)

class VehicleSpeedDatasetLong(Dataset):
    """
    A custom PyTorch Dataset for loading vehicle CAN signal 1_data and speed values
    from multiple `.csv` files for RNN-based speed prediction.
    """

    def __init__(self, data_path, extension="*.csv", seq_length=100, step_size=10):
        """
        Initialize the dataset object.

        Args:
            data_path (str): Path to the directory containing the `.csv` files.
            extension (str): The file pattern to search for (e.g., "*.csv").
            seq_length (int): The number of timesteps in each sequence.
            step_size (int): The step size for the sliding window (default is 10).
        """
        self.csv_files = glob.glob(os.path.join(data_path, extension))
        if len(self.csv_files) == 0:
            raise RuntimeError(f"No files found in directory '{data_path}' with extension '{extension}'.")

        self.seq_length = seq_length
        self.step_size = step_size
        self.data = []

        # Pre-compute the number of valid sequences across all files
        for file in self.csv_files:
            df = pd.read_csv(file)

            if len(df) < self.seq_length:
                print(
                    f"Skipping file {file} because it has fewer rows ({len(df)}) than seq_length ({self.seq_length})!")
                continue

            file_sequences = 0  # Counter for this file's sequences
            for i in range(0, len(df) - self.seq_length + 1, self.step_size):
                self.data.append((file, i))
                file_sequences += 1

    def __len__(self):
        """
        Return the total number of sequences across all files.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single sequence and its target value.

        Args:
            idx (int): The index of the sequence.

        Returns:
            can_signals (torch.Tensor): A tensor of CAN signal sequences.
            speed_value (torch.Tensor): A tensor of the target value (speed).
        """
        # Get the file and start index for this sequence
        file, start_idx = self.data[idx]

        # Read the file
        try:
            df = pd.read_csv(file)
        except Exception as e:
            raise RuntimeError(f"Error reading file {file}: {e}")

        # Check for both velocity columns
        if 'veh_u' not in df.columns or 'veh_v' not in df.columns:
            raise ValueError("Expected columns 'veh_u' and 'veh_v' not found in the file!")

        # Extract CAN signals and speed values
        can_signals = df.drop(columns=['veh_u', 'veh_v', 'Time', 'imu_COG_acc_z', 'imu_COG_gyro_roll_rate', 'imu_COG_gyro_pitch_rate'
            , 'drive_torque_FR', 'drive_torque_RR', 'brake_pressure_FR', 'brake_pressure_RR', 'rwa_RM']).values
        speed_values_u = df['veh_u'].values

        # Extract the sequence
        can_sequence = can_signals[start_idx:start_idx + self.seq_length, :]
        speed_target_u = speed_values_u[start_idx + self.seq_length - 1]

        # Convert to PyTorch tensors
        can_sequence = torch.tensor(can_sequence, dtype=torch.float32)
        speed_target = torch.tensor([speed_target_u], dtype=torch.float32)

        return can_sequence, speed_target.unsqueeze(0)

class VehicleSpeedDatasetLat(Dataset):
    """
    A custom PyTorch Dataset for loading vehicle CAN signal 1_data and speed values
    from multiple `.csv` files for RNN-based speed prediction.
    """

    def __init__(self, data_path, extension="*.csv", seq_length=100, step_size=10):
        """
        Initialize the dataset object.

        Args:
            data_path (str): Path to the directory containing the `.csv` files.
            extension (str): The file pattern to search for (e.g., "*.csv").
            seq_length (int): The number of timesteps in each sequence.
            step_size (int): The step size for the sliding window (default is 10).
        """
        self.csv_files = glob.glob(os.path.join(data_path, extension))
        if len(self.csv_files) == 0:
            raise RuntimeError(f"No files found in directory '{data_path}' with extension '{extension}'.")

        self.seq_length = seq_length
        self.step_size = step_size
        self.data = []

        # Pre-compute the number of valid sequences across all files
        for file in self.csv_files:
            df = pd.read_csv(file)

            if len(df) < self.seq_length:
                print(
                    f"Skipping file {file} because it has fewer rows ({len(df)}) than seq_length ({self.seq_length})!")
                continue

            file_sequences = 0  # Counter for this file's sequences
            for i in range(0, len(df) - self.seq_length + 1, self.step_size):
                self.data.append((file, i))
                file_sequences += 1

    def __len__(self):
        """
        Return the total number of sequences across all files.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single sequence and its target value.

        Args:
            idx (int): The index of the sequence.

        Returns:
            can_signals (torch.Tensor): A tensor of CAN signal sequences.
            speed_value (torch.Tensor): A tensor of the target value (speed).
        """
        # Get the file and start index for this sequence
        file, start_idx = self.data[idx]

        # Read the file
        try:
            df = pd.read_csv(file)
        except Exception as e:
            raise RuntimeError(f"Error reading file {file}: {e}")

        # Check for both velocity columns
        if 'veh_u' not in df.columns or 'veh_v' not in df.columns:
            raise ValueError("Expected columns 'veh_u' and 'veh_v' not found in the file!")

        # Extract CAN signals and speed values
        can_signals = df.drop(columns=['veh_u', 'veh_v', 'time_series']).values
        speed_values_v = df['veh_v'].values

        # Extract the sequence
        can_sequence = can_signals[start_idx:start_idx + self.seq_length, :]
        speed_target_v = speed_values_v[start_idx + self.seq_length - 1]

        # Convert to PyTorch tensors
        can_sequence = torch.tensor(can_sequence, dtype=torch.float32)
        speed_target = torch.tensor([speed_target_v], dtype=torch.float32)

        return can_sequence, speed_target.unsqueeze(0)

class FullFileForTestings(Dataset):
    """
    A dataset class that loads a single CSV file and prepares it as one batch for testing.
    """

    def __init__(self, csv_file_path):
        """
        Args:
            csv_file_path (str): Path to the CSV file for testing.
        """
        self.csv_file_path = csv_file_path

        # Load 1_data
        self.data = pd.read_csv(csv_file_path)

    def get_full_data(self):
        """
        Extracts input features (CAN signals) and speed values.

        Returns:
            can_signals (torch.Tensor): The tensor containing all CAN signal inputs.
            speed_values (torch.Tensor): The tensor containing all speed values.
        """
        # Check for both velocity columns
        if 'veh_u' not in self.data.columns or 'veh_v' not in self.data.columns:
            raise ValueError("Required columns 'veh_u' and 'veh_v' not found in CSV file!")

        can_signals = self.data.drop(columns=['veh_u', 'veh_v', 'Time', 'imu_COG_acc_z', 'imu_COG_gyro_roll_rate', 'imu_COG_gyro_pitch_rate'
            , 'drive_torque_FR', 'drive_torque_RR', 'brake_pressure_FR', 'brake_pressure_RR', 'rwa_RM']).values
        speed_values_u = self.data['veh_u'].values
        speed_values_v = self.data['veh_v'].values

        # Stack both velocities together
        speed_values = np.stack([speed_values_u, speed_values_v], axis=1)

        # Convert 1_data to PyTorch tensors
        can_signals_tensor = torch.tensor(can_signals, dtype=torch.float32)
        speed_values_tensor = torch.tensor(speed_values, dtype=torch.float32)

        return can_signals_tensor, speed_values_tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class SpeedEstimatorTransformer(nn.Module):
    """
    Transformer encoder-based speed estimator.
    Expects input of shape (batch, seq_len, input_size).
    Produces (batch, output_size) by reading the last time step.
    """
    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_layers: int,
        output_size: int = 1,
        nhead: int = 4,
        dim_feedforward: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # keep (batch, seq, feature)
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)                # (batch, seq, d_model)
        x = self.pos_enc(x)                  # (batch, seq, d_model)
        x = self.encoder(x)                  # (batch, seq, d_model)
        last_t = x[:, -1, :]                 # (batch, d_model)
        out = self.head(last_t)              # (batch, output_size)
        return out

class Chomp1d(nn.Module):
    """Crops the last 'chomp_size' timesteps to preserve causality when padding is used."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

def make_activation(name: str):
    name = name.lower()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()

def make_norm(norm: str, num_channels: int):
    norm = norm.lower()
    if norm == "batch":
        return nn.BatchNorm1d(num_channels)
    if norm == "layer":
        # Approximate LayerNorm across channels with GroupNorm(1, C)
        return nn.GroupNorm(1, num_channels)
    return nn.Identity()

class TemporalBlock(nn.Module):
    """
    Residual TCN block with N causal/non-causal Conv1d layers.
    Each conv in the block uses the same dilation factor (standard TCN design).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        convolutions_per_block: int = 2,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
        activation: str = "relu",
        norm_in_block: str = "none",
        causal: bool = True,
    ):
        super().__init__()
        act = make_activation(activation)
        norm_ctor = lambda c: make_norm(norm_in_block, c)

        layers = []
        current_in = in_channels
        for _ in range(max(1, convolutions_per_block)):
            # Padding for causal or "same" non-causal
            if causal:
                padding = (kernel_size - 1) * dilation
                chomp = Chomp1d(padding)
            else:
                padding = ((kernel_size - 1) * dilation) // 2
                chomp = nn.Identity()

            conv = nn.Conv1d(current_in, out_channels, kernel_size, padding=padding, dilation=dilation)
            if use_weight_norm:
                conv = weight_norm(conv)

            layers.extend([
                conv,
                chomp,
                norm_ctor(out_channels),
                act,
                nn.Dropout(dropout),
            ])
            current_in = out_channels

        self.net = nn.Sequential(*layers)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.activation = make_activation(activation)

        # He init for convs
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
    """
    Stack of TemporalBlocks with user-defined channels and dilation schedule.
    """
    def __init__(
        self,
        in_channels: int,
        channels_per_layer: List[int],
        kernel_size: int,
        dilation_schedule: List[int],
        convolutions_per_block: int = 2,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
        activation: str = "relu",
        norm_in_block: str = "none",
        causal: bool = True,
    ):
        super().__init__()
        assert len(channels_per_layer) >= 1, "channels_per_layer must be a non-empty list"
        assert len(dilation_schedule) >= 1, "dilation_schedule must be a non-empty list"

        layers = []
        c_in = in_channels
        L = len(channels_per_layer)
        for i in range(L):
            d = dilation_schedule[i] if i < len(dilation_schedule) else dilation_schedule[-1]
            c_out = channels_per_layer[i]
            layers.append(
                TemporalBlock(
                    c_in,
                    c_out,
                    kernel_size=kernel_size,
                    dilation=d,
                    convolutions_per_block=convolutions_per_block,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    activation=activation,
                    norm_in_block=norm_in_block,
                    causal=causal,
                )
            )
            c_in = c_out

        self.network = nn.Sequential(*layers)
        self.out_channels = channels_per_layer[-1]

    def forward(self, x_b_l_c: torch.Tensor) -> torch.Tensor:
        # Accept [B, L, C] -> transpose to [B, C, L] for Conv1d
        x = x_b_l_c.transpose(1, 2)
        y = self.network(x)  # [B, C_out, L]
        return y.transpose(1, 2)  # back to [B, L, C_out]

class SpeedEstimatorTCN(nn.Module):
    """
    TCN-based speed estimator (sequence-to-one for current speed).
    Flexible constructor:
      - Use channels_per_layer + dilation_schedule + num_residual_blocks + convolutions_per_block (preferred)
      - Or provide hidden_size + num_layers (compat with your other classes); this expands to a uniform channels list and exponential dilations.
    Reads last timestep by default (head_pooling='last'), or global average if head_pooling='global_avg'.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        # New preferred knobs:
        channels_per_layer: Optional[List[int]] = None,
        num_residual_blocks: Optional[int] = None,
        convolutions_per_block: int = 2,
        kernel_size: int = 3,
        dilation_schedule: Optional[List[int]] = None,
        # Back-compat knobs (optional):
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        # Regularization/behavior:
        dropout: float = 0.1,
        use_weight_norm: bool = True,
        activation: str = "relu",
        norm_in_block: str = "none",
        head_pooling: str = "last",  # 'last' | 'global_avg'
        causal: bool = True,
        output_clamp_min: Optional[float] = None,
    ):
        super().__init__()
        # Derive channels_per_layer and dilation_schedule if not fully specified
        if channels_per_layer is None:
            nl = num_residual_blocks if num_residual_blocks is not None else (num_layers if num_layers is not None else 4)
            hs = hidden_size if hidden_size is not None else 64
            channels_per_layer = [hs] * nl
        if dilation_schedule is None:
            dilation_schedule = [2 ** i for i in range(len(channels_per_layer))]

        self.tcn = TemporalConvNet(
            in_channels=input_size,
            channels_per_layer=channels_per_layer,
            kernel_size=kernel_size,
            dilation_schedule=dilation_schedule,
            convolutions_per_block=convolutions_per_block,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            activation=activation,
            norm_in_block=norm_in_block,
            causal=causal,
        )
        self.head_pooling = head_pooling.lower()
        self.output_clamp_min = output_clamp_min
        self.head = nn.Linear(self.tcn.out_channels, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size); allow (batch, input_size) by adding a seq dim
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feats = self.tcn(x)  # [B, L, C_tcn]
        if self.head_pooling == "global_avg":
            pooled = feats.mean(dim=1)  # [B, C_tcn]
        else:
            pooled = feats[:, -1, :]     # last timestep [B, C_tcn]
        out = self.head(pooled)          # [B, output_size]
        if self.output_clamp_min is not None:
            out = torch.clamp(out, min=self.output_clamp_min)
        return out

class SpeedEstimatorTCNModified(nn.Module):
    """
    Sequence-to-sequence TCN: returns per-timestep predictions [B, L, output_size].
    Include only if you need per-timestep supervision. Not used for "current speed".
    """
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        channels_per_layer: Optional[List[int]] = None,
        num_residual_blocks: Optional[int] = None,
        convolutions_per_block: int = 2,
        kernel_size: int = 3,
        dilation_schedule: Optional[List[int]] = None,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
        activation: str = "relu",
        norm_in_block: str = "none",
        causal: bool = True,
    ):
        super().__init__()
        if channels_per_layer is None:
            nl = num_residual_blocks if num_residual_blocks is not None else (num_layers if num_layers is not None else 4)
            hs = hidden_size if hidden_size is not None else 64
            channels_per_layer = [hs] * nl
        if dilation_schedule is None:
            dilation_schedule = [2 ** i for i in range(len(channels_per_layer))]

        self.tcn = TemporalConvNet(
            in_channels=input_size,
            channels_per_layer=channels_per_layer,
            kernel_size=kernel_size,
            dilation_schedule=dilation_schedule,
            convolutions_per_block=convolutions_per_block,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            activation=activation,
            norm_in_block=norm_in_block,
            causal=causal,
        )
        self.head = nn.Linear(self.tcn.out_channels, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feats = self.tcn(x)          # [B, L, C_tcn]
        out = self.head(feats)       # [B, L, output_size]
        return out