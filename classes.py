import torch, torch.nn as nn
import os, glob, pandas as pd
from torch.utils.data import Dataset
import numpy as np


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
        self.fc = nn.Linear(hidden_size, output_size)  # Now outputs 2 values: longitudinal and lateral velocity

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

class VehicleSpeedDataset(Dataset):
    """
    A custom PyTorch Dataset for loading vehicle CAN signal data and speed values
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

        # Load data
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

        can_signals = self.data.drop(columns=['veh_u', 'veh_v', 'time_series']).values
        speed_values_u = self.data['veh_u'].values
        speed_values_v = self.data['veh_v'].values

        # Stack both velocities together
        speed_values = np.stack([speed_values_u, speed_values_v], axis=1)

        # Convert data to PyTorch tensors
        can_signals_tensor = torch.tensor(can_signals, dtype=torch.float32)
        speed_values_tensor = torch.tensor(speed_values, dtype=torch.float32)

        return can_signals_tensor, speed_values_tensor