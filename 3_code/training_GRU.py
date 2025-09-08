import torch.nn as nn
import torch.optim as optim
from classes import SpeedEstimatorRNN, VehicleSpeedDatasetLongLat, VehicleSpeedDatasetLong, SpeedEstimatorLSTM, SpeedEstimatorGRU
from torch.utils.data import DataLoader
import torch.onnx
import pandas as pd

df = pd.read_csv("hyperparams.csv", delimiter=";")

if torch.cuda.is_available():
    print("CUDA is available! You can use a GPU for 1_training.")
    print("Number of GPUs available:", torch.cuda.device_count())
    print("Current GPU being used:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available. Training will be performed on the CPU.")


# Set dataset path
training_data_path = "../../1_data/i7/it_1/1_training"
extension = "*.csv"

test_data_path = "../../1_data/i7/it_1/2_testing"

# Hyperparameters that will alter throughout the model creations
input_size = 20  # Number of CAN signals per timestep
hidden_size = [32, 64, 128, 64, 96]
num_layers = [1, 2, 1, 2, 2]
learning_rate = [0.0001] * 5
# num of sequences in one batch
batch_size = [128] * 5
dropout_rate = [0.2] * 5
sequence_length = [400, 600, 800, 400, 800]


# parameters of the simulation
step_size = 10 # what the overlap between the sequences should look like in the extracted dataset

output_size = 1

num_epochs = 60

num_models = len(num_layers)

location_state = "Simple RNN/trained_models/i7/it_1/state_models/model_"
location_traced = "Simple RNN/trained_models/i7/it_1/traced_models/model_"

location_state_LSTM = "LSTM/trained_models/i7/it_1/state_models/model_LSTM_"
location_traced_LSTM = "LSTM/trained_models/i7/it_1/traced_models/model_LSTM_"

location_state_GRU = "GRU/trained_models/i7/it_3/state_models/long/model_GRU_long_"
location_traced_GRU = "GRU/trained_models/i7/it_3/traced_models/long/model_GRU_long_"