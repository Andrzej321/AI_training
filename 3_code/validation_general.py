import torch
from classes import FullFileForTestings  # Import dataset class and model
from classes import SpeedEstimatorRNNModified, SpeedEstimatorLSTMModified, SpeedEstimatorGRUModified
import os
import pandas as pd

if __name__ == '__main__':
    # Loading in the hyperparameters
    hyperparams_loc = "../2_trained_models/LSTM/trained_models/i7/it_3_norm/hyperparams_LSTM_it_3.csv"
    hyperparams_df = pd.read_csv(hyperparams_loc, delimiter=";")

    # Reading in the num of models and model locations
    model_folder_loc = "../2_trained_models/LSTM/trained_models/i7/it_3_norm/state_models/lon/"
    pt_files = [f for f in os.listdir(model_folder_loc) if f.endswith(".pt")]

    num_of_models = len(pt_files)

    column_names = [None] * (num_of_models + 3)
    for i in range(num_of_models):
        if i <= num_of_models:
            column_names[i + 1] = pt_files[i]

    column_names[0] = 'time'
    column_names[num_of_models + 1] = 'veh_u'
    column_names[num_of_models + 2] = 'veh_v'

    # Creating the dataframe for the storing the results
    results_df = pd.DataFrame(columns=column_names)

    # Loading in the reference speeds
    validation_data_loc = "../1_data/i7/it_1/it_1_100_norm/3_validation/"
    validation_files = [f for f in os.listdir(validation_data_loc) if f.endswith(".csv")]

    csv_save_loc = "../2_trained_models/LSTM/trained_models/i7/it_3_norm/results/lon/"

    for i in range(len(validation_files)):
        validation_df = pd.read_csv(validation_data_loc + validation_files[i], delimiter=",")

        results_df['time'] = validation_df['Time']
        results_df['veh_u'] = validation_df['veh_u']
        results_df['veh_v'] = validation_df['veh_v']

        # The test dataset is prepared for to be fed in to the model
        validation_dataset = FullFileForTestings(validation_data_loc + validation_files[i])
        features, actual_speeds = validation_dataset.get_full_data()

        for j in range(num_of_models):

            # Pass all data to the model directly
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Ensure batch dimension is present: (1, seq_len, input_size)
            features_batched = features.unsqueeze(0).to(device)

            # Load checkpoint first to get the exact hyperparameters used for training
            checkpoint_path = model_folder_loc + pt_files[j]
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            input_size = int(checkpoint.get("input_size"))
            hidden_size = int(checkpoint.get("hidden_size"))
            num_layers = int(checkpoint.get("num_layers"))
            output_size = int(checkpoint.get("output_size"))
            seq_len = int(checkpoint.get("sequence_length"))
            # If you also saved step_size in the checkpoint, use it; otherwise set to the value used in training
            step_size = int(checkpoint.get("step_size", 5))

            # Define and load the model USED IN TRAINING (not the Modified one)
            from classes import SpeedEstimatorLSTM  # local import to avoid changing earlier imports
            model = SpeedEstimatorLSTM(input_size, hidden_size, num_layers, output_size).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Prepare sliding windows matching training setup
            # Expecting features shape: (T, input_size)
            feats = features.to(device)
            T = feats.shape[0]
            windows = []
            end_indices = []
            start = 0
            while start + seq_len <= T:
                windows.append(feats[start:start + seq_len, :].unsqueeze(0))  # (1, seq_len, input_size)
                end_indices.append(start + seq_len - 1)
                start += step_size

            if len(windows) == 0:
                # Not enough timesteps for this model's sequence length
                results_df[pt_files[j]] = pd.Series([float("nan")] * len(validation_df))
                continue

            batch = torch.cat(windows, dim=0)  # (N_windows, seq_len, input_size)

            # Inference: SpeedEstimatorGRU returns (N_windows, output_size) at the last timestep, with sigmoid*100 applied
            with torch.no_grad():
                preds = model(batch).detach().cpu().numpy().reshape(-1)

            # Map window-end predictions back to a full-length series
            series = pd.Series([float("nan")] * T)
            for idx, pred in zip(end_indices, preds):
                series.iloc[idx] = pred

            # Optional: fill in-between timesteps by forward-filling predictions
            series = series.ffill().bfill()

            results_df[pt_files[j]] = series.values[:len(results_df)]


        results_df.to_csv(csv_save_loc + validation_files[i], index=False)
        results_df = results_df.iloc[0:0]




