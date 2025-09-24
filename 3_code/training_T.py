import os
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import pandas as pd
from torch.utils.data import DataLoader
from classes import VehicleSpeedDatasetLong, VehicleSpeedDatasetLongLat, VehicleSpeedDatasetLat, SpeedEstimatorTransformer

if __name__ == "__main__":
    # CUDA info
    if torch.cuda.is_available():
        print("CUDA is available! You can use a GPU for training.")
        print("Number of GPUs available:", torch.cuda.device_count())
        print("Current GPU being used:", torch.cuda.current_device())
        print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available. Training will be performed on the CPU.")

    # Data
    training_data_path = "../1_data/i7/it_1/it_1_100_norm/1_training"
    test_data_path = "../1_data/i7/it_1/it_1_100_norm/2_testing"
    extension = "*.csv"

    # Hyperparams grid from CSV
    hyperparams_loc = "../2_trained_models/Transformer/i7/it_1_norm/hyperparams_T_it_4.csv"
    df = pd.read_csv(hyperparams_loc, delimiter=";")
    # Filter rows for Transformer if model_type column exists
    if "model_type" in df.columns:
        df = df[df["model_type"].astype(str).str.lower().str.contains("transformer")]
        if len(df) == 0:
            raise ValueError("No Transformer rows found in hyperparams.csv (model_type contains 'Transformer').")

    # Fixed/problem-level params
    input_size = 12           # number of CAN signals per timestep
    output_size = 1           # longitudinal speed
    learning_rate = 1e-4
    batch_size = 128
    step_size = 5             # sequence overlap during dataset extraction
    num_epochs = 70
    patience = 5

    # Optional Transformer params from CSV columns (fallback defaults used if missing)
    def get_col(row, name, default):
        return int(row[name]) if name in df.columns and not pd.isna(row[name]) else default

    def get_float(row, name, default):
        return float(row[name]) if name in df.columns and not pd.isna(row[name]) else default

    # Output locations
    location_state = "../2_trained_models/Transformer/i7/it_1_norm/state_models/lon/model_T_lon_"
    location_traced = "../2_trained_models/Transformer/i7/it_1_norm/traced_models/lon/model_T_lon_"
    os.makedirs(os.path.dirname(location_state), exist_ok=True)
    os.makedirs(os.path.dirname(location_traced), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = df.reset_index(drop=True)
    num_models = len(df)

    for j in range(num_models):

        seq_len = int(df["sequence_size"][j])
        d_model = int(df["d_model"][j])           # map hidden_size -> d_model
        num_layers = int(df["num_of_layers"][j])       # map num_of_layers -> encoder layers
        nhead = int(df["nhead"][j])
        dim_ff = int(df["dim_ff"][j])
        raw_dropout = df["dropout"][j]
        dropout = float(raw_dropout) if pd.notna(raw_dropout) else 0.1
        id = int(df["ID"][j])

        print("-------------------------------------")
        print(f"Training has started for Transformer model {j} | seq={seq_len}, d_model={d_model}, layers={num_layers}, nhead={nhead}, ff={dim_ff}")

        # Datasets and loaders
        train_dataset = VehicleSpeedDatasetLong(training_data_path, extension, seq_length=seq_len, step_size=step_size)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

        test_dataset = VehicleSpeedDatasetLong(test_data_path, extension, seq_length=seq_len, step_size=step_size)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Model/opt/criterion
        model = SpeedEstimatorTransformer(
            input_size=input_size,
            d_model=d_model,
            num_layers=num_layers,
            output_size=output_size,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        example_input = torch.rand(1, seq_len, input_size, device=device)

        best_test_loss = float("inf")
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0

            print("-------------------------------------")
            print(f"Epoch [{epoch+1}] has started")

            for features, speeds in train_dataloader:
                # features: (B, T, input_size); speeds: (B, 1) possibly with extra dim
                speeds = speeds.squeeze(1)  # ensure shape (B,) or (B,1) -> (B,)
                if speeds.dim() == 1:
                    speeds = speeds.unsqueeze(1)  # (B,) -> (B,1)

                features, speeds = features.to(device), speeds.to(device)

                outputs = model(features)
                assert outputs.shape == speeds.shape, f"Shape mismatch: outputs {outputs.shape} vs speeds {speeds.shape}"

                loss = criterion(outputs, speeds)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_dataloader))
            print(f"Model: {j}, Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for features, speeds in test_dataloader:
                    speeds = speeds.squeeze(1)
                    if speeds.dim() == 1:
                        speeds = speeds.unsqueeze(1)
                    features, speeds = features.to(device), speeds.to(device)

                    test_outputs = model(features)
                    test_loss = criterion(test_outputs, speeds)
                    total_test_loss += test_loss.item()

            avg_test_loss = total_test_loss / max(1, len(test_dataloader))
            print(f"Model: {j}, Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")

            # Checkpoint on improvement
            if avg_test_loss < best_test_loss:
                print(f"New best model found! Test Loss improved from {best_test_loss:.4f} to {avg_test_loss:.4f}")
                best_test_loss = avg_test_loss
                early_stopping_counter = 0

                # Save state dict
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "sequence_length": seq_len,
                        "input_size": input_size,
                        "d_model": d_model,
                        "num_layers": num_layers,
                        "nhead": nhead,
                        "dim_feedforward": dim_ff,
                        "dropout": dropout,
                        "output_size": output_size,
                        "learning_rate": learning_rate,
                        "num_epochs": num_epochs,
                    },
                    location_state + str(j) + ".pt",
                )
                print("model " + location_state + str(id) + ".pt" + " saved")

                # TorchScript trace
                scripted = torch.jit.script(model.eval())
                scripted = torch.jit.freeze(scripted)
                torch.jit.save(scripted, location_traced + str(j) + "_scripted_jit_save.pt")
                print("model " + location_traced + str(j) + "_scripted_jit_save.pt" + " saved")

                scripted.save(location_traced + str(id) + "_scripted_simple_save.pt")
                print("model " + location_traced + str(id) + "_scripted_simple_save.pt" + " saved")

                # ONNX export
                onnx_model_path = location_traced + str(id) + "_traced.onnx"
                torch.onnx.export(
                    model.eval(),
                    example_input,
                    onnx_model_path,
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                )
                print("model " + onnx_model_path + " saved")
                print("---------------------")
                print(f"all model_{id} saved")
            else:
                early_stopping_counter += 1
                print(f"Test loss has not improved; early stopping counter: {early_stopping_counter}")

            if early_stopping_counter >= patience:
                print("Early stopping triggered -> starting next model!")
                print("------------------------------------------------")
                break

        if early_stopping_counter < patience:
            print("We're out of epochs but patience limit has not been reached -> starting next model!")
            print("-----------------------------------------------------------------------------------")
