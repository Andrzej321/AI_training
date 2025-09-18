import torch.nn as nn
import torch.optim as optim
from classes import SpeedEstimatorRNN, VehicleSpeedDatasetLongLat, VehicleSpeedDatasetLong, SpeedEstimatorLSTM, SpeedEstimatorGRU, SpeedEstimatorRNN
from torch.utils.data import DataLoader
import torch.onnx
import pandas as pd

if __name__ == '__main__':

    if torch.cuda.is_available():
        print("CUDA is available! You can use a GPU for 1_training.")
        print("Number of GPUs available:", torch.cuda.device_count())
        print("Current GPU being used:", torch.cuda.current_device())
        print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available. Training will be performed on the CPU.")

    # Set dataset path
    training_data_path = "../1_data/i7/it_1/it_1_100_norm/1_training"
    extension = "*.csv"

    test_data_path = "../1_data/i7/it_1/it_1_100_norm/2_testing"

    # Hyperparameters that will alter throughout the model creations
    df = pd.read_csv("hyperparams.csv", delimiter=";")

    input_size = 12  # Number of CAN signals per timestep
    learning_rate = 0.0001
    # num of sequences in one batch
    batch_size = 128
    dropout_rate = 0.2


    # parameters of the simulation
    step_size = 5 # what the overlap between the sequences should look like in the extracted dataset
    output_size = 1
    num_epochs = 70
    num_models = len(df["model_type"])

    location_state_GRU = "../2_trained_models/Simple_RNN/trained_models/i7/it_2/state_models/lon/model_RNN_lon_"
    location_traced_GRU = "../2_trained_models/Simple_RNN/trained_models/i7/it_2/traced_models/lon/model_RNN_lon_"

    # Initialize variables to track the best test/validation loss
    patience = 5

    # Training loops
    for j in range(0, num_models):

        early_stopping_counter = 0
        best_test_loss = float('inf')

        print("-------------------------------------")
        print(f"Training has started for RNN model {j}")

        # Load dataset and DataLoader
        train_dataset = VehicleSpeedDatasetLong(training_data_path, extension, seq_length = df["sequence_size"][j], step_size = step_size)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers= 6, pin_memory=True)

        # Load test dataset and DataLoader
        test_dataset = VehicleSpeedDatasetLong(test_data_path, extension, seq_length=df["sequence_size"][j], step_size=step_size)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch size = 1 for test eval

        # Initialize model, loss function, and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SpeedEstimatorGRU(input_size, int(df["hidden_size"][j]), int(df["num_of_layers"][j]), output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        example_input = torch.rand(1, df["sequence_size"][j], input_size).to(device)  # Example input matching model dimensions

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0

            print("-------------------------------------")
            print(f"Epoch [{epoch+1}] has started")

            for batch_idx, (features, speeds) in enumerate(train_dataloader):
                speeds = speeds.squeeze(1)  # Remove extra dimension from speeds if present
                features, speeds = features.to(device), speeds.to(device)

                # Forward pass
                outputs = model(features)

                assert outputs.shape == speeds.shape, f"Shape mismatch: outputs {outputs.shape} vs speeds {speeds.shape}"

                train_loss = criterion(outputs, speeds)

                # Backward pass
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                total_train_loss += train_loss.item()

            print(f"Model: {j}, Epoch [{epoch+1}/{num_epochs}], Loss: {total_train_loss/len(train_dataloader):.4f}")

            model.eval()
            total_test_loss = 0

            with torch.no_grad():  # No need to compute gradients for validation/test
                for features, speeds in test_dataloader:
                    speeds = speeds.squeeze(1)
                    features, speeds = features.to(device), speeds.to(device)

                    # Forward pass
                    test_outputs = model(features)
                    test_loss = criterion(test_outputs, speeds)

                    total_test_loss += test_loss.item()

            avg_test_loss = total_test_loss / len(test_dataloader)

            print(f"Model: {j}, Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")

             # Checkpoint: Save model if test loss improves
            if avg_test_loss < best_test_loss:
                print(f"New best model found! Test Loss improved from {best_test_loss:.4f} to {avg_test_loss:.4f}")
                best_test_loss = avg_test_loss
                early_stopping_counter = 0

                # Save model state and optimizer state
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "sequence_length": df["sequence_size"][j],
                    "input_size": input_size,
                    "hidden_size": df["hidden_size"][j],
                    "num_layers": df["num_of_layers"][j],
                    "output_size": output_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs
                }, location_state_GRU + str(j) + ".pt")
                print("model " + location_state_GRU + str(j) + ".pt" + " saved")

                # Save traced model for MATLAB -> taken out

                traced_model = torch.jit.trace(model, example_input)
                torch.jit.save(traced_model, location_traced_GRU + str(j) + "_traced_jit_save.pt")  # Save as traced TorchScript model
                print("model " + location_traced_GRU + str(j) + "_traced_jit_save.pt" + " saved")

                traced_model.save(location_traced_GRU + str(j) + "_traced_simple_save.pt")  # Save as traced TorchScript model
                print("model " + location_traced_GRU + str(j) + "_traced_simple_save.pt" + " saved")



                # Export model to ONNX
                onnx_model_path = location_traced_GRU + str(j) + "_traced.onnx"

                torch.onnx.export(
                    model,                     # PyTorch model
                    example_input,             # Example input (same as used for tracing)
                    onnx_model_path,           # Output filename
                    export_params=True,
                    opset_version=11,          # MATLAB supports up to opset 11/12 reliably
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                print("model " + location_traced_GRU + str(j) + "_traced.onnx" + " saved")
                print("---------------------")
                print(f"all model_{j} saved")
            else:
                early_stopping_counter += 1
                print(f"Test loss has not improved; early stopping counter: {early_stopping_counter}")

            if early_stopping_counter >= patience:
                print("Early stopping triggered -> starting next model!")
                print("------------------------------------------------")
                break  # Exit the 1_training loop early

        if early_stopping_counter < patience:
            print("We're out of epochs but patience limit has not been reached -> starting next model!")
            print("-----------------------------------------------------------------------------------")