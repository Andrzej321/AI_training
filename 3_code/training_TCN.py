import os
import json
import time
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import pandas as pd
from torch.utils.data import DataLoader

from classes import VehicleSpeedDatasetLong, SpeedEstimatorTCN


def parse_list_cell(val, cast=int) -> Optional[List[int]]:
    """
    Parse a semicolon-CSV cell that may contain a list like:
    "1,2,4,8" or "[1, 2, 4, 8]". Returns None if val is NaN/empty.
    """
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, list):
        return [cast(x) for x in val]
    s = str(val).strip()
    if s == "" or s.lower() == "none":
        return None
    s = s.strip("[]")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [cast(p) for p in parts]


def to_bool(val, default=False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n"):
        return False
    return default


if __name__ == "__main__":
    # Device info
    if torch.cuda.is_available():
        print("CUDA is available! You can use a GPU for training.")
        print("Number of GPUs available:", torch.cuda.device_count())
        print("Current GPU being used:", torch.cuda.current_device())
        print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available. Training will be performed on the CPU.")

    # Paths
    training_data_path = "../1_data/i7/it_1/it_1_100_norm/1_training"
    test_data_path = "../1_data/i7/it_1/it_1_100_norm/2_testing"
    extension = "*.csv"

    # Load hyperparameters table (semicolon-delimited to match your other trainers)
    df = pd.read_csv("../2_trained_models/TCN/i7/it_1_norm/hyperparams_TCN_it_1.csv", delimiter=";")

    # Fixed or defaults (as per your instruction)
    fixed_input_size = 12     # All created models have 1 input
    fixed_dropout = 0.1      # Dropout fixed
    fixed_step_size = 5      # Dataset sliding window stride

    # Training defaults (can be overridden per-row if present)
    default_learning_rate = 1e-3
    default_weight_decay = 0.0
    default_optimizer = "adam"    # adam | adamw | sgd
    default_loss = "mse"          # mse | smooth_l1 | mae
    default_grad_clip = 1.0
    default_batch_size = 128
    default_epochs = 100
    patience = 5
    min_delta = 0.0  # improvement threshold for early stopping
    default_seed = 42

    # Model behavior/stability defaults
    default_use_weight_norm = True
    default_activation = "relu"       # relu | leaky_relu | gelu
    default_norm_in_block = "none"    # none | batch | layer
    default_head_pooling = "last"     # last | global_avg
    default_causal = True
    default_output_clamp_min = None   # e.g., 0.0 to enforce non-negative speeds

    # Output locations
    location_state_TCN = "../2_trained_models/TCN/trained_models/i7/it_1_norm/state_models/lon/model_TCN_lon_"
    location_traced_TCN = "../2_trained_models/TCN/trained_models/i7/it_1_norm/traced_models/lon/model_TCN_lon_"

    num_models = len(df.index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for j in range(num_models):
        print("-------------------------------------")
        print(f"Training has started for TCN model row {j}")

        # Required core knobs you selected
        sequence_length = int(df["sequence_size"][j]) if "sequence_size" in df.columns else int(df["sequence_length"][j])
        kernel_size = int(df["kernel_size"][j]) if "kernel_size" in df.columns else 3
        num_residual_blocks = int(df["num_residual_blocks"][j]) if "num_residual_blocks" in df.columns else int(df["num_of_layers"][j]) if "num_of_layers" in df.columns else 4
        convolutions_per_block = int(df["convolutions_per_block"][j]) if "convolutions_per_block" in df.columns else 2
        channels_per_layer = parse_list_cell(df["channels_per_layer"][j]) if "channels_per_layer" in df.columns else None
        dilation_schedule = parse_list_cell(df["dilation_schedule"][j]) if "dilation_schedule" in df.columns else None

        # If channels_per_layer isn't provided, fallback to hidden_size repeated
        if channels_per_layer is None:
            hidden_size = int(df["hidden_size"][j]) if "hidden_size" in df.columns else 64
            channels_per_layer = [hidden_size] * num_residual_blocks

        # If dilation_schedule isn't provided, use exponential schedule [1, 2, 4, ...]
        if dilation_schedule is None:
            dilation_schedule = [2 ** i for i in range(num_residual_blocks)]

        # Optional behavior/stability knobs
        use_weight_norm = to_bool(df["use_weight_norm"][j], default_use_weight_norm) if "use_weight_norm" in df.columns else default_use_weight_norm
        activation = str(df["activation"][j]).lower() if "activation" in df.columns and isinstance(df["activation"][j], str) else default_activation
        norm_in_block = str(df["norm_in_block"][j]).lower() if "norm_in_block" in df.columns and isinstance(df["norm_in_block"][j], str) else default_norm_in_block
        head_pooling = str(df["head_pooling"][j]).lower() if "head_pooling" in df.columns and isinstance(df["head_pooling"][j], str) else default_head_pooling
        causal = to_bool(df["causal"][j], default_causal) if "causal" in df.columns else default_causal
        output_clamp_min = None
        if "output_clamp_min" in df.columns:
            try:
                v = df["output_clamp_min"][j]
                if not (isinstance(v, float) and math.isnan(v)):
                    output_clamp_min = float(v)
            except Exception:
                output_clamp_min = default_output_clamp_min

        # Training hyperparams (with per-row overrides if present)
        learning_rate = float(df["learning_rate"][j]) if "learning_rate" in df.columns else default_learning_rate
        weight_decay = float(df["weight_decay"][j]) if "weight_decay" in df.columns else default_weight_decay
        optimizer_name = str(df["optimizer"][j]).lower() if "optimizer" in df.columns else default_optimizer
        loss_name = str(df["loss"][j]).lower() if "loss" in df.columns else default_loss
        grad_clip = float(df["grad_clip"][j]) if "grad_clip" in df.columns else default_grad_clip
        batch_size = int(df["batch_size"][j]) if "batch_size" in df.columns else default_batch_size
        num_epochs = int(df["epochs"][j]) if "epochs" in df.columns else default_epochs
        seed = int(df["seed"][j]) if "seed" in df.columns else default_seed

        # Seed everything
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Data
        train_dataset = VehicleSpeedDatasetLong(training_data_path, extension, seq_length=sequence_length, step_size=fixed_step_size)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

        test_dataset = VehicleSpeedDatasetLong(test_data_path, extension, seq_length=sequence_length, step_size=fixed_step_size)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Model
        model = SpeedEstimatorTCN(
            input_size=fixed_input_size,
            output_size=1,
            channels_per_layer=channels_per_layer,
            num_residual_blocks=len(channels_per_layer),  # authoritative
            convolutions_per_block=convolutions_per_block,
            kernel_size=kernel_size,
            dilation_schedule=dilation_schedule,
            dropout=fixed_dropout,
            use_weight_norm=use_weight_norm,
            activation=activation,
            norm_in_block=norm_in_block,
            head_pooling=head_pooling,
            causal=causal,
            output_clamp_min=output_clamp_min,
        ).to(device)

        # Loss
        if loss_name == "smooth_l1" or loss_name == "huber":
            criterion = nn.SmoothL1Loss()
        elif loss_name == "mae" or loss_name == "l1":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()

        # Optimizer
        if optimizer_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Example input for exports
        example_input = torch.rand(1, sequence_length, fixed_input_size).to(device)

        best_test_loss = float("inf")
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0

            print("-------------------------------------")
            print(f"Epoch [{epoch + 1}] has started")

            for batch_idx, (features, speeds) in enumerate(train_dataloader):
                # Dataset yields features of shape [B, L, C_all]. If fixed_input_size < C_all, take first channel(s).
                if features.size(-1) > fixed_input_size:
                    features = features[:, :, :fixed_input_size]

                speeds = speeds.squeeze(1)  # [B, 1]
                features, speeds = features.to(device), speeds.to(device)

                outputs = model(features)   # [B, 1]
                assert outputs.shape == speeds.shape, f"Shape mismatch: outputs {outputs.shape} vs speeds {speeds.shape}"

                loss = criterion(outputs, speeds)

                optimizer.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_dataloader))
            print(f"Model: {j}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}")

            # Evaluate on test
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for features, speeds in test_dataloader:
                    if features.size(-1) > fixed_input_size:
                        features = features[:, :, :fixed_input_size]

                    speeds = speeds.squeeze(1)
                    features, speeds = features.to(device), speeds.to(device)
                    test_outputs = model(features)
                    test_loss = criterion(test_outputs, speeds)
                    total_test_loss += test_loss.item()

            avg_test_loss = total_test_loss / max(1, len(test_dataloader))
            print(f"Model: {j}, Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.6f}")

            improved = (best_test_loss - avg_test_loss) > min_delta
            if improved:
                print(f"New best model! Test Loss improved from {best_test_loss:.6f} to {avg_test_loss:.6f}")
                best_test_loss = avg_test_loss
                early_stopping_counter = 0

                # Save model state and optimizer state
                os.makedirs(os.path.dirname(location_state_TCN), exist_ok=True)
                os.makedirs(os.path.dirname(location_traced_TCN), exist_ok=True)

                state_path = location_state_TCN + str(j) + ".pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "sequence_length": sequence_length,
                    "input_size": fixed_input_size,
                    "channels_per_layer": channels_per_layer,
                    "dilation_schedule": dilation_schedule,
                    "num_residual_blocks": len(channels_per_layer),
                    "convolutions_per_block": convolutions_per_block,
                    "kernel_size": kernel_size,
                    "dropout": fixed_dropout,
                    "use_weight_norm": use_weight_norm,
                    "activation": activation,
                    "norm_in_block": norm_in_block,
                    "head_pooling": head_pooling,
                    "causal": causal,
                    "output_clamp_min": output_clamp_min,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "optimizer": optimizer_name,
                    "loss": loss_name,
                    "grad_clip": grad_clip,
                    "batch_size": batch_size,
                    "epochs": num_epochs,
                    "seed": seed,
                    "best_test_loss": best_test_loss,
                }, state_path)
                print("model " + state_path + " saved")

                # TorchScript
                traced_model = torch.jit.trace(model, example_input)
                traced_jit_path = location_traced_TCN + str(j) + "_traced_jit_save.pt"
                torch.jit.save(traced_model, traced_jit_path)
                print("model " + traced_jit_path + " saved")

                traced_simple_path = location_traced_TCN + str(j) + "_traced_simple_save.pt"
                traced_model.save(traced_simple_path)
                print("model " + traced_simple_path + " saved")

                # ONNX
                onnx_model_path = location_traced_TCN + str(j) + "_traced.onnx"
                torch.onnx.export(
                    model,
                    example_input,
                    onnx_model_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                )
                print("model " + onnx_model_path + " saved")
                print("---------------------")
                print(f"all model_{j} artifacts saved")
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