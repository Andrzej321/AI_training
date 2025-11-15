import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx  # NEW: ONNX export
from torch.utils.data import DataLoader

from classes import SpeedEstimatorGRU
from datasets_cached import VehicleSpeedDatasetLongCached

# -------------------- Top-level collate function (picklable on Windows) --------------------
def collate_fn(batch):
    xs, ys = zip(*batch)          # xs: list of [T,F], ys: list of [1]
    x = torch.stack(xs, dim=0)    # [B, T, F]
    y = torch.stack(ys, dim=0)    # [B, 1]
    return x, y

def seed_worker(worker_id):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)


def main():
    # -------------------- Paths --------------------
    training_data_path = "../1_data/i7/it_1/it_1_100_norm/1_training"
    test_data_path     = "../1_data/i7/it_1/it_1_100_norm/2_testing"

    # Your hyperparams CSV (semicolon-delimited)
    hyperparams_csv = "../2_trained_models/GRU/trained_models/i7/it_5_norm/hyperparams_GRU_it_5.csv"

    # Column drops to mirror your VehicleSpeedDatasetLong in classes.py (lon variant)
    drop_columns = [
        "veh_u", "veh_v", "Time",
        "imu_COG_acc_z", "imu_COG_gyro_roll_rate", "imu_COG_gyro_pitch_rate",
        "drive_torque_FR", "drive_torque_RR", "brake_pressure_FR", "brake_pressure_RR",
        "rwa_RM"
    ]
    target_column = "veh_u"  # longitudinal target

    # -------------------- Global training knobs --------------------
    learning_rate = 1e-4
    batch_size    = 128
    num_epochs    = 100
    patience      = 5
    step_size_default = 5     # will be overridden per row if present
    use_amp       = True

    # DataLoader perf knobs
    requested_num_workers = 8
    pin_memory            = True
    persistent_workers    = True
    prefetch_factor       = 4

    # Exports
    export_onnx = True        # NEW: export best model per config at end
    onnx_opset  = 11

    # Paths to save (one checkpoint per config row)
    save_prefix_root = "../2_trained_models/GRU/trained_models/i7/it_fast_csv/state_models/lon/"
    os.makedirs(save_prefix_root, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Read hyperparameter table
    print(f"Reading hyperparameters from {hyperparams_csv}")
    df = pd.read_csv(hyperparams_csv, delimiter=";")

    # Normalize column names to lower for robust access
    cols = {c.lower(): c for c in df.columns}
    def get_col(name: str) -> Optional[str]:
        return cols.get(name.lower(), None)

    # Optional columns in your CSV
    col_model_type   = get_col("model_type")
    col_id           = get_col("id")
    col_input_id     = get_col("input_id")
    col_seq_len      = get_col("sequence_size")
    col_hidden_size  = get_col("hidden_size")
    col_num_layers   = get_col("num_of_layers")
    col_step_size    = get_col("step_size")
    col_input_size   = get_col("input_size")
    col_output_size  = get_col("output_size")
    col_dropout      = get_col("dropout_rate")  # if present

    # Iterate over each row (one model per row)
    for j, row in df.iterrows():
        # Extract params with fallback defaults
        seq_length  = int(row[col_seq_len]) if col_seq_len else 100
        hidden_size = int(row[col_hidden_size]) if col_hidden_size else 128
        num_layers  = int(row[col_num_layers]) if col_num_layers else 2
        step_size   = int(row[col_step_size]) if col_step_size else step_size_default
        input_size_expected  = int(row[col_input_size]) if col_input_size else None
        output_size = int(row[col_output_size]) if col_output_size else 1
        dropout_rate = float(row[col_dropout]) if col_dropout else 0.0

        # Derive an ID for save name
        cfg_id = int(row[col_id]) if col_id else j
        model_type = (row[col_model_type] if col_model_type else "GRU")

        print("\n===============================================")
        print(f"Config {j} (ID={cfg_id})  Model={model_type}  seq_len={seq_length}  hidden={hidden_size}  layers={num_layers}  step={step_size}")
        print("===============================================")

        # Build datasets for this seq_length (cached)
        train_dataset = VehicleSpeedDatasetLongCached(
            training_data_path,
            extension="*.csv",
            seq_length=seq_length,
            step_size=step_size,
            drop_columns=drop_columns,
            target_column=target_column,
        )
        test_dataset = VehicleSpeedDatasetLongCached(
            test_data_path,
            extension="*.csv",
            seq_length=seq_length,
            step_size=step_size,
            drop_columns=drop_columns,
            target_column=target_column,
        )

        # Validate input feature size against CSV (if provided)
        detected_input_size = train_dataset.input_size
        if input_size_expected is not None:
            assert detected_input_size == input_size_expected, (
                f"CSV input_size={input_size_expected} differs from detected={detected_input_size}. "
                f"Adjust input_size in CSV or update drop_columns."
            )

        def make_loader(ds, is_train):
            workers = max(0, requested_num_workers)
            pw = (workers > 0) and persistent_workers
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=is_train,
                num_workers=workers,
                pin_memory=(pin_memory and device.type == "cuda"),
                persistent_workers=pw,
                prefetch_factor=(prefetch_factor if workers > 0 else None),
                collate_fn=collate_fn,
                drop_last=False,
                worker_init_fn=seed_worker if workers > 0 else None,
            )

        try:
            train_loader = make_loader(train_dataset, is_train=True)
            test_loader  = make_loader(test_dataset,  is_train=False)
            # quick test batch
            _fx, _fy = next(iter(train_loader))
            print(f"Sample batch -> X:{_fx.shape}  y:{_fy.shape}")
        except Exception as e:
            print("DataLoader failed with multiprocessing. Falling back to num_workers=0.")
            print(f"Original exception: {e}")
            requested_num_workers = 0
            train_loader = make_loader(train_dataset, is_train=True)
            test_loader  = make_loader(test_dataset,  is_train=False)

        # Build model
        model = SpeedEstimatorGRU(
            input_size=detected_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout_rate=dropout_rate,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

        # Early stopping state
        best_val    = float("inf")
        best_epoch  = -1
        early_count = 0

        # File path per config (will overwrite on improvements)
        save_prefix = os.path.join(save_prefix_root, f"model_GRU_lon_{cfg_id}")
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        ckpt_path = f"{save_prefix}.pt"

        for epoch in range(num_epochs):
            # -------- Train --------
            model.train()
            total = 0.0
            nsteps = 0
            for features, speeds in train_loader:
                features = features.to(device, non_blocking=True)  # [B, T, F]
                speeds   = speeds.to(device, non_blocking=True)    # [B, 1]

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    outputs = model(features)          # [B, 1]
                    loss = criterion(outputs, speeds)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total += loss.item()
                nsteps += 1

            train_loss = total / max(1, nsteps)

            # -------- Validate --------
            model.eval()
            vtotal = 0.0
            vsteps = 0
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                for features, speeds in test_loader:
                    features = features.to(device, non_blocking=True)
                    speeds   = speeds.to(device, non_blocking=True)
                    outputs  = model(features)
                    vloss    = criterion(outputs, speeds)
                    vtotal  += vloss.item()
                    vsteps  += 1
            val_loss = vtotal / max(1, vsteps)

            print(f"[ID={cfg_id}] Epoch [{epoch+1}/{num_epochs}]  train={train_loss:.6f}  val={val_loss:.6f}")

            # -------- Checkpoint on improvement --------
            if val_loss < best_val:
                print(f"  >> Improved from {best_val:.6f} to {val_loss:.6f}")
                best_val = val_loss
                best_epoch = epoch
                early_count = 0

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "sequence_length": seq_length,
                    "input_size": detected_input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "output_size": output_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "step_size": step_size,
                    "dropout_rate": dropout_rate
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")
            else:
                early_count += 1
                print(f"  No improvement ({early_count}/{patience})")
                if early_count >= patience:
                    print("  Early stopping.")
                    break

        print(f"[ID={cfg_id}] Best val {best_val:.6f} at epoch {best_epoch+1 if best_epoch>=0 else 'N/A'}")

        # -------- ONNX Export (only once, best model) --------
        if export_onnx and best_epoch >= 0:
            onnx_path = f"{save_prefix}.onnx"
            print(f"[ID={cfg_id}] Exporting best model to ONNX: {onnx_path}")

            model_cpu = SpeedEstimatorGRU(
                input_size=detected_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout_rate=dropout_rate,
            ).cpu()
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model_cpu.load_state_dict(ckpt["model_state_dict"])
            model_cpu.eval()

            example_input = torch.randn(1, seq_length, detected_input_size, dtype=torch.float32)
            torch.onnx.export(
                model_cpu,
                example_input,
                onnx_path,
                export_params=True,
                opset_version=onnx_opset,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable batch, fixed seq_len/features
            )
            print(f"[ID={cfg_id}] ONNX saved: {onnx_path}")

    print("\nAll configurations processed.")


if __name__ == "__main__":
    # For Windows: if you still get spawn issues, uncomment:
    # import multiprocessing as mp; mp.set_start_method("spawn", force=True)
    main()