import os
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import DataLoader

from classes import SpeedEstimatorLSTM
from datasets_cached import VehicleSpeedDatasetLongCached


# -------------------- Collate (picklable) --------------------
def collate_fn(batch):
    xs, ys = zip(*batch)          # xs: list of [T,F], ys: list of [1]
    x = torch.stack(xs, dim=0)    # [B, T, F]
    y = torch.stack(ys, dim=0)    # [B, 1]
    return x, y


def seed_worker(worker_id):
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)


def main():
    # -------------------- Paths (adjust as needed) --------------------
    training_data_path = "../1_data/i7/it_1/it_1_100_norm/1_training"
    test_data_path     = "../1_data/i7/it_1/it_1_100_norm/2_testing"
    hyperparams_csv    = "../2_trained_models/LSTM/i7/it_3_norm/hyperparams_LSTM_it_3.csv"

    # Output locations (prefixes)
    location_state = "../2_trained_models/LSTM/trained_models/i7/it_3_norm/state_models/lon/model_LSTM_lon_"
    location_onnx  = "../2_trained_models/LSTM/trained_models/i7/it_3_norm/traced_models/lon/model_LSTM_lon_"
    os.makedirs(os.path.dirname(location_state), exist_ok=True)
    os.makedirs(os.path.dirname(location_onnx), exist_ok=True)

    # -------------------- Fixed / defaults --------------------
    fixed_step_size      = 5
    output_size          = 1
    learning_rate        = 1e-4
    default_batch_size   = 128
    default_epochs       = 150
    patience             = 5
    min_delta            = 0.0
    use_amp              = True       # Mixed precision
    use_compile          = True       # torch.compile (PyTorch 2.x)
    grad_clip_val        = 1.0        # Set to None to disable gradient clipping
    default_seed         = 42

    # Scheduler selection (choose at most one True)
    use_cosine_scheduler  = False
    use_plateau_scheduler = True

    # DataLoader performance knobs
    requested_num_workers = 8
    pin_memory            = True
    persistent_workers    = True
    prefetch_factor       = 4

    # ONNX export control
    export_onnx  = True
    onnx_opset   = 11

    # Filter hyperparameters CSV for LSTM rows
    filter_for_model_type = True

    # Dataset column behavior (single regression target: veh_u)
    target_column = "veh_u"
    drop_columns = [
        "veh_u", "veh_v", "Time",
        "imu_COG_acc_z", "imu_COG_gyro_roll_rate", "imu_COG_gyro_pitch_rate",
        "drive_torque_FR", "drive_torque_RR", "brake_pressure_FR", "brake_pressure_RR",
        "rwa_RM"
    ]

    # -------------------- Device --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Enabled cudnn.benchmark for performance.")

    # -------------------- Load hyperparameters table --------------------
    print(f"Reading hyperparameters from {hyperparams_csv}")
    df = pd.read_csv(hyperparams_csv, delimiter=";")

    if filter_for_model_type and "model_type" in df.columns:
        before_rows = len(df)
        df = df[df["model_type"].astype(str).str.lower().str.contains("lstm")]
        print(f"Filtered model_type for 'lstm': {before_rows} -> {len(df)} rows")
        if len(df) == 0:
            raise ValueError("No rows with model_type containing 'lstm' found.")

    df = df.reset_index(drop=True)

    # Normalize column access
    cols_lc = {c.lower(): c for c in df.columns}
    def col(name: str) -> Optional[str]:
        return cols_lc.get(name.lower())

    c_seq_len   = col("sequence_size") or col("sequence_length")
    c_hidden    = col("hidden_size")
    c_layers    = col("num_of_layers") or col("num_layers")
    c_dropout   = col("dropout_rate") or col("dropout")
    c_batch     = col("batch_size")
    c_epochs    = col("epochs") or col("num_epochs")
    c_seed      = col("seed")
    c_id        = col("ID") or col("id")
    c_input_sz  = col("input_size")

    # Dataset cache: key=(sequence_length, step_size)
    dataset_cache: dict[Tuple[int, int], Tuple[VehicleSpeedDatasetLongCached,
                                               VehicleSpeedDatasetLongCached,
                                               int]] = {}

    # -------------------- Iterate configurations --------------------
    for row_idx, row in df.iterrows():
        sequence_length = int(row[c_seq_len]) if c_seq_len else 100
        hidden_size     = int(row[c_hidden]) if c_hidden else 128
        num_layers      = int(row[c_layers]) if c_layers else 2
        dropout_rate    = float(row[c_dropout]) if c_dropout else 0.0
        batch_size      = int(row[c_batch]) if c_batch else default_batch_size
        num_epochs      = int(row[c_epochs]) if c_epochs else default_epochs
        seed            = int(row[c_seed]) if c_seed else default_seed
        cfg_id          = int(row[c_id]) if c_id else row_idx
        input_size_expected = int(row[c_input_sz]) if c_input_sz and pd.notna(row[c_input_sz]) else None

        print("\n================================================")
        print(f"LSTM Config row={row_idx} (ID={cfg_id}) seq_len={sequence_length} hidden={hidden_size} layers={num_layers} dropout={dropout_rate}")
        print("================================================")

        # Seeding
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # Dataset build / reuse
        ds_key = (sequence_length, fixed_step_size)
        if ds_key in dataset_cache:
            train_dataset, test_dataset, detected_input_size = dataset_cache[ds_key]
            print(f"[Reuse] Using cached datasets for seq_len={sequence_length}, step_size={fixed_step_size}")
        else:
            print(f"[Build] Creating datasets for seq_len={sequence_length}, step_size={fixed_step_size}")
            t0 = time.perf_counter()
            train_dataset = VehicleSpeedDatasetLongCached(
                training_data_path,
                extension="*.csv",
                seq_length=sequence_length,
                step_size=fixed_step_size,
                drop_columns=drop_columns,
                target_column=target_column,
            )
            test_dataset = VehicleSpeedDatasetLongCached(
                test_data_path,
                extension="*.csv",
                seq_length=sequence_length,
                step_size=fixed_step_size,
                drop_columns=drop_columns,
                target_column=target_column,
            )
            detected_input_size = train_dataset.input_size
            dataset_cache[ds_key] = (train_dataset, test_dataset, detected_input_size)
            t1 = time.perf_counter()
            print(f"[Build] Dataset construction took {(t1 - t0):.2f}s")

        if input_size_expected is not None:
            assert detected_input_size == input_size_expected, (
                f"CSV input_size={input_size_expected} differs from detected={detected_input_size}. "
                f"Adjust CSV or drop_columns."
            )

        # DataLoader factory
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
            _fx, _fy = next(iter(train_loader))
            print(f"Sample batch -> X:{_fx.shape} y:{_fy.shape}")
        except Exception as e:
            print("DataLoader multiprocessing failed; falling back to num_workers=0.")
            print(f"Original exception: {e}")
            def make_loader_fallback(ds, is_train):
                return DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=is_train,
                    num_workers=0,
                    pin_memory=(pin_memory and device.type == "cuda"),
                    collate_fn=collate_fn,
                    drop_last=False,
                )
            train_loader = make_loader_fallback(train_dataset, is_train=True)
            test_loader  = make_loader_fallback(test_dataset,  is_train=False)

        # Model
        model = SpeedEstimatorLSTM(
            input_size=detected_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout_rate=dropout_rate,
        ).to(device)

        if use_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("Model compiled with torch.compile()")
            except Exception as e:
                print(f"torch.compile failed -> continuing without compile. Reason: {e}")

        # Loss / Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Scheduler
        scheduler = None
        if use_cosine_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            print("Using CosineAnnealingLR scheduler.")
        elif use_plateau_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, verbose=True
            )
            print("Using ReduceLROnPlateau scheduler.")

        # AMP scaler
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

        # Checkpoint bookkeeping
        ckpt_state_path = f"{location_state}{cfg_id}.pt"
        best_val    = float("inf")
        best_epoch  = -1
        early_count = 0

        # -------------------- Training loop --------------------
        for epoch in range(num_epochs):
            model.train()
            running = 0.0
            steps = 0

            for features, speeds in train_loader:
                features = features.to(device, non_blocking=True)
                speeds   = speeds.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    outputs = model(features)
                    loss = criterion(outputs, speeds)

                scaler.scale(loss).backward()
                if grad_clip_val is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

                scaler.step(optimizer)
                scaler.update()

                running += loss.item()
                steps += 1

            train_loss = running / max(1, steps)

            # Validation
            model.eval()
            vtotal = 0.0
            vsteps = 0
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                for features, speeds in test_loader:
                    features = features.to(device, non_blocking=True)
                    speeds   = speeds.to(device, non_blocking=True)
                    vout     = model(features)
                    vloss    = criterion(vout, speeds)
                    vtotal  += vloss.item()
                    vsteps  += 1
            val_loss = vtotal / max(1, vsteps)

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[ID={cfg_id}] Epoch [{epoch+1}/{num_epochs}] lr={current_lr:.2e} train={train_loss:.6f} val={val_loss:.6f}")

            improved = (best_val - val_loss) > min_delta
            if improved:
                print(f"  >> Improved from {best_val:.6f} to {val_loss:.6f}")
                best_val   = val_loss
                best_epoch = epoch
                early_count = 0

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "sequence_length": sequence_length,
                    "input_size": detected_input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout_rate": dropout_rate,
                    "output_size": output_size,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": num_epochs,
                    "seed": seed,
                    "best_val_loss": best_val,
                    "model_type": "lstm",
                }, ckpt_state_path)
                print(f"  Saved checkpoint: {ckpt_state_path}")
            else:
                early_count += 1
                print(f"  No improvement ({early_count}/{patience})")
                if early_count >= patience:
                    print("  Early stopping.")
                    break

        print(f"[ID={cfg_id}] Best val={best_val:.6f} at epoch {best_epoch+1 if best_epoch>=0 else 'N/A'}")

        # -------------------- ONNX export --------------------
        if export_onnx and best_epoch >= 0:
            onnx_path = f"{location_onnx}{cfg_id}.onnx"
            print(f"[ID={cfg_id}] Exporting best model to ONNX: {onnx_path}")

            # Rebuild CPU model (avoid compiled wrapper)
            model_cpu = SpeedEstimatorLSTM(
                input_size=detected_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout_rate=dropout_rate,
            ).cpu()

            ckpt = torch.load(ckpt_state_path, map_location="cpu")
            model_cpu.load_state_dict(ckpt["model_state_dict"])
            model_cpu.eval()

            example_input = torch.randn(1, sequence_length, detected_input_size, dtype=torch.float32)
            torch.onnx.export(
                model_cpu,
                example_input,
                onnx_path,
                export_params=True,
                opset_version=onnx_opset,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            print(f"[ID={cfg_id}] ONNX saved: {onnx_path}")

    print("\nAll LSTM configurations processed.")


if __name__ == "__main__":
    main()