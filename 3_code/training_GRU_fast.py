import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from classes import SpeedEstimatorGRU
from datasets_cached import VehicleSpeedDatasetLongCached  # Provided earlier

# -------------------- Top-level collate function (must be picklable on Windows) --------------------
def collate_fn(batch):
    xs, ys = zip(*batch)          # xs: list of [T,F], ys: list of [1]
    x = torch.stack(xs, dim=0)    # [B, T, F]
    y = torch.stack(ys, dim=0)    # [B, 1]
    return x, y

# Optional: if you need full reproducibility across worker processes
def seed_worker(worker_id):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    # You can seed python's random here too if needed.

def main():
    # -------------------- CONFIG --------------------
    training_data_path = "../1_data/i7/it_1/it_1_100_norm/1_training"
    test_data_path     = "../1_data/i7/it_1/it_1_100_norm/2_testing"

    drop_columns = [
        "veh_u", "veh_v", "Time",
        "imu_COG_acc_z", "imu_COG_gyro_roll_rate", "imu_COG_gyro_pitch_rate",
        "drive_torque_FR", "drive_torque_RR", "brake_pressure_FR", "brake_pressure_RR",
        "rwa_RM"
    ]
    target_column = "veh_u"

    seq_length    = 100
    hidden_size   = 128
    num_layers    = 2
    input_size    = 12      # Must match features after drops
    output_size   = 1
    learning_rate = 1e-4
    batch_size    = 128
    num_epochs    = 60
    patience      = 5

    # Performance knobs
    requested_num_workers = 8          # Try >0 for speed; fallback to 0 if issues
    pin_memory            = True       # Good if using CUDA
    persistent_workers    = True       # Only works if num_workers > 0
    prefetch_factor       = 4
    use_amp               = True       # Mixed precision
    grad_clip             = 0.0        # Set >0 to enable gradient clipping
    log_every_batches     = 0          # Set >0 to print intermediate batch losses

    # Heavy exports disabled for speed
    do_tracing = False
    do_onnx    = False

    # Checkpoint path
    state_path_prefix = "../2_trained_models/GRU/trained_models/i7/it_fast/state_models/lon/model_GRU_lon_fast"
    os.makedirs(os.path.dirname(state_path_prefix), exist_ok=True)

    # -------------------- Device --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # -------------------- Dataset Load (Cached) --------------------
    print("Loading training dataset (cached)...")
    train_dataset = VehicleSpeedDatasetLongCached(
        training_data_path,
        extension="*.csv",
        seq_length=seq_length,
        step_size=5,
        drop_columns=drop_columns,
        target_column=target_column,
    )
    print("Loading validation/test dataset (cached)...")
    test_dataset = VehicleSpeedDatasetLongCached(
        test_data_path,
        extension="*.csv",
        seq_length=seq_length,
        step_size=5,
        drop_columns=drop_columns,
        target_column=target_column,
    )

    assert train_dataset.input_size == input_size, (
        f"Detected input_size={train_dataset.input_size}, "
        f"but configured input_size={input_size}. Adjust input_size or drop_columns."
    )

    # -------------------- DataLoader Safe Construction --------------------
    def make_loader(ds, is_train):
        workers = requested_num_workers
        if workers < 0:
            workers = 0

        # persistent_workers must be False if workers==0
        pw = persistent_workers and workers > 0

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

        # Quick integrity check (pull one batch)
        test_iter = iter(train_loader)
        feat_chk, speed_chk = next(test_iter)
        print(f"Initial batch: features {feat_chk.shape}, speeds {speed_chk.shape}")
    except Exception as e:
        print("DataLoader failed with multiprocessing. Falling back to num_workers=0.")
        print(f"Original exception: {e}")
        requested_num_workers = 0
        train_loader = make_loader(train_dataset, is_train=True)
        test_loader  = make_loader(test_dataset,  is_train=False)

    # -------------------- Model / Optim / AMP --------------------
    model = SpeedEstimatorGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout_rate=0.0,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    # -------------------- Training Loop --------------------
    best_val     = float("inf")
    best_epoch   = -1
    early_count  = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        steps = 0

        for batch_idx, (features, speeds) in enumerate(train_loader):
            features = features.to(device, non_blocking=True)   # [B, T, F]
            speeds   = speeds.to(device, non_blocking=True)     # [B, 1]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                outputs = model(features)  # Expect [B, 1]
                loss = criterion(outputs, speeds)

            scaler.scale(loss).backward()

            if grad_clip > 0.0:
                # Unscale first if using AMP
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            steps += 1

            if log_every_batches > 0 and (batch_idx + 1) % log_every_batches == 0:
                print(f"  Batch {batch_idx+1}: loss={loss.item():.6f}")

        train_loss = total_loss / max(1, steps)

        # Validation
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

        print(f"Epoch [{epoch+1}/{num_epochs}] train={train_loss:.6f} val={val_loss:.6f}")

        # Checkpoint on improvement
        if val_loss < best_val:
            print(f"  >> Improved from {best_val:.6f} to {val_loss:.6f}")
            best_val = val_loss
            best_epoch = epoch
            early_count = 0

            ckpt_path = f"{state_path_prefix}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "sequence_length": seq_length,
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "output_size": output_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            if (do_tracing or do_onnx) and device.type == "cuda":
                example_input = torch.randn(1, seq_length, input_size, device=device)
                if do_tracing:
                    traced = torch.jit.trace(model, example_input)
                    torch.jit.save(traced, f"{state_path_prefix}_traced_jit_save.pt")
                if do_onnx:
                    onnx_path = f"{state_path_prefix}.onnx"
                    torch.onnx.export(
                        model,
                        example_input,
                        onnx_path,
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes=None,
                    )
                    print(f"  Exported ONNX: {onnx_path}")
        else:
            early_count += 1
            print(f"  No improvement ({early_count}/{patience})")
            if early_count >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best val loss {best_val:.6f} at epoch {best_epoch+1 if best_epoch>=0 else 'N/A'}")


if __name__ == "__main__":
    # On Windows, if you still see spawn-related issues, uncomment the line below:
    # import multiprocessing as mp; mp.set_start_method("spawn", force=True)
    main()