import os
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import DataLoader

from classes import SpeedEstimatorTransformer
from datasets_cached import VehicleSpeedDatasetLongCached


# -------------------- Collate (picklable) --------------------
def collate_fn(batch):
    xs, ys = zip(*batch)          # xs: list of [T,F], ys: list of [1]
    x = torch.stack(xs, dim=0)    # [B, T, F]
    y = torch.stack(ys, dim=0)    # [B, 1]
    return x, y


def seed_worker(worker_id):
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)


def get_int(row, name, default):
    return int(row[name]) if name in row and pd.notna(row[name]) else default


def get_float(row, name, default):
    return float(row[name]) if name in row and pd.notna(row[name]) else default


def main():
    # -------------------- Paths --------------------
    training_data_path = "../1_data/i7/it_1/it_1_100_norm/1_training"
    test_data_path     = "../1_data/i7/it_1/it_1_100_norm/2_testing"
    hyperparams_csv    = "../2_trained_models/Transformer/i7/it_1_norm/hyperparams_T_it_4.csv"

    # Output locations (prefixes)
    location_state  = "../2_trained_models/Transformer/i7/it_1_norm/state_models/lon/model_T_lon_"
    location_onnx   = "../2_trained_models/Transformer/i7/it_1_norm/traced_models/lon/model_T_lon_"
    os.makedirs(os.path.dirname(location_state), exist_ok=True)
    os.makedirs(os.path.dirname(location_onnx), exist_ok=True)

    # -------------------- Fixed / defaults --------------------
    fixed_step_size = 5

    # Training defaults
    default_learning_rate = 1e-4
    default_weight_decay  = 0.0
    default_batch_size    = 128
    default_epochs        = 100
    patience              = 5
    min_delta             = 0.0
    default_seed          = 42

    # Transformer defaults (if not in CSV)
    default_d_model       = 128
    default_nhead         = 4
    default_dim_feedforward = 256
    default_num_layers    = 2
    default_dropout       = 0.1

    # -------------------- Performance knobs --------------------
    requested_num_workers = 8
    pin_memory            = True
    persistent_workers    = True
    prefetch_factor       = 4
    use_amp               = True  # mixed precision

    # -------------------- Export knobs --------------------
    export_onnx = True
    onnx_opset  = 11

    # -------------------- Dataset column behavior --------------------
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

    # -------------------- Read and filter hyperparameters --------------------
    print(f"Reading hyperparameters from {hyperparams_csv}")
    df = pd.read_csv(hyperparams_csv, delimiter=";")

    # Filter rows for Transformer if model_type column exists
    if "model_type" in df.columns:
        df = df[df["model_type"].astype(str).str.lower().str.contains("transformer")]
        if len(df) == 0:
            raise ValueError("No Transformer rows found in hyperparams CSV (model_type contains 'Transformer').")

    # Normalize keys to robustly access columns
    cols_lc = {c.lower(): c for c in df.columns}
    def col(name: str) -> Optional[str]:
        return cols_lc.get(name.lower())

    c_seq_len   = col("sequence_size") or col("sequence_length")
    c_d_model   = col("d_model")
    c_nhead     = col("nhead")
    c_dim_ff    = col("dim_ff")  # CSV may name it 'dim_ff'
    c_layers    = col("num_of_layers") or col("num_layers") or col("encoder_layers")
    c_dropout   = col("dropout")
    c_lr        = col("learning_rate") or col("lr")
    c_wd        = col("weight_decay")
    c_batch     = col("batch_size")
    c_epochs    = col("epochs")
    c_seed      = col("seed")
    c_id        = col("ID") or col("id")
    c_input_sz  = col("input_size")  # optional sanity check

    df = df.reset_index(drop=True)

    # -------------------- Dataset cache: (seq_length, step_size) -> (train_ds, test_ds, input_size) --------------------
    dataset_cache = {}

    for j, row in df.iterrows():
        # -------------------- Extract per-row config --------------------
        seq_len          = int(row[c_seq_len]) if c_seq_len else 100
        d_model          = get_int(row, c_d_model, default_d_model) if c_d_model else default_d_model
        nhead            = get_int(row, c_nhead, default_nhead) if c_nhead else default_nhead
        dim_feedforward  = get_int(row, c_dim_ff, default_dim_feedforward) if c_dim_ff else default_dim_feedforward
        num_layers       = get_int(row, c_layers, default_num_layers) if c_layers else default_num_layers
        dropout          = get_float(row, c_dropout, default_dropout) if c_dropout else default_dropout

        learning_rate    = get_float(row, c_lr, default_learning_rate) if c_lr else default_learning_rate
        weight_decay     = get_float(row, c_wd, default_weight_decay) if c_wd else default_weight_decay
        batch_size       = get_int(row, c_batch, default_batch_size) if c_batch else default_batch_size
        num_epochs       = get_int(row, c_epochs, default_epochs) if c_epochs else default_epochs
        seed             = get_int(row, c_seed, default_seed) if c_seed else default_seed
        cfg_id           = get_int(row, c_id, j) if c_id else j
        input_size_expected = get_int(row, c_input_sz, None) if c_input_sz else None

        print("\n================================================")
        print(f"Transformer Config row={j} (ID={cfg_id}) seq_len={seq_len} d_model={d_model} nhead={nhead} dim_feedforward={dim_feedforward} layers={num_layers}")
        print("================================================")

        # -------------------- Seeding --------------------
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # -------------------- Dataset reuse or creation --------------------
        dataset_key = (seq_len, fixed_step_size)
        if dataset_key in dataset_cache:
            train_dataset, test_dataset, detected_input_size = dataset_cache[dataset_key]
            print(f"Reusing cached datasets for seq_length={seq_len}, step_size={fixed_step_size}")
        else:
            print(f"Creating datasets for seq_length={seq_len}, step_size={fixed_step_size}")
            train_dataset = VehicleSpeedDatasetLongCached(
                training_data_path,
                extension="*.csv",
                seq_length=seq_len,
                step_size=fixed_step_size,
                drop_columns=drop_columns,
                target_column=target_column,
            )
            test_dataset = VehicleSpeedDatasetLongCached(
                test_data_path,
                extension="*.csv",
                seq_length=seq_len,
                step_size=fixed_step_size,
                drop_columns=drop_columns,
                target_column=target_column,
            )
            detected_input_size = train_dataset.input_size
            dataset_cache[dataset_key] = (train_dataset, test_dataset, detected_input_size)

        if input_size_expected is not None:
            assert detected_input_size == input_size_expected, (
                f"CSV input_size={input_size_expected} differs from detected={detected_input_size}. "
                f"Adjust CSV or drop_columns."
            )

        # -------------------- DataLoader factory --------------------
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

        # -------------------- Model --------------------
        model = SpeedEstimatorTransformer(
            input_size=detected_input_size,
            d_model=d_model,
            num_layers=num_layers,
            output_size=1,
            nhead=nhead,
            dim_feedforward=dim_feedforward,  # FIX: correct kw for constructor
            dropout=dropout,
        ).to(device)

        # -------------------- Loss and Optimizer --------------------
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

        # -------------------- Checkpointing --------------------
        ckpt_state_path = f"{location_state}{cfg_id}.pt"
        best_val = float("inf")
        best_epoch = -1
        early_count = 0

        # -------------------- Training loop --------------------
        for epoch in range(num_epochs):
            model.train()
            running = 0.0
            steps = 0

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

            print(f"[ID={cfg_id}] Epoch [{epoch+1}/{num_epochs}] train={train_loss:.6f} val={val_loss:.6f}")

            improved = (best_val - val_loss) > min_delta
            if improved:
                print(f"  >> Improved from {best_val:.6f} to {val_loss:.6f}")
                best_val = val_loss
                best_epoch = epoch
                early_count = 0

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "sequence_length": seq_len,
                    "input_size": detected_input_size,
                    "d_model": d_model,
                    "nhead": nhead,
                    "dim_feedforward": dim_feedforward,  # store for reproducibility
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "epochs": num_epochs,
                    "seed": seed,
                    "best_val_loss": best_val,
                }, ckpt_state_path)
                print(f"  Saved checkpoint: {ckpt_state_path}")
            else:
                early_count += 1
                print(f"  No improvement ({early_count}/{patience})")
                if early_count >= patience:
                    print("  Early stopping.")
                    break

        print(f"[ID={cfg_id}] Best val={best_val:.6f} at epoch {best_epoch+1 if best_epoch>=0 else 'N/A'}")

        # -------------------- ONNX Export (only once after training) --------------------
        if export_onnx and best_epoch >= 0:
            onnx_path = f"{location_onnx}{cfg_id}.onnx"
            print(f"[ID={cfg_id}] Exporting best model to ONNX: {onnx_path}")

            model_cpu = SpeedEstimatorTransformer(
                input_size=detected_input_size,
                d_model=d_model,
                num_layers=num_layers,
                output_size=1,
                nhead=nhead,
                dim_feedforward=dim_feedforward,  # FIX: correct kw for constructor
                dropout=dropout,
            ).cpu()
            ckpt = torch.load(ckpt_state_path, map_location="cpu")
            model_cpu.load_state_dict(ckpt["model_state_dict"])
            model_cpu.eval()

            example_input = torch.randn(1, seq_len, detected_input_size, dtype=torch.float32)
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

    print("\nAll Transformer configurations processed.")


if __name__ == "__main__":
    main()