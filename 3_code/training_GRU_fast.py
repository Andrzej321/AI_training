import os
import math
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import DataLoader

from classes import SpeedEstimatorTCN
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


def parse_list_cell(val, cast=int) -> Optional[List[int]]:
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


def main():
    # -------------------- Paths --------------------
    training_data_path = "../1_data/i7/it_1/it_1_100_norm/1_training"
    test_data_path     = "../1_data/i7/it_1/it_1_100_norm/2_testing"
    hyperparams_csv    = "../2_trained_models/TCN/i7/it_2_norm/hyperparams_TCN_it_2.csv"

    # Output locations (prefixes)
    location_state_TCN  = "../2_trained_models/TCN/i7/it_2_norm/state_models/lon/model_TCN_lon_"
    location_traced_TCN = "../2_trained_models/TCN/i7/it_2_norm/traced_models/lon/model_TCN_lon_"
    os.makedirs(os.path.dirname(location_state_TCN), exist_ok=True)
    os.makedirs(os.path.dirname(location_traced_TCN), exist_ok=True)

    # -------------------- Fixed / defaults --------------------
    fixed_input_size   = 12
    fixed_dropout      = 0.1
    fixed_step_size    = 5

    default_learning_rate = 1e-4
    default_weight_decay  = 0.0
    default_optimizer     = "adam"       # adam | adamw | sgd
    default_loss          = "mse"        # mse | smooth_l1 | mae
    default_grad_clip     = 1.0
    default_batch_size    = 128
    default_epochs        = 100
    patience              = 10
    min_delta             = 0.0
    default_seed          = 42

    # Model behavior defaults
    default_use_weight_norm = True
    default_activation      = "relu"
    default_norm_in_block   = "none"
    default_head_pooling    = "last"
    default_causal          = True
    default_output_clamp_min = None

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

    # -------------------- Read hyperparameters table --------------------
    print(f"Reading hyperparameters from {hyperparams_csv}")
    df = pd.read_csv(hyperparams_csv, delimiter=";")

    cols_lc = {c.lower(): c for c in df.columns}
    def get_col(name: str) -> Optional[str]:
        return cols_lc.get(name.lower(), None)

    col_sequence_size     = get_col("sequence_size") or get_col("sequence_length")
    col_kernel_size       = get_col("kernel_size")
    col_num_residual      = get_col("num_residual_blocks") or get_col("num_of_layers")
    col_conv_per_block    = get_col("convolutions_per_block")
    col_channels_per_layer= get_col("channels_per_layer")
    col_dilation_schedule = get_col("dilation_schedule")
    col_use_weight_norm   = get_col("use_weight_norm")
    col_activation        = get_col("activation")
    col_norm_in_block     = get_col("norm_in_block")
    col_head_pooling      = get_col("head_pooling")
    col_causal            = get_col("causal")
    col_output_clamp_min  = get_col("output_clamp_min")
    col_learning_rate     = get_col("learning_rate")
    col_weight_decay      = get_col("weight_decay")
    col_optimizer         = get_col("optimizer")
    col_loss              = get_col("loss")
    col_grad_clip         = get_col("grad_clip")
    col_batch_size        = get_col("batch_size")
    col_epochs            = get_col("epochs")
    col_seed              = get_col("seed")
    col_hidden_size       = get_col("hidden_size")
    col_id                = get_col("ID") or get_col("id")
    col_input_size        = get_col("input_size")  # optional consistency check

    for row_idx, row in df.iterrows():
        # -------------------- Extract per-row config --------------------
        sequence_length = int(row[col_sequence_size]) if col_sequence_size else 100
        kernel_size = int(row[col_kernel_size]) if col_kernel_size else 3
        num_residual_blocks = int(row[col_num_residual]) if col_num_residual else 4
        convolutions_per_block = int(row[col_conv_per_block]) if col_conv_per_block else 2

        channels_per_layer = parse_list_cell(row[col_channels_per_layer]) if col_channels_per_layer else None
        dilation_schedule  = parse_list_cell(row[col_dilation_schedule]) if col_dilation_schedule else None

        if channels_per_layer is None:
            hidden_size = int(row[col_hidden_size]) if col_hidden_size else 64
            channels_per_layer = [hidden_size] * num_residual_blocks

        if dilation_schedule is None:
            dilation_schedule = [2 ** i for i in range(num_residual_blocks)]

        use_weight_norm = to_bool(row[col_use_weight_norm], default_use_weight_norm) if col_use_weight_norm else default_use_weight_norm
        activation      = str(row[col_activation]).lower() if col_activation and isinstance(row[col_activation], str) else default_activation
        norm_in_block   = str(row[col_norm_in_block]).lower() if col_norm_in_block and isinstance(row[col_norm_in_block], str) else default_norm_in_block
        head_pooling    = str(row[col_head_pooling]).lower() if col_head_pooling and isinstance(row[col_head_pooling], str) else default_head_pooling
        causal          = to_bool(row[col_causal], default_causal) if col_causal else default_causal

        output_clamp_min = None
        if col_output_clamp_min:
            v = row[col_output_clamp_min]
            if not (isinstance(v, float) and math.isnan(v)):
                try:
                    output_clamp_min = float(v)
                except Exception:
                    output_clamp_min = default_output_clamp_min

        learning_rate = float(row[col_learning_rate]) if col_learning_rate else default_learning_rate
        weight_decay  = float(row[col_weight_decay]) if col_weight_decay else default_weight_decay
        optimizer_name = str(row[col_optimizer]).lower() if col_optimizer else default_optimizer
        loss_name     = str(row[col_loss]).lower() if col_loss else default_loss
        grad_clip     = float(row[col_grad_clip]) if col_grad_clip else default_grad_clip
        batch_size    = int(row[col_batch_size]) if col_batch_size else default_batch_size
        num_epochs    = int(row[col_epochs]) if col_epochs else default_epochs
        seed          = int(row[col_seed]) if col_seed else default_seed
        cfg_id        = int(row[col_id]) if col_id else row_idx

        input_size_expected = int(row[col_input_size]) if col_input_size else None

        print("\n================================================")
        print(f"TCN Config row={row_idx} (ID={cfg_id}) seq_len={sequence_length} kernel={kernel_size} residual_blocks={num_residual_blocks}")
        print("================================================")

        # -------------------- Seeding --------------------
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # -------------------- Cached datasets --------------------
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
            requested_num_workers = 0
            train_loader = make_loader(train_dataset, is_train=True)
            test_loader  = make_loader(test_dataset,  is_train=False)

        # -------------------- Model --------------------
        model = SpeedEstimatorTCN(
            input_size=detected_input_size,
            output_size=1,
            channels_per_layer=channels_per_layer,
            num_residual_blocks=len(channels_per_layer),
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

        # -------------------- Loss --------------------
        if loss_name in ("smooth_l1", "huber"):
            criterion = nn.SmoothL1Loss()
        elif loss_name in ("mae", "l1"):
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()

        # -------------------- Optimizer --------------------
        if optimizer_name == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

        # -------------------- Checkpoint paths --------------------
        ckpt_state_path = f"{location_state_TCN}{cfg_id}.pt"
        best_val = float("inf")
        early_count = 0
        best_epoch = -1

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

                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

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
                    "sequence_length": sequence_length,
                    "input_size": detected_input_size,
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
            onnx_path = f"{location_traced_TCN}{cfg_id}.onnx"
            print(f"[ID={cfg_id}] Exporting best model to ONNX: {onnx_path}")

            model_cpu = SpeedEstimatorTCN(
                input_size=detected_input_size,
                output_size=1,
                channels_per_layer=channels_per_layer,
                num_residual_blocks=len(channels_per_layer),
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

    print("\nAll TCN configurations processed.")


if __name__ == "__main__":
    main()