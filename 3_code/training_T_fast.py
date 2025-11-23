import os
from typing import Optional, List, Tuple

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


# -------------------- Helpers to safely read ints/floats --------------------
def get_int(row, name, default):
    return int(row[name]) if name in row and pd.notna(row[name]) else default

def get_float(row, name, default):
    return float(row[name]) if name in row and pd.notna(row[name]) else default

def to_bool(val, default=False):
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


class EMAHelper:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)


def main():
    # -------------------- Paths --------------------
    training_data_path = "../1_data/i7/it_1/it_1_100_norm/1_training"
    test_data_path     = "../1_data/i7/it_1/it_1_100_norm/2_testing"
    hyperparams_csv    = "../2_trained_models/Transformer/i7/it_2_norm/hyperparams_T_it_2_2_3.csv"

    # Output locations (prefixes)
    location_state  = "../2_trained_models/Transformer/i7/it_2_norm/state_models/lon/model_T_lon_"
    location_onnx   = "../2_trained_models/Transformer/i7/it_2_norm/traced_models/lon/model_T_lon_"
    os.makedirs(os.path.dirname(location_state), exist_ok=True)
    os.makedirs(os.path.dirname(location_onnx), exist_ok=True)

    # -------------------- Fixed / defaults --------------------
    fixed_step_size = 5

    # Training defaults
    default_learning_rate   = 1e-4
    default_weight_decay    = 0.0
    default_batch_size      = 128
    default_epochs          = 100
    patience                = 5
    min_delta               = 0.0
    default_seed            = 42

    # Transformer architecture defaults
    default_d_model         = 128
    default_nhead           = 4
    default_dim_feedforward = 256
    default_num_layers      = 2
    default_dropout         = 0.1

    # Additional options exposed via CSV if present
    default_loss            = "mse"      # mse | smooth_l1 | mae

    # -------------------- Enhancement toggles --------------------
    use_amp               = True         # mixed precision
    use_compile           = True         # torch.compile (PyTorch >=2)
    use_cosine_scheduler  = False
    use_plateau_scheduler = True         # Ignored if cosine True
    warmup_epochs         = 5            # 0 disables linear warmup
    grad_clip_val         = 1.0          # None or 0 disables
    ema_decay             = 0.0          # e.g., 0.999 to enable EMA
    grad_accum_steps      = 1            # >1 for gradient accumulation
    print_every_batches   = 0            # >0 for intra-epoch logging
    max_batches_per_epoch = None         # int to cap training batches early (debug)
    filter_for_model_type = True         # Only rows with model_type containing 'transformer'

    # -------------------- DataLoader performance knobs --------------------
    requested_num_workers = 8
    pin_memory            = True
    persistent_workers    = True
    prefetch_factor       = 4

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
        print("Enabled cudnn.benchmark for performance.")

    # -------------------- Read and filter hyperparameters --------------------
    print(f"Reading hyperparameters from {hyperparams_csv}")
    df = pd.read_csv(hyperparams_csv, delimiter=";")

    if filter_for_model_type and "model_type" in df.columns:
        before = len(df)
        df = df[df["model_type"].astype(str).str.lower().str.contains("transformer")]
        print(f"Filtered model_type for 'transformer': {before} -> {len(df)} rows")
        if len(df) == 0:
            raise ValueError("No Transformer rows found in hyperparams CSV.")

    # Normalize keys to robustly access columns
    cols_lc = {c.lower(): c for c in df.columns}
    def col(name: str) -> Optional[str]:
        return cols_lc.get(name.lower())

    c_seq_len   = col("sequence_size") or col("sequence_length")
    c_d_model   = col("d_model")
    c_nhead     = col("nhead")
    c_dim_ff    = col("dim_feedforward") or col("dim_ff")
    c_layers    = col("num_of_layers") or col("num_layers") or col("encoder_layers")
    c_dropout   = col("dropout")
    c_lr        = col("learning_rate") or col("lr")
    c_wd        = col("weight_decay")
    c_batch     = col("batch_size")
    c_epochs    = col("epochs")
    c_seed      = col("seed")
    c_id        = col("ID") or col("id")
    c_input_sz  = col("input_size")
    c_loss      = col("loss")
    c_grad_clip = col("grad_clip")
    c_ema_decay = col("ema_decay")
    c_accum     = col("grad_accum_steps")

    df = df.reset_index(drop=True)

    # -------------------- Dataset cache: (seq_length, step_size) -> (train_ds, test_ds, input_size) --------------------
    dataset_cache: dict[Tuple[int, int], Tuple[VehicleSpeedDatasetLongCached,
                                               VehicleSpeedDatasetLongCached,
                                               int]] = {}

    starting_point = 712

    for j, row in df.iterrows():

        if j < starting_point: continue

        # -------------------- Extract per-row config --------------------
        seq_len          = get_int(row, c_seq_len, 100) if c_seq_len else 100
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

        loss_name        = str(row[c_loss]).lower() if c_loss and pd.notna(row[c_loss]) else default_loss
        local_grad_clip  = get_float(row, c_grad_clip, grad_clip_val) if c_grad_clip else grad_clip_val
        local_ema_decay  = get_float(row, c_ema_decay, ema_decay) if c_ema_decay else ema_decay
        local_accum      = get_int(row, c_accum, grad_accum_steps) if c_accum else grad_accum_steps

        print("\n================================================")
        print(f"Transformer Config row={j} (ID={cfg_id}) seq_len={seq_len} d_model={d_model} nhead={nhead} dim_feedforward={dim_feedforward} layers={num_layers}")
        print("================================================")

        # -------------------- Seeding --------------------
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # -------------------- Dataset reuse or creation --------------------
        dataset_key = (seq_len, fixed_step_size)
        if dataset_key in dataset_cache:
            train_dataset, test_dataset, detected_input_size = dataset_cache[dataset_key]
            print(f"[Reuse] Cached datasets for seq_length={seq_len}, step_size={fixed_step_size}")
        else:
            print(f"[Build] Creating datasets for seq_length={seq_len}, step_size={fixed_step_size}")
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
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=(pin_memory and device.type == "cuda"),
                collate_fn=collate_fn
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=(pin_memory and device.type == "cuda"),
                collate_fn=collate_fn
            )

        # -------------------- Model --------------------
        model = SpeedEstimatorTransformer(
            input_size=detected_input_size,
            d_model=d_model,
            num_layers=num_layers,
            output_size=1,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device)

        if use_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("Model compiled with torch.compile()")
            except Exception as e:
                print(f"torch.compile failed -> continuing without compile. Reason: {e}")

        # -------------------- Loss --------------------
        if loss_name in ("smooth_l1", "huber"):
            criterion = nn.SmoothL1Loss()
        elif loss_name in ("mae", "l1"):
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()

        # -------------------- Optimizer --------------------
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # -------------------- Scheduler --------------------
        scheduler = None
        if use_cosine_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            print("Using CosineAnnealingLR scheduler.")
        elif use_plateau_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
            print("Using ReduceLROnPlateau scheduler.")

        # -------------------- AMP / EMA / GradScaler --------------------
        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
        ema = EMAHelper(model, local_ema_decay) if local_ema_decay and local_ema_decay > 0 else None

        # -------------------- Helper: Warmup factor --------------------
        def lr_warmup_factor(ep: int):
            if warmup_epochs <= 0 or ep >= warmup_epochs:
                return 1.0
            return (ep + 1) / warmup_epochs

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

            # Warmup LR adjustment
            if warmup_epochs > 0 and epoch < warmup_epochs:
                scaled_lr = learning_rate * lr_warmup_factor(epoch)
                for pg in optimizer.param_groups:
                    pg["lr"] = scaled_lr
            elif scheduler is None and warmup_epochs > 0 and epoch == warmup_epochs:
                for pg in optimizer.param_groups:
                    pg["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)

            for batch_idx, (features, speeds) in enumerate(train_loader):
                features = features.to(device, non_blocking=True)
                speeds   = speeds.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    outputs = model(features)
                    loss = criterion(outputs, speeds)

                scaler.scale(loss).backward()

                if local_grad_clip and local_grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), local_grad_clip)

                if (batch_idx + 1) % local_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema:
                        ema.update(model)

                running += loss.item()
                steps += 1

                if print_every_batches > 0 and (batch_idx + 1) % print_every_batches == 0:
                    print(f"  Batch {batch_idx+1} loss={loss.item():.6f}")

                if max_batches_per_epoch and steps >= max_batches_per_epoch:
                    break

            # Handle leftover accumulation
            if local_accum > 1 and steps % local_accum != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema:
                    ema.update(model)

            train_loss = running / max(1, steps)

            # -------------------- Validation --------------------
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
                best_val = val_loss
                best_epoch = epoch
                early_count = 0

                # Prepare model for saving (use EMA weights if enabled)
                save_model = model
                if ema:
                    ema_model = SpeedEstimatorTransformer(
                        input_size=detected_input_size,
                        d_model=d_model,
                        num_layers=num_layers,
                        output_size=1,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                    ).to(device)
                    ema.apply_to(ema_model)
                    save_model = ema_model

                torch.save({
                    "model_state_dict": save_model.state_dict(),
                    "sequence_length": seq_len,
                    "input_size": detected_input_size,
                    "d_model": d_model,
                    "nhead": nhead,
                    "dim_feedforward": dim_feedforward,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "epochs": num_epochs,
                    "seed": seed,
                    "best_val_loss": best_val,
                    "loss_name": loss_name,
                    "scheduler": "cosine" if use_cosine_scheduler else ("plateau" if use_plateau_scheduler else "none"),
                    "warmup_epochs": warmup_epochs,
                    "compile_used": use_compile and hasattr(torch, 'compile'),
                    "ema_decay": local_ema_decay,
                    "grad_accum_steps": local_accum,
                    "grad_clip_val": local_grad_clip,
                }, ckpt_state_path)
                print(f"  Saved checkpoint: {ckpt_state_path}")
            else:
                early_count += 1
                print(f"  No improvement ({early_count}/{patience})")
                if early_count >= patience:
                    print("  Early stopping.")
                    break

        print(f"[ID={cfg_id}] Best val={best_val:.6f} at epoch {best_epoch+1 if best_epoch>=0 else 'N/A'}")

        # -------------------- ONNX Export --------------------
        if export_onnx and best_epoch >= 0:
            onnx_path = f"{location_onnx}{cfg_id}.onnx"
            print(f"[ID={cfg_id}] Exporting best model to ONNX: {onnx_path}")

            # Load checkpoint weights (EMA if saved)
            model_cpu = SpeedEstimatorTransformer(
                input_size=detected_input_size,
                d_model=d_model,
                num_layers=num_layers,
                output_size=1,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
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