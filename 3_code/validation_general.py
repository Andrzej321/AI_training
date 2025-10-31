import os
import torch
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from classes import (
    FullFileForTestings,
    SpeedEstimatorRNN,
    SpeedEstimatorLSTM,
    SpeedEstimatorGRU,
    SpeedEstimatorTransformer,
    SpeedEstimatorTCN,
)


def get_chk_int(chk: Dict[str, Any], *keys, default: Optional[int] = None) -> Optional[int]:
    for k in keys:
        if k in chk and chk[k] is not None:
            try:
                return int(chk[k])
            except Exception:
                pass
    return default


def get_chk_float(chk: Dict[str, Any], *keys, default: Optional[float] = None) -> Optional[float]:
    for k in keys:
        if k in chk and chk[k] is not None:
            try:
                return float(chk[k])
            except Exception:
                pass
    return default


def get_chk_list(chk: Dict[str, Any], *keys, default: Optional[List[int]] = None) -> Optional[List[int]]:
    for k in keys:
        if k in chk and chk[k] is not None:
            v = chk[k]
            if isinstance(v, (list, tuple)):
                return list(v)
            # also accept comma-separated strings
            if isinstance(v, str):
                try:
                    return [int(x.strip()) for x in v.split(",")]
                except Exception:
                    pass
    return default


def build_model_from_checkpoint(
    model_type: str,
    checkpoint: Dict[str, Any],
    device: torch.device,
    fallback_input_size: int,
) -> Tuple[torch.nn.Module, int, int, int]:
    """
    Returns (model, seq_len, step_size, output_size) based on model_type and checkpoint.
    fallback_input_size is used if checkpoint does not include an input size.
    """
    mt = model_type.strip().lower()

    # Common keys
    seq_len = get_chk_int(checkpoint, "sequence_length", "seq_len")
    if seq_len is None:
        raise ValueError("sequence_length (or seq_len) is missing in checkpoint")

    step_size = get_chk_int(checkpoint, "step_size", "fixed_step", default=5)

    # Input/output sizes (input_size may be inferred from validation features)
    input_size = get_chk_int(checkpoint, "input_size", "fixed_input_size", "feature_count", default=fallback_input_size)
    output_size = get_chk_int(checkpoint, "output_size", default=1)

    if mt == "rnn":
        hidden_size = get_chk_int(checkpoint, "hidden_size", default=64)
        num_layers = get_chk_int(checkpoint, "num_layers", default=2)
        model = SpeedEstimatorRNN(input_size, hidden_size, num_layers, output_size).to(device)

    elif mt == "lstm":
        hidden_size = get_chk_int(checkpoint, "hidden_size", default=64)
        num_layers = get_chk_int(checkpoint, "num_layers", default=2)
        model = SpeedEstimatorLSTM(input_size, hidden_size, num_layers, output_size).to(device)

    elif mt == "gru":
        hidden_size = get_chk_int(checkpoint, "hidden_size", default=64)
        num_layers = get_chk_int(checkpoint, "num_layers", default=2)
        model = SpeedEstimatorGRU(input_size, hidden_size, num_layers, output_size).to(device)

    elif mt == "transformer":
        d_model = get_chk_int(checkpoint, "d_model", default=64)
        num_layers = get_chk_int(checkpoint, "num_layers", default=2)
        nhead = get_chk_int(checkpoint, "nhead", default=4)
        dim_feedforward = get_chk_int(checkpoint, "dim_feedforward", default=4 * d_model)
        dropout = get_chk_float(checkpoint, "dropout", "fixed_dropout", default=0.1) or 0.1
        model = SpeedEstimatorTransformer(
            input_size=input_size,
            d_model=d_model,
            num_layers=num_layers,
            output_size=output_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device)

    elif mt == "tcn":
        channels_per_layer = get_chk_list(checkpoint, "channels_per_layer")
        dilation_schedule = get_chk_list(checkpoint, "dilation_schedule")
        # Some trainings might save 'convolution_per_block' vs 'convolutions_per_block'
        convolutions_per_block = get_chk_int(checkpoint, "convolutions_per_block", "convolution_per_block", default=2)
        kernel_size = get_chk_int(checkpoint, "kernel_size", default=3)
        dropout = get_chk_float(checkpoint, "dropout", "fixed_dropout", default=0.1) or 0.1

        # Back-compat expansion if channels weren't saved
        if channels_per_layer is None:
            hidden_size = get_chk_int(checkpoint, "hidden_size", default=64)
            num_layers = get_chk_int(checkpoint, "num_layers", default=4)
            channels_per_layer = [hidden_size] * num_layers
        if dilation_schedule is None:
            dilation_schedule = [2 ** i for i in range(len(channels_per_layer))]

        model = SpeedEstimatorTCN(
            input_size=input_size,
            output_size=output_size,
            channels_per_layer=channels_per_layer,
            dilation_schedule=dilation_schedule,
            convolutions_per_block=convolutions_per_block,
            kernel_size=kernel_size,
            dropout=float(dropout),
            # you can add: head_pooling='last', causal=True if you also saved them in the checkpoint
        ).to(device)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Load weights
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing 'model_state_dict'")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, seq_len, step_size, output_size


def sliding_window_predict(
    model: torch.nn.Module,
    features: torch.Tensor,
    seq_len: int,
    step_size: int,
    device: torch.device,
    output_size: int,
) -> pd.DataFrame:
    """
    Applies sliding window inference and maps predictions back to timeline.
    Returns a DataFrame with shape (T, output_size), forward/back-filled.
    """
    feats = features.to(device)
    T = feats.shape[0]

    windows = []
    end_indices = []
    start = 0
    while start + seq_len <= T:
        windows.append(feats[start:start + seq_len, :].unsqueeze(0))  # (1, L, C)
        end_indices.append(start + seq_len - 1)
        start += max(1, step_size)

    # If not enough timesteps: return an all-NaN DataFrame
    if len(windows) == 0:
        return pd.DataFrame([[float("nan")] * output_size for _ in range(T)], columns=[f"y{i}" for i in range(output_size)])

    batch = torch.cat(windows, dim=0)  # (N, L, C)

    with torch.no_grad():
        preds = model(batch).detach().cpu()  # (N, output_size)

    # Convert to numpy (N, output_size)
    preds_np = preds.numpy()
    # Build full-length series, NaNs everywhere except window ends
    full = pd.DataFrame([[float("nan")] * output_size for _ in range(T)], columns=[f"y{i}" for i in range(output_size)])
    for idx, row in zip(end_indices, preds_np):
        for k in range(output_size):
            full.iat[idx, k] = float(row[k])

    # Forward/back-fill for readability; keep NaN if you prefer sparse
    full = full.ffill().bfill()

    return full


if __name__ == "__main__":
    # ========= User-editable controls =========
    model_type = "Transformer"          # "TCN" | "RNN" | "LSTM" | "GRU" | "Transformer"
    task = "lon"                # subdir for models/results; e.g., "lon", "lat", or "both"
    iteration_num = "1_norm"           # iteration number
    car_model = "i7"

    # Data and model locations
    validation_data_loc = f"../1_data/{car_model}/it_1/it_1_100_norm/3_validation/"
    base_model_dir = f"../2_trained_models/{model_type}/{car_model}/it_{iteration_num}/"
    model_folder_loc = os.path.join(base_model_dir, "state_models", task)
    csv_save_loc = os.path.join(base_model_dir, "results", task)
    os.makedirs(csv_save_loc, exist_ok=True)
    # =========================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gather model checkpoints
    if not os.path.isdir(model_folder_loc):
        raise RuntimeError(f"Model folder not found: {model_folder_loc}")
    pt_files = [f for f in os.listdir(model_folder_loc) if f.endswith(".pt")]
    pt_files.sort()

    # Gather validation CSV files
    if not os.path.isdir(validation_data_loc):
        raise RuntimeError(f"Validation data folder not found: {validation_data_loc}")
    validation_files = [f for f in os.listdir(validation_data_loc) if f.endswith(".csv")]
    validation_files.sort()

    if len(pt_files) == 0:
        raise RuntimeError(f"No .pt files found in {model_folder_loc}")
    if len(validation_files) == 0:
        raise RuntimeError(f"No validation .csv files found in {validation_data_loc}")

    for csv_name in validation_files:
        csv_path = os.path.join(validation_data_loc, csv_name)
        validation_df = pd.read_csv(csv_path)

        # Prepare result frame per validation file
        results_df = pd.DataFrame()
        # copy reference columns if present
        if "Time" in validation_df.columns:
            results_df["time"] = validation_df["Time"]
        if "veh_u" in validation_df.columns:
            results_df["veh_u"] = validation_df["veh_u"]
        if "veh_v" in validation_df.columns:
            results_df["veh_v"] = validation_df["veh_v"]

        # Prepare features once
        test_dataset = FullFileForTestings(csv_path)
        features, actual_speeds = test_dataset.get_full_data()  # features: (T, C)

        for pt in pt_files:
            checkpoint_path = os.path.join(model_folder_loc, pt)
            # Load checkpoint (weights_only may not exist in older torch)
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Build model from checkpoint
            model, seq_len, step_size, output_size = build_model_from_checkpoint(
                model_type=model_type,
                checkpoint=checkpoint,
                device=device,
                fallback_input_size=features.shape[1],
            )

            # Sliding-window inference
            pred_df = sliding_window_predict(
                model=model,
                features=features,
                seq_len=seq_len,
                step_size=step_size,
                device=device,
                output_size=output_size,
            )  # columns: y0, y1, ...

            # Attach to results with stable column names
            if output_size == 1:
                results_df[pt] = pred_df["y0"].values[: len(results_df)]
            else:
                for k in range(output_size):
                    results_df[f"{pt}[{k}]"] = pred_df[f"y{k}"].values[: len(results_df)]

        # Save results for this validation file
        out_path = os.path.join(csv_save_loc, csv_name)
        results_df.to_csv(out_path, index=False)