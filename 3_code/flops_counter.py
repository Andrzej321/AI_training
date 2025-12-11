import os
import torch
from typing import Any, Dict, Tuple, List

from ptflops import get_model_complexity_info

# Reuse helpers from your validation script
from validation_general import build_model_from_checkpoint, get_chk_int, get_chk_list


# ============== USER-EDITABLE SETTINGS ==============
# Path to a single .pt checkpoint file
CHECKPOINT_PATH = "../2_trained_models/best_models_lon/ai_models/TCN/state_models/model_TCN_lon_129.pt"

# Model type used during training: "RNN" | "LSTM" | "GRU" | "Transformer" | "TCN"
MODEL_TYPE = "TCN"

# Batch size for FLOP counting (per batch for RNN/Transformer).
# For TCN we simulate N windows; FLOPs scale linearly with this.
BATCH_SIZE = 1

# If not None, override the input_size used when creating the model and dummy input.
# If None, try to read input_size from checkpoint ("input_size", "fixed_input_size", "feature_count").
FORCED_INPUT_SIZE = None  # e.g. 32, or None to auto-detect

# Backend for ptflops; "aten" is recommended for non-CNN models and TCNs.
PTFLOPS_BACKEND = "aten"
# ====================================================


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {checkpoint_path}")
    return checkpoint


def infer_input_size_from_checkpoint(checkpoint: Dict[str, Any], fallback: int = 1) -> int:
    """
    Tries to infer input_size from checkpoint, falling back to provided default.
    Mirrors logic in validation_general.py.
    """
    input_size = get_chk_int(checkpoint, "input_size", "fixed_input_size", "feature_count", default=fallback)
    if input_size is None:
        input_size = fallback
    return input_size


def infer_tcn_channels_and_kernel(
    checkpoint: Dict[str, Any],
    default_hidden_size: int = 64,
    default_num_layers: int = 4,
    default_kernel_size: int = 3,
) -> Tuple[List[int], int, int]:
    """
    Extracts TCN-specific metadata from checkpoint: channels_per_layer, convolutions_per_block, kernel_size.
    Follows the same fallbacks as build_model_from_checkpoint.
    """
    channels_per_layer = get_chk_list(checkpoint, "channels_per_layer")
    dilation_schedule = get_chk_list(checkpoint, "dilation_schedule")  # we don't actually need this for FLOPs
    convolutions_per_block = get_chk_int(checkpoint, "convolutions_per_block", "convolution_per_block", default=2)
    kernel_size = get_chk_int(checkpoint, "kernel_size", default=default_kernel_size)

    # Back-compat expansion if channels weren't saved
    if channels_per_layer is None:
        hidden_size = get_chk_int(checkpoint, "hidden_size", default=default_hidden_size)
        num_layers = get_chk_int(checkpoint, "num_layers", default=default_num_layers)
        channels_per_layer = [hidden_size] * num_layers

    return channels_per_layer, convolutions_per_block, kernel_size


def build_model_and_meta(
    checkpoint_path: str,
    model_type: str,
    device: torch.device,
    fallback_input_size: int,
) -> Tuple[torch.nn.Module, int, int, int]:
    """
    Uses build_model_from_checkpoint to reconstruct the model
    and obtain (model, seq_len, step_size, output_size).
    """
    checkpoint = load_checkpoint(checkpoint_path, device)
    model, seq_len, step_size, output_size = build_model_from_checkpoint(
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
        fallback_input_size=fallback_input_size,
    )
    return model, seq_len, step_size, output_size


def build_dummy_input(
    model_type: str,
    seq_len: int,
    batch_size: int,
    input_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Constructs a dummy input tensor with the shape expected by your models.

    - RNN/LSTM/GRU/Transformer: (seq_len, batch, input_size)
    - TCN: (N_windows, seq_len, input_size) as in sliding_window_predict,
           where N_windows = batch_size.
    """
    mt = model_type.strip().lower()

    if mt in ("rnn", "lstm", "gru", "transformer"):
        # (seq_len, batch, input_size)
        input_shape = (seq_len, batch_size, input_size)
        x = torch.randn(*input_shape, device=device)
    elif mt == "tcn":
        # sliding_window_predict builds windows of shape (N, L, C) and passes to model
        # Here we simulate N = batch_size windows:
        # (batch_size, seq_len, input_size)
        input_shape = (batch_size, seq_len, input_size)
        x = torch.randn(*input_shape, device=device)
    else:
        raise ValueError(f"Unsupported model_type for dummy input: {model_type}")

    return x, input_shape


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_checkpoint = load_checkpoint(CHECKPOINT_PATH, device)

    if FORCED_INPUT_SIZE is not None:
        input_size = int(FORCED_INPUT_SIZE)
    else:
        input_size = infer_input_size_from_checkpoint(raw_checkpoint, fallback=1)

    # Build model from checkpoint
    model, seq_len, step_size, output_size = build_model_and_meta(
        checkpoint_path=CHECKPOINT_PATH,
        model_type=MODEL_TYPE,
        device=device,
        fallback_input_size=input_size,
    )
    model.eval()

    print(f"Loaded model from: {CHECKPOINT_PATH}")
    print(f"  Model type:       {MODEL_TYPE}")
    print(f"  sequence_length:  {seq_len}")
    print(f"  input_size:       {input_size}")
    print(f"  output_size:      {output_size}")
    print(f"  step_size (chk):  {step_size}")
    print(f"  batch_size:       {BATCH_SIZE}")

    mt = MODEL_TYPE.strip().lower()
    if mt == "tcn":
        # TCN-specific meta (for your own reference / sanity check)
        channels_per_layer, convolutions_per_block, kernel_size = infer_tcn_channels_and_kernel(raw_checkpoint)
        print("  TCN channels_per_layer:", channels_per_layer)
        print("  TCN convolutions_per_block:", convolutions_per_block)
        print("  TCN kernel_size:", kernel_size)

    # Build dummy input and input_constructor for ptflops
    dummy_x, input_res = build_dummy_input(
        model_type=MODEL_TYPE,
        seq_len=seq_len,
        batch_size=BATCH_SIZE,
        input_size=input_size,
        device=device,
    )

    def input_constructor(_input_res):
        # ptflops will call this to obtain input tensors for model.forward
        mt_inner = MODEL_TYPE.strip().lower()
        if mt_inner in ("rnn", "lstm", "gru", "transformer"):
            # forward(x) where x is a *tensor*, not a tuple
            return dummy_x
        elif mt_inner == "tcn":
            # forward(x) where x is (batch_size, seq_len, input_size) tensor
            return dummy_x
        else:
            raise ValueError(f"Unsupported model_type: {MODEL_TYPE}")

    # Compute MACs/params with ptflops
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model,
            input_res=input_res,
            as_strings=True,
            input_constructor=input_constructor,
            print_per_layer_stat=True,
            verbose=True,
            backend=PTFLOPS_BACKEND,
        )

    print("=" * 60)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Model type: {MODEL_TYPE}")
    print(f"MACs:   {macs}")
    print(f"Params: {params}")
    print("Note: If you treat 1 MAC = 2 FLOPs, then FLOPs ≈ 2 × MACs.")


if __name__ == "__main__":
    main()