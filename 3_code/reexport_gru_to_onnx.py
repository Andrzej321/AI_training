#!/usr/bin/env python3
# Usage:
# 1) Set ckpt_path (and optionally seq_len and out_path) below.
# 2) Run: python 3_code/reexport_single.py
#
# Notes:
# - seq_len: If None, the script reads "sequence_length" from the checkpoint.
# - The exported ONNX fixes the sequence length (T) and keeps only the batch dim dynamic.
# - The GRU forward uses h_n[-1] instead of slicing the time dimension to avoid Gather/Squeeze ops in ONNX.

import os
import torch
import torch.nn as nn

# ==== USER CONFIG =================================================================
# Absolute or relative path to your .pt checkpoint (saved by 3_code/training_GRU.py)
ckpt_path = "../2_trained_models/GRU/trained_models/i7/it_4_norm/state_models/lon/model_GRU_lon_24.pt"

# Fixed sequence length T; set to None to read from checkpoint["sequence_length"]
seq_len = None  # e.g., 50

# Output ONNX path; set to None to auto-generate next to the checkpoint with suffix "_codegen.onnx"
out_path = "../2_trained_models/GRU/trained_models/i7/it_4_norm/for_matlab/lon/model_GRU_lon_24_codegen.onnx"

# ONNX opset (MATLAB codegen safest with 11)
opset = 11
# ===================================================================================


class SpeedEstimatorGRUForExport(nn.Module):
    """
    Architecture-compatible with your trained GRU+FC, but forward returns fc(h_n[-1]).
    This avoids ONNX time-slice ops (Gather/Squeeze) and keeps the GRU stateless at the interface.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0  # has no effect in eval; keeps export stable
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No initial state inputs; uses zeros internally (stateless interface)
        _, h_n = self.gru(x)   # h_n: [num_layers, B, H]
        last = h_n[-1]         # [B, H]
        out = self.fc(last)    # [B, output_size]
        return out


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only = False)
    if not isinstance(ckpt, dict):
        # If only a raw state_dict was saved
        ckpt = {"model_state_dict": ckpt}
    if "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint at {path} does not contain 'model_state_dict'.")
    return ckpt


def build_model_from_ckpt(ckpt: dict, fallback_input_size=12, fallback_output_size=1):
    input_size = int(ckpt.get("input_size", fallback_input_size))
    hidden_size = int(ckpt.get("hidden_size", 64))
    num_layers = int(ckpt.get("num_layers", 2))
    output_size = int(ckpt.get("output_size", fallback_output_size))

    model = SpeedEstimatorGRUForExport(input_size, hidden_size, num_layers, output_size)
    state_dict = ckpt["model_state_dict"]

    # Remove potential DataParallel 'module.' prefixes
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning while loading state_dict: missing={missing}, unexpected={unexpected}")

    model.eval()
    return model, input_size


def export_onnx(model: nn.Module, output_path: str, seq_len: int, input_size: int, opset: int = 11):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dummy = torch.randn(1, seq_len, input_size, dtype=torch.float32)  # B=1, fixed T, fixed C

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={  # only batch is dynamic
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )
    print(f"Saved ONNX: {output_path}")


def main():
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)

    # Determine sequence length
    T = seq_len if seq_len is not None else ckpt.get("sequence_length", None)
    if T is None:
        raise SystemExit("sequence_length missing in checkpoint. Please set seq_len at the top of this script.")

    model, input_size = build_model_from_ckpt(ckpt)

    # Derive output path if not provided
    out = out_path
    if out is None:
        base, _ = os.path.splitext(ckpt_path)
        out = base + "_codegen.onnx"

    print(f"Exporting with parameters: T={T}, C={input_size}, opset={opset}")
    export_onnx(model, out, int(T), input_size, opset)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()