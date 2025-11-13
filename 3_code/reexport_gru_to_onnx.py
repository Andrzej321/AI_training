#!/usr/bin/env python3
# Re-export a trained GRU checkpoint to ONNX with no state I/O (HasStateInputs/Outputs = false).
#
# How to use:
# 1) Set ckpt_path to your .pt/.pth (saved during training).
# 2) Set seq_len to the fixed sequence length T (or leave None to read ckpt['sequence_length']).
# 3) Optional: set out_path. If None, it saves next to the checkpoint.
# 4) Run: python export_stateless_onnx.py
#
# Notes:
# - Default export_mode="unrolled": replaces GRU with fixed-T stack of GRUCell ops.
#   This avoids ONNX GRU, shape/gather/squeeze artifacts, and state ports entirely.
# - If you prefer ONNX GRU operator, set export_mode="gru". We still do not pass h0 and only
#   consume h_n internally, so the network has no external state ports. However, MATLAB
#   dlnetwork initialize in older releases may still object to the GRU variant; use unrolled if unsure.
# - This script assumes your training checkpoint dict contains keys like:
#   'model_state_dict', 'input_size', 'hidden_size', 'num_layers', 'output_size', 'sequence_length'.
# - PyTorch 2.6+ default torch.load(weights_only=True) breaks older checkpoints; we force weights_only=False.

import os
import torch
import torch.nn as nn


# ========= USER CONFIG =========
ckpt_path = "../2_trained_models/GRU/trained_models/i7/it_4_norm/state_models/lon/model_GRU_lon_24.pt"
seq_len   = None   # e.g., 50. If None, tries ckpt['sequence_length']
out_path  = "../2_trained_models/GRU/trained_models/i7/it_4_norm/for_matlab/lon/model_GRU_lon_24.onnx"  # if None, derives from ckpt_path with suffix below
export_mode = "unrolled"  # "unrolled" (recommended) or "gru"
opset = 11                # MATLAB codegen safest with 11
# ==============================


class SpeedEstimatorGRUForExport(nn.Module):
    """
    GRU + FC that returns fc(h_n[-1]) to avoid time-slice ops at the output.
    This keeps the model stateless at the interface (no h0 passed, no state outputs exposed).
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Do NOT pass initial state; exporter will keep it internal → no state input port
        _, h_n = self.gru(x)   # h_n: [num_layers, B, H]
        last = h_n[-1]         # [B, H] (select last layer's h)
        return self.fc(last)   # [B, output_size]


class GRUStackUnrolled(nn.Module):
    """
    Multi-layer GRU rebuilt as a stack of GRUCell layers, unrolled for fixed T.
    Eliminates the ONNX GRU op and any state or time-slice artifacts.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, T: int):
        super().__init__()
        self.T = int(T)
        self.hidden_size = int(hidden_size)
        self.cells = nn.ModuleList(
            [nn.GRUCell(input_size if l == 0 else hidden_size, hidden_size) for l in range(num_layers)]
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] batch_first
        B, T_in, _ = x.shape
        if T_in != self.T:
            raise RuntimeError(f"Expected sequence length T={self.T}, got {T_in}")
        # zero init states (internal) → no state inputs/outputs at interface
        h = [x.new_zeros((B, self.hidden_size)) for _ in range(len(self.cells))]
        for t in range(self.T):
            inp = x[:, t, :]
            for l, cell in enumerate(self.cells):
                h[l] = cell(inp, h[l])
                inp = h[l]
        return self.fc(h[-1])  # [B, output_size]


def load_checkpoint(path: str):
    # PyTorch 2.6+: default weights_only=True is too strict; use False for your training ckpt
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        ckpt = {"model_state_dict": ckpt}
    if "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint at {path} lacks 'model_state_dict'.")
    return ckpt


def strip_dataparallel(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def derive_hparams(ckpt: dict, fallback_input_size=12, fallback_output_size=1):
    input_size = int(ckpt.get("input_size", fallback_input_size))
    hidden_size = int(ckpt.get("hidden_size", 64))
    num_layers = int(ckpt.get("num_layers", 2))
    output_size = int(ckpt.get("output_size", fallback_output_size))
    return input_size, hidden_size, num_layers, output_size


def build_model_gru(ckpt: dict):
    input_size, hidden_size, num_layers, output_size = derive_hparams(ckpt)
    model = SpeedEstimatorGRUForExport(input_size, hidden_size, num_layers, output_size)
    sd = strip_dataparallel(ckpt["model_state_dict"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] GRU load: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, input_size


def build_model_unrolled(ckpt: dict, T: int):
    input_size, hidden_size, num_layers, output_size = derive_hparams(ckpt)
    model = GRUStackUnrolled(input_size, hidden_size, num_layers, output_size, T=T)
    sd = strip_dataparallel(ckpt["model_state_dict"])

    # Copy FC weights if present
    if "fc.weight" in sd and "fc.bias" in sd:
        with torch.no_grad():
            model.fc.weight.copy_(sd["fc.weight"])
            model.fc.bias.copy_(sd["fc.bias"])

    # Map GRU weights to GRUCell layers
    for l, cell in enumerate(model.cells):
        w_ih = sd[f"gru.weight_ih_l{l}"]  # [3H, in]
        w_hh = sd[f"gru.weight_hh_l{l}"]  # [3H, H]
        b_ih = sd.get(f"gru.bias_ih_l{l}", torch.zeros(w_ih.size(0), dtype=w_ih.dtype))
        b_hh = sd.get(f"gru.bias_hh_l{l}", torch.zeros(w_hh.size(0), dtype=w_hh.dtype))
        with torch.no_grad():
            cell.weight_ih.copy_(w_ih)
            cell.weight_hh.copy_(w_hh)
            if cell.bias is not None:
                cell.bias_ih.copy_(b_ih)
                cell.bias_hh.copy_(b_hh)
    model.eval()
    return model, input_size


def export_onnx(model: nn.Module, output_path: str, T: int, C: int, opset: int = 11):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dummy = torch.randn(1, T, C, dtype=torch.float32)  # [B, T, C], batch_first

    # Only batch axis is dynamic; T and C are fixed for codegen
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"[ok] Saved ONNX: {output_path}")


def main():
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)
    T = seq_len if seq_len is not None else ckpt.get("sequence_length", None)
    if T is None:
        raise SystemExit("sequence_length missing. Set seq_len at the top of this script.")
    T = int(T)

    if export_mode.lower() == "unrolled":
        model, C = build_model_unrolled(ckpt, T)
        suffix = "_unrolled.onnx"
    elif export_mode.lower() == "gru":
        model, C = build_model_gru(ckpt)
        suffix = "_codegen.onnx"
    else:
        raise SystemExit('export_mode must be "unrolled" or "gru".')

    out = out_path or (os.path.splitext(ckpt_path)[0] + suffix)
    print(f"[info] Exporting ({export_mode}) with T={T}, C={C}, opset={opset}")
    with torch.no_grad():
        export_onnx(model, out, T, C, opset)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()