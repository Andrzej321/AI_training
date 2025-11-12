#!/usr/bin/env python3
# Re-export trained TCN (Conv1d) models as functionally equivalent Conv2d models for MATLAB codegen.
# - Removes weight norm
# - Replaces Conv1d (dilated causal) with Conv2d(kernel=(1,k), dilation=(1,d)) and explicit left-only padding
# - Returns full sequence (sequence-to-sequence) to avoid in-graph Gather/Slice in ONNX
#
# Expected checkpoint content (as saved by training_TCN.py):
#  {
#    "model_state_dict": ...,
#    "sequence_length": int,
#    "input_size": int,
#    "channels_per_layer": List[int],
#    "dilation_schedule": List[int],
#    "num_residual_blocks": int,
#    "convolutions_per_block": int,
#    "kernel_size": int,
#    "dropout": float,
#    "use_weight_norm": bool,
#    "activation": str,
#    "norm_in_block": str,       # "none" | "batch" | "layer"
#    "head_pooling": str,        # "last" | "global_avg"
#    "causal": bool,
#    "output_clamp_min": Optional[float],
#    ...
#  }

import os
import glob
import argparse
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your original 1D TCN model
from classes import SpeedEstimatorTCN


def make_activation(name: str):
    name = (name or "relu").lower()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


def make_norm2d(norm: str, num_channels: int):
    norm = (norm or "none").lower()
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "layer":
        # GroupNorm(1, C) approximates LayerNorm across channels and works for any spatial dims
        return nn.GroupNorm(1, num_channels)
    return nn.Identity()


class CausalDilatedConv2d(nn.Module):
    """
    2D causal dilated convolution over width (time) dimension only.
    Input:  [B, C, 1, L]
    Output: [B, C_out, 1, L]
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding=(0, 0),  # we do explicit left-only pad
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.pad order for 4D (N,C,H,W) is (W_left, W_right, H_left, H_right)
        x = F.pad(x, (self.left_pad, 0, 0, 0))
        return self.conv(x)


class TemporalBlock2D(nn.Module):
    """
    Residual TCN block (2D): N causal Conv2d layers with same dilation, optional norm/activation/dropout, plus residual.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        convolutions_per_block: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_in_block: str = "none",
        causal: bool = True,  # kept for parity; we enforce causal design
    ):
        super().__init__()
        act = make_activation(activation)
        norm_ctor = lambda c: make_norm2d(norm_in_block, c)

        layers: List[nn.Module] = []
        current_in = in_channels
        for _ in range(max(1, convolutions_per_block)):
            conv = CausalDilatedConv2d(current_in, out_channels, kernel_size, dilation)
            layers.extend([conv, norm_ctor(out_channels), act, nn.Dropout(dropout)])
            current_in = out_channels

        self.net = nn.Sequential(*layers)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        self.activation = make_activation(activation)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class TemporalConvNet2D(nn.Module):
    """
    Stack of TemporalBlock2D with user-defined channels and dilation schedule.
    Accepts [B, L, C] and returns [B, L, C_out].
    """
    def __init__(
        self,
        in_channels: int,
        channels_per_layer: List[int],
        kernel_size: int,
        dilation_schedule: List[int],
        convolutions_per_block: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_in_block: str = "none",
        causal: bool = True,
    ):
        super().__init__()
        assert len(channels_per_layer) >= 1
        assert len(dilation_schedule) >= 1

        blocks: List[nn.Module] = []
        c_in = in_channels
        for i, c_out in enumerate(channels_per_layer):
            d = dilation_schedule[i] if i < len(dilation_schedule) else dilation_schedule[-1]
            blocks.append(
                TemporalBlock2D(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    dilation=d,
                    convolutions_per_block=convolutions_per_block,
                    dropout=dropout,
                    activation=activation,
                    norm_in_block=norm_in_block,
                    causal=causal,
                )
            )
            c_in = c_out

        self.network = nn.Sequential(*blocks)
        self.out_channels = channels_per_layer[-1]

    def forward(self, x_b_l_c: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B, C, 1, L]
        x = x_b_l_c.transpose(1, 2).unsqueeze(2)
        y = self.network(x)         # [B, C_out, 1, L]
        y = y.squeeze(2).transpose(1, 2)  # [B, L, C_out]
        return y


class SpeedEstimatorTCN2D(nn.Module):
    """
    Conv2d-based TCN wrapper for regression.
    If return_sequence=True, outputs [B, L, out_dim] (recommended for ONNX export to avoid Gather).
    If return_sequence=False, applies 'last' or 'global_avg' pooling then outputs [B, out_dim].
    """
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        channels_per_layer: Optional[List[int]] = None,
        num_residual_blocks: Optional[int] = None,
        convolutions_per_block: int = 2,
        kernel_size: int = 3,
        dilation_schedule: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_in_block: str = "none",
        causal: bool = True,
        head_pooling: str = "last",   # 'last' | 'global_avg' (used when return_sequence=False)
        output_clamp_min: Optional[float] = None,
        return_sequence: bool = True,  # default True for codegen-friendly ONNX
    ):
        super().__init__()
        # Derive channels and dilations if needed
        if channels_per_layer is None:
            nl = num_residual_blocks if num_residual_blocks is not None else 4
            channels_per_layer = [64] * nl
        if dilation_schedule is None:
            dilation_schedule = [2 ** i for i in range(len(channels_per_layer))]

        self.tcn = TemporalConvNet2D(
            in_channels=input_size,
            channels_per_layer=channels_per_layer,
            kernel_size=kernel_size,
            dilation_schedule=dilation_schedule,
            convolutions_per_block=convolutions_per_block,
            dropout=dropout,
            activation=activation,
            norm_in_block=norm_in_block,
            causal=causal,
        )
        self.return_sequence = return_sequence
        self.head_pooling = (head_pooling or "last").lower()
        self.output_clamp_min = output_clamp_min
        self.head = nn.Linear(self.tcn.out_channels, output_size)

    def forward(self, x_b_l_c: torch.Tensor) -> torch.Tensor:
        feats = self.tcn(x_b_l_c)  # [B, L, C_tcn]
        if self.return_sequence:
            out = self.head(feats)  # [B, L, out_dim]
        else:
            if self.head_pooling == "global_avg":
                pooled = feats.mean(dim=1)     # [B, C_tcn]
            else:
                pooled = feats[:, -1, :]       # [B, C_tcn]
            out = self.head(pooled)            # [B, out_dim]
        if self.output_clamp_min is not None:
            out = torch.clamp(out, min=self.output_clamp_min)
        return out


def remove_weight_norm_inplace(module: nn.Module):
    """Remove weight norm hooks from all Conv1d layers if present."""
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            try:
                nn.utils.remove_weight_norm(m)
            except (ValueError, AttributeError):
                pass


def collect_convs_1d(block_1d: nn.Module) -> List[nn.Conv1d]:
    return [m for m in block_1d.net.modules() if isinstance(m, nn.Conv1d)]


def collect_convs_2d(block_2d: nn.Module) -> List[nn.Conv2d]:
    return [m for m in block_2d.net.modules() if isinstance(m, nn.Conv2d)]


def collect_bn_1d(block_1d: nn.Module) -> List[nn.BatchNorm1d]:
    return [m for m in block_1d.net.modules() if isinstance(m, nn.BatchNorm1d)]


def collect_bn_2d(block_2d: nn.Module) -> List[nn.BatchNorm2d]:
    return [m for m in block_2d.net.modules() if isinstance(m, nn.BatchNorm2d)]


def transfer_block_weights_1d_to_2d(block_1d: nn.Module, block_2d: nn.Module):
    # Conv weights
    convs1 = collect_convs_1d(block_1d)
    convs2 = collect_convs_2d(block_2d)
    assert len(convs1) == len(convs2), "Conv count mismatch between 1D and 2D blocks"

    for c1, c2 in zip(convs1, convs2):
        with torch.no_grad():
            # [out, in, k] -> [out, in, 1, k]
            c2.weight.copy_(c1.weight.unsqueeze(2))
            if c1.bias is not None and c2.bias is not None:
                c2.bias.copy_(c1.bias)

    # Downsample if present
    if getattr(block_1d, "downsample", None) is not None and getattr(block_2d, "downsample", None) is not None:
        d1: nn.Conv1d = block_1d.downsample
        d2: nn.Conv2d = block_2d.downsample
        with torch.no_grad():
            d2.weight.copy_(d1.weight.unsqueeze(2))  # [out,in,1] -> [out,in,1,1]
            if d1.bias is not None and d2.bias is not None:
                d2.bias.copy_(d1.bias)

    # BatchNorm if used
    bns1 = collect_bn_1d(block_1d)
    bns2 = collect_bn_2d(block_2d)
    if len(bns1) or len(bns2):
        assert len(bns1) == len(bns2), "BatchNorm count mismatch between 1D and 2D blocks"
        for b1, b2 in zip(bns1, bns2):
            with torch.no_grad():
                if b1.affine and b2.affine:
                    b2.weight.copy_(b1.weight)
                    b2.bias.copy_(b1.bias)
                b2.running_mean.copy_(b1.running_mean)
                b2.running_var.copy_(b1.running_var)
                if hasattr(b1, "num_batches_tracked") and hasattr(b2, "num_batches_tracked"):
                    b2.num_batches_tracked.copy_(b1.num_batches_tracked)


def build_1d_model_from_ckpt(ckpt: dict) -> SpeedEstimatorTCN:
    model = SpeedEstimatorTCN(
        input_size=int(ckpt["input_size"]),
        output_size=1,
        channels_per_layer=list(ckpt["channels_per_layer"]),
        num_residual_blocks=int(ckpt["num_residual_blocks"]),
        convolutions_per_block=int(ckpt["convolutions_per_block"]),
        kernel_size=int(ckpt["kernel_size"]),
        dilation_schedule=list(ckpt["dilation_schedule"]),
        dropout=float(ckpt["dropout"]),
        use_weight_norm=bool(ckpt["use_weight_norm"]),
        activation=str(ckpt["activation"]),
        norm_in_block=str(ckpt["norm_in_block"]),
        head_pooling=str(ckpt["head_pooling"]),
        causal=bool(ckpt["causal"]),
        output_clamp_min=ckpt.get("output_clamp_min", None),
    )
    return model


def build_2d_model_matching_ckpt(ckpt: dict, return_sequence: bool = True) -> SpeedEstimatorTCN2D:
    model2d = SpeedEstimatorTCN2D(
        input_size=int(ckpt["input_size"]),
        output_size=1,
        channels_per_layer=list(ckpt["channels_per_layer"]),
        num_residual_blocks=int(ckpt["num_residual_blocks"]),
        convolutions_per_block=int(ckpt["convolutions_per_block"]),
        kernel_size=int(ckpt["kernel_size"]),
        dilation_schedule=list(ckpt["dilation_schedule"]),
        dropout=float(ckpt["dropout"]),
        activation=str(ckpt["activation"]),
        norm_in_block=str(ckpt["norm_in_block"]),
        causal=bool(ckpt["causal"]),
        head_pooling=str(ckpt["head_pooling"]),
        output_clamp_min=ckpt.get("output_clamp_min", None),
        return_sequence=return_sequence,
    )
    return model2d


def transfer_weights_model_1d_to_2d(model1d: SpeedEstimatorTCN, model2d: SpeedEstimatorTCN2D):
    # Transfer TCN blocks
    blocks1 = [b for b in model1d.tcn.network if isinstance(b, nn.Module)]
    blocks2 = [b for b in model2d.tcn.network if isinstance(b, nn.Module)]
    assert len(blocks1) == len(blocks2), "Number of residual blocks mismatch"
    for b1, b2 in zip(blocks1, blocks2):
        transfer_block_weights_1d_to_2d(b1, b2)

    # Transfer head (Linear) directly
    with torch.no_grad():
        model2d.head.weight.copy_(model1d.head.weight)
        if model1d.head.bias is not None and model2d.head.bias is not None:
            model2d.head.bias.copy_(model1d.head.bias)


def export_conv2d_onnx_from_checkpoint(
    ckpt_path: str,
    onnx_out_path: str,
    return_sequence: bool = True,
    opset: int = 13,
    verify_numerics: bool = True,
    atol: float = 1e-5,
) -> Tuple[str, Optional[float]]:
    """
    Load a 1D TCN checkpoint, convert to 2D TCN, transfer weights, and export ONNX.

    Returns:
        (onnx_out_path, max_abs_diff)  where max_abs_diff is None if verification disabled.
    """
    device = torch.device("cpu")
    state = torch.load(ckpt_path, map_location=device)
    ckpt = state

    # Rebuild and load 1D model
    model1d = build_1d_model_from_ckpt(ckpt).to(device)
    model1d.load_state_dict(ckpt["model_state_dict"], strict=True)
    remove_weight_norm_inplace(model1d)
    model1d.eval()

    # Build 2D model and transfer weights
    model2d = build_2d_model_matching_ckpt(ckpt, return_sequence=return_sequence).to(device)
    transfer_weights_model_1d_to_2d(model1d, model2d)
    model2d.eval()

    # Optional numerical check
    max_abs_diff = None
    if verify_numerics:
        B = 2
        L = int(ckpt["sequence_length"])
        C = int(ckpt["input_size"])
        x = torch.randn(B, L, C, device=device)
        with torch.no_grad():
            y1 = model1d.tcn(x)              # [B, L, C_tcn]
            if return_sequence:
                y1_out = model1d.head(y1)    # [B, L, out]
            else:
                if str(ckpt["head_pooling"]).lower() == "global_avg":
                    pooled = y1.mean(dim=1)
                else:
                    pooled = y1[:, -1, :]
                y1_out = model1d.head(pooled)  # [B, out]

            y2_out = model2d(x)
            max_abs_diff = (y1_out - y2_out).abs().max().item()
            print(f"[verify] max abs diff between 1D and 2D outputs: {max_abs_diff:.6e}")

    # Export ONNX (sequence-to-sequence by default)
    L = int(ckpt["sequence_length"])
    C = int(ckpt["input_size"])
    example_input = torch.randn(1, L, C, device=device)
    input_names = ["input"]
    output_names = ["output_seq" if return_sequence else "output"]
    dynamic_axes = {
        "input": {0: "batch_size", 1: "seq_len"},
        output_names[0]: {0: "batch_size", 1: "seq_len"} if return_sequence else {0: "batch_size"},
    }

    os.makedirs(os.path.dirname(onnx_out_path), exist_ok=True)
    torch.onnx.export(
        model2d,
        example_input,
        onnx_out_path,
        export_params=True,
        do_constant_folding=True,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"[export] Saved Conv2D-ONNX to: {onnx_out_path}")
    return onnx_out_path, max_abs_diff


def main():
    parser = argparse.ArgumentParser(description="Re-export TCN checkpoints to Conv2D ONNX for MATLAB codegen.")
    parser.add_argument("--ckpt_glob", type=str, required=False,
                        default="../2_trained_models/TCN/i7/it_1_norm/state_models/lon/model_TCN_lon_*.pt",
                        help="Glob for checkpoint files to export.")
    parser.add_argument("--out_dir", type=str, required=False,
                        default="../2_trained_models/TCN/i7/it_1_norm/traced_models/for_matlab/lon",
                        help="Output directory for ONNX files.")
    parser.add_argument("--return_sequence", type=int, default=1,
                        help="1: export sequence-to-sequence (recommended), 0: sequence-to-one.")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    parser.add_argument("--verify", type=int, default=1, help="1 to compare 1D vs 2D outputs.")
    args = parser.parse_args()

    ckpt_paths = sorted(glob.glob(args.ckpt_glob))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found for pattern: {args.ckpt_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    for ckpt_path in ckpt_paths:
        base = os.path.splitext(os.path.basename(ckpt_path))[0]
        onnx_name = f"{base}_conv2d.onnx" if args.return_sequence else f"{base}_conv2d_seq2one.onnx"
        onnx_out_path = os.path.join(args.out_dir, onnx_name)
        print(f"Converting: {ckpt_path} -> {onnx_out_path}")
        _, diff = export_conv2d_onnx_from_checkpoint(
            ckpt_path,
            onnx_out_path,
            return_sequence=bool(args.return_sequence),
            opset=int(args.opset),
            verify_numerics=bool(args.verify),
        )
        if diff is not None:
            print(f"Done. Max abs diff: {diff:.6e}")
        else:
            print("Done.")

if __name__ == "__main__":
    main()