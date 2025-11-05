#!/usr/bin/env python3
import os
import sys
import json
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ USER SETTING: point to your .pt checkpoint ============
folder_path = "../2_trained_models/TCN/i7/it_1_norm/state_models/lon/"

model_name = "model_TCN_lon_129.pt"

folder_to_save_path = "../2_trained_models/TCN/i7/it_1_norm/for_matlab/lon/"

path_model = folder_path + model_name

path_to_save = folder_to_save_path + model_name

# ====================================================================


# Make sure Python can import from 3_code/
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
THREE_CODE = os.path.join(REPO_ROOT, "3_code")
if THREE_CODE not in sys.path:
    sys.path.insert(0, THREE_CODE)

from classes import SpeedEstimatorTCN  # type: ignore


# --------------------- Conv1d -> Conv2d helpers ----------------------
class SamePadConv1dAs2d(nn.Module):
    """
    Replacement for Conv1d:
    - Applies explicit 'same' or 'causal' padding on the time axis
    - Uses Conv2d with kernel (K,1), stride (S,1), dilation (D,1)
    Inputs:  (N, C, L)
    Outputs: (N, F, L_out)  (equals L for stride=1)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, bias: bool = True,
                 padding_mode: str = "same"):
        super().__init__()
        assert padding_mode in ("same", "causal")
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.padding_mode = padding_mode

        self.conv2d = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride, 1),
            dilation=(self.dilation, 1),
            padding=(0, 0),  # manual padding
            bias=bias,
        )

    @torch.no_grad()
    def load_from_conv1d(self, conv1d: nn.Conv1d):
        # Conv1d [F, C, K] -> Conv2d [F, C, K, 1]
        self.conv2d.weight.copy_(conv1d.weight.data.unsqueeze(-1))
        if conv1d.bias is not None and self.conv2d.bias is not None:
            self.conv2d.bias.copy_(conv1d.bias.data)

    def _left_right_pad(self) -> Tuple[int, int]:
        eff = self.dilation * (self.kernel_size - 1)
        if self.padding_mode == "same":
            left = eff // 2
            right = eff - left
        else:
            left, right = eff, 0
        return left, right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        left, right = self._left_right_pad()
        x = x.unsqueeze(-1)                # (N, C, L, 1)
        x = F.pad(x, (0, 0, left, right))  # pad along L
        y = self.conv2d(x)                 # (N, F, L_out, 1)
        return y.squeeze(-1)               # (N, F, L_out)


def _is_chomp_like(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    return any(k in name for k in ("chomp", "crop", "trim", "slice"))


def strip_chomp_layers(model: nn.Module) -> int:
    replaced = 0
    for child_name, child in list(model.named_children()):
        if _is_chomp_like(child):
            setattr(model, child_name, nn.Identity())
            replaced += 1
        else:
            replaced += strip_chomp_layers(child)
    return replaced


def convert_conv1d_to_2d(model: nn.Module, padding_mode: str = "same") -> int:
    replaced = 0
    for child_name, child in list(model.named_children()):
        if isinstance(child, nn.Conv1d):
            new = SamePadConv1dAs2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size[0],
                stride=child.stride[0],
                dilation=child.dilation[0],
                bias=(child.bias is not None),
                padding_mode=padding_mode,
            )
            new.load_from_conv1d(child)
            setattr(model, child_name, new)
            replaced += 1
        else:
            replaced += convert_conv1d_to_2d(child, padding_mode=padding_mode)
    return replaced


# --------- Replace "last timestep" with pooling to avoid Gather ----------
class LastTimestepAsPool2d(nn.Module):
    """
    Returns the exact last timestep using masked AvgPool2d over time.
    Produces only Mul + AvgPool in ONNX (no Gather).
    """
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = int(seq_len)
        mask = torch.zeros(1, 1, self.seq_len, 1, dtype=torch.float32)
        mask[:, :, -1, :] = self.seq_len
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L) or (N, C, L, 1)
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # (N, C, L, 1)
        y = x * self.mask
        y = F.avg_pool2d(y, kernel_size=(self.seq_len, 1), stride=(self.seq_len, 1))
        return y.squeeze(-1).squeeze(-1)  # (N, C)


class ExportWrapper(nn.Module):
    """
    Uses model.tcn and model.head (as in your training code) and performs
    'last timestep' via pooling, avoiding ONNX Gather/Slice.
    Falls back to base.forward if structure differs.
    """
    def __init__(self, base: nn.Module, seq_len: int, reduce: str = "last"):
        super().__init__()
        self.base = base
        self.seq_len = int(seq_len)
        self.reduce = reduce.lower()

    def forward(self, x_b_l_c: torch.Tensor) -> torch.Tensor:
        # Preferred path: use tcn + head explicitly
        if hasattr(self.base, "tcn") and hasattr(self.base, "head"):
            feats = self.base.tcn(x_b_l_c)  # expect [B, L, C]
            if feats.dim() != 3:
                # Unexpected; fallback
                return self.base(x_b_l_c)
            if self.reduce == "mean":
                pooled = feats.mean(dim=1)  # [B, C]
            else:
                # "last" via pooling
                feats_ncl = feats.transpose(1, 2)  # [B, C, L]
                pooled = LastTimestepAsPool2d(self.seq_len)(feats_ncl)  # [B, C]
            out = self.base.head(pooled)  # [B, out_dim]
            return out
        # Fallback if structure is different
        return self.base(x_b_l_c)


def main():
    if not os.path.isfile(path_model):
        raise FileNotFoundError(f"Checkpoint not found: {path_model}")

    ckpt = torch.load(path_model, map_location="cpu")
    # Support formats: {"model_state_dict": ...} or plain state_dict
    state = ckpt.get("model_state_dict", ckpt)

    # Pull required hyperparameters from checkpoint (saved by your trainer)
    input_size = int(ckpt.get("input_size"))
    seq_len = int(ckpt.get("sequence_length"))
    kernel_size = int(ckpt.get("kernel_size"))
    channels_per_layer = ckpt.get("channels_per_layer")
    num_residual_blocks = int(ckpt.get("num_residual_blocks"))
    convolutions_per_block = int(ckpt.get("convolutions_per_block"))
    dilation_schedule = ckpt.get("dilation_schedule")
    dropout = float(ckpt.get("dropout", 0.0))
    use_weight_norm = bool(ckpt.get("use_weight_norm", True))
    activation = str(ckpt.get("activation", "relu"))
    norm_in_block = str(ckpt.get("norm_in_block", "none"))
    head_pooling = str(ckpt.get("head_pooling", "last"))
    causal = bool(ckpt.get("causal", True))
    output_clamp_min = ckpt.get("output_clamp_min", None)

    # Instantiate model exactly like in training_TCN.py
    model = SpeedEstimatorTCN(
        input_size=input_size,
        output_size=1,
        channels_per_layer=list(channels_per_layer),
        num_residual_blocks=num_residual_blocks,
        convolutions_per_block=convolutions_per_block,
        kernel_size=kernel_size,
        dilation_schedule=list(dilation_schedule),
        dropout=dropout,
        use_weight_norm=use_weight_norm,
        activation=activation,
        norm_in_block=norm_in_block,
        head_pooling=head_pooling,  # we'll override behavior in export wrapper
        causal=causal,
        output_clamp_min=output_clamp_min,
    )
    model.load_state_dict(state, strict=False)

    # Convert to MATLAB-friendly internals
    pad_mode = "causal" if causal else "same"
    chomp_removed = strip_chomp_layers(model)
    conv_replaced = convert_conv1d_to_2d(model, padding_mode=pad_mode)
    print(f"[Info] Stripped {chomp_removed} chomp/crop layers; replaced {conv_replaced} Conv1d -> Conv2d")

    # Wrap to avoid ONNX Gather for "last" (default behavior in your trainer)
    export_model = ExportWrapper(model, seq_len=seq_len, reduce="last")
    export_model.eval()

    # Prepare paths
    base, _ = os.path.splitext(path_to_save)
    export_path = base + "_matlab.onnx"

    # Export ONNX with fixed shapes (best for MATLAB codegen)
    dummy = torch.randn(1, seq_len, input_size, dtype=torch.float32)  # [B, L, C] as per your code
    torch.onnx.export(
        export_model, dummy, export_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # fixed L
        do_constant_folding=True,
    )
    print(f"[OK] Exported MATLAB-friendly ONNX -> {export_path}")


if __name__ == "__main__":
    main()