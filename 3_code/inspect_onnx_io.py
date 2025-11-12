#!/usr/bin/env python3
"""
Inspect an ONNX model's input/output shapes and print the correct MATLAB
InputDataFormats/OutputDataFormats to use with importONNXNetwork.

How to use:
1) Set model_path to the full path of ONE .onnx file, e.g.:
     model_path = r"C:\path\to\your_model.onnx"
   Leave glob_path empty ("") in that case.

2) OR set glob_path to a pattern that matches multiple models, e.g.:
     glob_path = r"..\2_trained_models\TCN\**\*.onnx"
   Leave model_path as "" in that case.

3) Run:
     python 3_code\inspect_onnx_io.py

It will print:
- Each model's real inputs/outputs with ranks and shapes
- A ready-to-copy MATLAB importONNXNetwork call with the correct formats
"""

import os
import glob
from typing import List, Optional

try:
    import onnx
    from onnx import shape_inference
except Exception as e:
    raise SystemExit(
        "This script requires the 'onnx' package.\n"
        "Install it with: pip install onnx\n"
        f"Import error: {e}"
    )

# -----------------------------------------------------------------------------
# EDIT JUST ONE OF THESE
# -----------------------------------------------------------------------------
# Option A: inspect a single ONNX file
model_path = r"C:\work\AI_training\2_trained_models\TCN\i7\it_1_norm\traced_models\for_matlab\lon\model_TCN_lon_128_conv2d.onnx"  # e.g., r"C:\path\to\your_model.onnx"

# Option B: inspect multiple ONNX files by glob
glob_path = r""   # e.g., r"..\2_trained_models\TCN\**\*.onnx"
# -----------------------------------------------------------------------------


def dim_to_str(dim) -> str:
    if dim.HasField("dim_value"):
        return str(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param  # symbolic dimension name
    return "?"


def shape_to_list_str(value_info) -> List[str]:
    if value_info is None or not value_info.type.HasField("tensor_type"):
        return ["?"]
    shp = value_info.type.tensor_type.shape
    return [dim_to_str(d) for d in shp.dim]


def list_real_inputs(graph) -> List[onnx.ValueInfoProto]:
    # Exclude initializers (weights) from inputs
    init_names = set(t.name for t in graph.initializer)
    return [inp for inp in graph.input if inp.name not in init_names]


def suggest_labels(rank: int) -> str:
    # For TCN sequence models we expect:
    #  - rank 3: [B, T, C] -> 'BTC'
    #  - rank 2: [B, T]    -> 'BT' (only if you exported a squeezed output)
    #  - rank 4 (rare for this exporter): treat as image-like 'SSCB' (H,W,C,B)
    if rank == 3:
        return "BTC"
    if rank == 2:
        return "BT"
    if rank == 4:
        return "SSCB"
    # Fallback: label every dim as batch 'B' to avoid length mismatch
    return "B" * max(rank, 1)


def inspect_one(path: str):
    print("=" * 88)
    print(f"Model: {path}")

    model = onnx.load(path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[warn] shape inference failed: {e}")

    g = model.graph
    real_inputs = list_real_inputs(g)
    outputs = list(g.output)

    if not real_inputs:
        print("[error] No data inputs detected (only initializers found).")
        return
    if not outputs:
        print("[error] No outputs found in the ONNX model.")
        return

    print("- Inputs:")
    for i, ii in enumerate(real_inputs, 1):
        shp = shape_to_list_str(ii)
        print(f"  {i}. name='{ii.name}' shape={shp} rank={len(shp)}")

    print("- Outputs:")
    for i, oo in enumerate(outputs, 1):
        shp = shape_to_list_str(oo)
        print(f"  {i}. name='{oo.name}' shape={shp} rank={len(shp)}")

    in_rank = len(shape_to_list_str(real_inputs[0]))
    out_rank = len(shape_to_list_str(outputs[0]))

    in_fmt = suggest_labels(in_rank)
    out_fmt = suggest_labels(out_rank)

    # MATLAB command suggestions
    print("- MATLAB import suggestions:")
    # 1) With explicit DataFormats
    # Note: Use OutputNetworkType='dlnetwork' for most flexible codegen/predict usage
    mpath = path.replace("\\", "\\\\")  # escape backslashes for copy-paste into MATLAB
    print(f"  net = importONNXNetwork('{mpath}', ...")
    print(f"      'ImportWeights', true, ...")
    print(f"      'OutputNetworkType', 'dlnetwork', ...")
    print(f"      'InputDataFormats',  {{'{in_fmt}'}}, ...")
    print(f"      'OutputDataFormats', {{'{out_fmt}'}} );")
    # 2) Without formats (let MATLAB infer)
    print("  % Or let MATLAB infer formats:")
    print(f"  net = importONNXNetwork('{mpath}', 'ImportWeights', true, 'OutputNetworkType', 'dlnetwork');")

    # Prediction example for rank-3 input [B,T,C]
    if in_rank == 3:
        print("  % Example predict (rank-3 input [B,T,C] labeled 'BTC'):")
        print("  B = 2; T = 128; C = 12;  % set C to your input feature count")
        print("  X = rand(B,T,C,'single');")
        print("  Y = predict(net, dlarray(X,'BTC'));  % expect [B,T,1] or [B,T]")
    elif in_rank == 2:
        print("  % Example predict (rank-2 input [B,T] labeled 'BT'):")
        print("  B = 2; T = 128;")
        print("  X = rand(B,T,'single');")
        print("  Y = predict(net, dlarray(X,'BT'));   % expect [B,T] or [B,?]")

    print()  # spacer


def main():
    paths: List[str] = []
    if model_path and os.path.isfile(model_path):
        paths.append(model_path)
    if glob_path:
        paths.extend(glob.glob(glob_path, recursive=True))

    # De-duplicate while preserving order
    seen = set()
    uniq_paths = []
    for p in paths:
        if p not in seen:
            uniq_paths.append(p)
            seen.add(p)

    if not uniq_paths:
        print("No ONNX files found.\n"
              "- Set 'model_path' to a single .onnx file, OR\n"
              "- Set 'glob_path' to a pattern like r\"..\\2_trained_models\\**\\*.onnx\"")
        return

    for p in uniq_paths:
        inspect_one(p)


if __name__ == "__main__":
    main()