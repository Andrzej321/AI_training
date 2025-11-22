import torch
import os
import sys

# Set this variable to your checkpoint path:
model_path = "../2_trained_models/Simple RNN/i7/it_1_norm/state_models/lon/model_RNN_lon_155.pt"

def infer_model_type(state_dict):
    # Look at parameter name prefixes
    for prefix, name in [
        ("rnn.", "rnn"),
        ("gru.", "gru"),
        ("lstm.", "lstm"),
        ("tcn.", "tcn"),
        ("encoder.", "transformer"),
        ("input_proj.", "transformer"),
    ]:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            return name
    # Fallback: look for fc/head only (could be simple MLP but in your case likely RNN)
    if any("fc." in k for k in state_dict.keys()):
        return "rnn?"  # ambiguous fallback
    if any("head." in k for k in state_dict.keys()):
        return "tcn/transformer?"
    return "unknown"

def infer_rnn_params(state_dict):
    # Works for plain RNN / GRU / LSTM (same key pattern: weight_ih_lX)
    weight_ih_keys = [k for k in state_dict if "weight_ih_l" in k]
    if not weight_ih_keys:
        return {}
    # Layer 0 to get hidden & input size
    first = [k for k in weight_ih_keys if k.startswith(weight_ih_keys[0].split("l")[0] + "l0") or k.endswith("l0")]
    # Safer: just pick the one with l0
    first_key = None
    for k in weight_ih_keys:
        if "l0" in k:
            first_key = k
            break
    if first_key is None:
        first_key = weight_ih_keys[0]
    w = state_dict[first_key]
    hidden_size = w.shape[0]
    input_size = w.shape[1]
    # Count number of layers
    num_layers = sum(1 for k in weight_ih_keys if "weight_ih_l" in k)
    # Output size: fc or head
    output_size = None
    if "fc.weight" in state_dict:
        output_size = state_dict["fc.weight"].shape[0]
    elif "head.weight" in state_dict:
        output_size = state_dict["head.weight"].shape[0]
    return {
        "input_size_inferred": input_size,
        "hidden_size_inferred": hidden_size,
        "num_layers_inferred": num_layers,
        "output_size_inferred": output_size,
    }

def infer_tcn_params(state_dict):
    # For TCN you can’t easily recover channels list cleanly here; just output head size
    info = {}
    if "head.weight" in state_dict:
        info["output_size_inferred"] = state_dict["head.weight"].shape[0]
        info["tcn_last_channel_inferred"] = state_dict["head.weight"].shape[1]
    return info

def infer_transformer_params(state_dict):
    info = {}
    if "input_proj.weight" in state_dict:
        info["input_size_inferred"] = state_dict["input_proj.weight"].shape[1]
        info["d_model_inferred"] = state_dict["input_proj.weight"].shape[0]
    if "head.weight" in state_dict:
        info["output_size_inferred"] = state_dict["head.weight"].shape[0]
    # Number of encoder layers: count submodules like 'encoder.layers.X.'
    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith("encoder.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))
    if layer_indices:
        info["num_layers_inferred"] = len(layer_indices)
    return info

def main():
    if not os.path.isfile(model_path):
        print(f"ERROR: File not found: {model_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # Your checkpoints have a dict with 'model_state_dict' + metadata keys
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    else:
        state_dict = ckpt
        metadata = {}

    # 1. Model type
    saved_type = str(metadata.get("model_type", "UNKNOWN")).lower()
    inferred_type = infer_model_type(state_dict)

    print(f"Checkpoint: {model_path}")
    print("==== Model Type ====")
    print(f"Saved (metadata): {saved_type}")
    print(f"Inferred (weights): {inferred_type}")

    # 2. Metadata parameters (main ones)
    print("\n==== Metadata Parameters (if present) ====")
    keys_of_interest = [
        "input_size", "hidden_size", "num_layers", "output_size",
        "dropout_rate", "sequence_length", "learning_rate", "batch_size", "epochs", "seed", "best_val_loss"
    ]
    found_any = False
    for k in keys_of_interest:
        if k in metadata:
            print(f"{k}: {metadata[k]}")
            found_any = True
    if not found_any:
        print("No standard metadata keys found.")

    # 3. Inferred parameters (fallbacks)
    print("\n==== Inferred From State Dict ====")
    inferred = {}
    if inferred_type in ("rnn", "gru", "lstm", "rnn?"):
        inferred.update(infer_rnn_params(state_dict))
    elif inferred_type.startswith("tcn"):
        inferred.update(infer_tcn_params(state_dict))
    elif "transformer" in inferred_type:
        inferred.update(infer_transformer_params(state_dict))
    else:
        # Try generic RNN inference anyway
        inferred.update(infer_rnn_params(state_dict))

    if inferred:
        for k, v in inferred.items():
            print(f"{k}: {v}")
    else:
        print("Could not infer structural parameters.")

    # 4. Parameter count
    total_params = sum(v.numel() for v in state_dict.values())
    print("\n==== Parameter Count ====")
    print(f"Total parameters: {total_params:,}")

    print("\nDone.")

if __name__ == "__main__":
    main()