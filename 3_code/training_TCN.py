import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tcn import TCNRegressor


# Optional: if your project exposes dataloaders/datasets in classes.py, import them.
# Replace these with your actual names if different.
PROJECT_DATALOADERS_FN = None
try:
    from classes import build_dataloaders as PROJECT_DATALOADERS_FN  # type: ignore
except Exception:
    try:
        from classes import get_dataloaders as PROJECT_DATALOADERS_FN  # type: ignore
    except Exception:
        PROJECT_DATALOADERS_FN = None


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ArraySequenceDataset(Dataset):
    """
    Fallback dataset if the project-specific loaders are not available.
    Expects numpy arrays saved at paths for X (B, L, C) and y (B,) or (B, 1) or (B, L, out_dim).
    """
    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path)  # (B, L, C)
        self.y = np.load(y_path)  # (B,) or (B, 1) or (B, L, out_dim)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        assert self.X.shape[0] == self.y.shape[0], "X and y must have same batch size"

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return x, y


def build_fallback_dataloaders(hp: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    train_X = hp.get("train_X")
    train_y = hp.get("train_y")
    val_X = hp.get("val_X")
    val_y = hp.get("val_y")
    test_X = hp.get("test_X")
    test_y = hp.get("test_y")

    assert train_X and train_y, "Provide 'train_X' and 'train_y' npy paths in hyperparams when no project dataloaders are available."

    batch_size = int(hp.get("batch_size", 64))
    num_workers = int(hp.get("num_workers", 0))

    train_ds = ArraySequenceDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = None
    test_loader = None
    if val_X and val_y:
        val_loader = DataLoader(ArraySequenceDataset(val_X, val_y), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if test_X and test_y:
        test_loader = DataLoader(ArraySequenceDataset(test_X, test_y), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def select_hp_row(hp_csv: str, exp_id: Optional[str]) -> Dict[str, Any]:
    df = pd.read_csv(hp_csv)
    # Prefer explicit 'model' column with 'TCN'
    candidates = df[df.columns].copy()
    row = None
    if exp_id is not None:
        if "exp_id" in df.columns:
            m = df["exp_id"].astype(str) == str(exp_id)
            if m.any():
                row = df[m].iloc[0]
        if row is None and "id" in df.columns:
            m = df["id"].astype(str) == str(exp_id)
            if m.any():
                row = df[m].iloc[0]
    if row is None:
        # First row where model == 'TCN' if such column exists; else just first row.
        if "model" in df.columns:
            m = df["model"].astype(str).str.upper() == "TCN"
            row = df[m].iloc[0] if m.any() else df.iloc[0]
        else:
            row = df.iloc[0]
    hp = row.to_dict()
    return hp


def coerce_list(s: Any) -> List[int]:
    if isinstance(s, list):
        return [int(x) for x in s]
    if isinstance(s, str):
        # Expected like "64,64,64"
        parts = [p.strip() for p in s.replace("[", "").replace("]", "").split(",") if p.strip() != ""]
        return [int(p) for p in parts]
    return [int(s)]


def make_model(hp: Dict[str, Any]) -> nn.Module:
    input_dim = int(hp.get("input_dim", hp.get("input_size", 1)))
    out_dim = int(hp.get("out_dim", hp.get("output_dim", hp.get("output_size", 1))))
    kernel_size = int(hp.get("kernel_size", 3))
    dropout = float(hp.get("dropout", 0.0))
    return_sequence = bool(hp.get("return_sequence", False))
    use_global_pool = bool(hp.get("use_global_pool", False))
    channels = coerce_list(hp.get("channels", hp.get("hidden_channels", [64, 64, 64])))

    model = TCNRegressor(
        input_dim=input_dim,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
        out_dim=out_dim,
        return_sequence=return_sequence,
        use_global_pool=use_global_pool,
    )
    return model


def make_optim(model: nn.Module, hp: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(hp.get("lr", 1e-3))
    wd = float(hp.get("weight_decay", 0.0))
    opt_name = str(hp.get("optimizer", "adam")).lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        momentum = float(hp.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        y_true = y_true.view(y_true.size(0), -1)
        y_pred = y_pred.view(y_pred.size(0), -1)
        mse = torch.mean((y_true - y_pred) ** 2).item()
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        rmse = float(np.sqrt(mse))
        # Safe R2 for simple regression
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item() + 1e-8
        r2 = 1.0 - ss_res / ss_tot
        return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_true.append(y.detach().cpu())
            all_pred.append(y_pred.detach().cpu())
    avg_loss = total_loss / max(n, 1)
    y_true = torch.cat(all_true, dim=0) if all_true else torch.empty(0)
    y_pred = torch.cat(all_pred, dim=0) if all_pred else torch.empty(0)
    return avg_loss, metrics(y_true, y_pred) if y_true.numel() > 0 else {"mse": 0, "mae": 0, "rmse": 0, "r2": 0}


def train_one(hp: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    # Data
    if PROJECT_DATALOADERS_FN is not None:
        train_loader, val_loader, test_loader = PROJECT_DATALOADERS_FN(hp)  # type: ignore
    else:
        train_loader, val_loader, test_loader = build_fallback_dataloaders(hp)

    # Model / Optim / Loss
    model = make_model(hp).to(device)
    loss_name = str(hp.get("loss", "mse")).lower()
    loss_fn = nn.MSELoss() if loss_name in ("mse", "l2") else nn.L1Loss()
    optimizer = make_optim(model, hp)
    scheduler = None
    if str(hp.get("scheduler", "")).lower() == "steplr":
        step_size = int(hp.get("step_size", 20))
        gamma = float(hp.get("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training
    epochs = int(hp.get("epochs", 50))
    patience = int(hp.get("patience", 10))
    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(hp.get("grad_clip", 1.0)))
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        if scheduler is not None:
            scheduler.step()
        train_loss = running / max(n, 1)

        val_loss, val_metrics = (0.0, {}) if val_loader is None else evaluate(model, val_loader, device, loss_fn)

        if val_loader is not None:
            current = val_loss
            improved = current < best_val - float(hp.get("min_delta", 1e-6))
        else:
            current = train_loss
            improved = current < best_val - float(hp.get("min_delta", 1e-6))

        if improved:
            best_val = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}" + (f" | {val_metrics}" if val_loader is not None else ""))

        if patience_ctr >= patience:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval
    results: Dict[str, Any] = {"best_val_loss": best_val}
    if val_loader is not None:
        vloss, vmet = evaluate(model, val_loader, device, loss_fn)
        results.update({f"val_{k}": v for k, v in vmet.items()})
        results["val_loss"] = vloss
    if test_loader is not None:
        tloss, tmet = evaluate(model, test_loader, device, loss_fn)
        results.update({f"test_{k}": v for k, v in tmet.items()})
        results["test_loss"] = tloss
    return {"model": model, "metrics": results}


def save_run(model: nn.Module, hp: Dict[str, Any], results: Dict[str, Any], base_dir: str = "../2_trained_models/TCN") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = str(hp.get("run_name", f"tcn_{ts}"))
    out_dir = Path(base_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    model_path = out_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "hyperparams": hp}, model_path)

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save resolved hyperparams for reproducibility
    with open(out_dir / "hyperparams_resolved.json", "w") as f:
        json.dump(hp, f, indent=2)

    print(f"Saved to: {out_dir}")
    return str(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Train TCN model from hyperparameters.")
    parser.add_argument("--hyperparams", type=str, default="3_code/hyperparams.csv", help="Path to hyperparams CSV.")
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment id to select a row (matches exp_id or id).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="2_trained_models/TCN")
    # Optional overrides like --override channels=[64,64,64] lr=0.001
    parser.add_argument("--override", nargs="*", default=[], help="Override key=value pairs, e.g., channels=[64,64,64] lr=1e-3")
    args = parser.parse_args()

    set_seed(args.seed)
    hp = select_hp_row(args.hyperparams, args.exp_id)

    # Apply CLI overrides
    for kv in args.override:
        if "=" in kv:
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            # Try to parse lists and numbers
            if v.startswith("[") and v.endswith("]"):
                hp[k] = v
            else:
                try:
                    if "." in v:
                        hp[k] = float(v)
                    else:
                        hp[k] = int(v)
                except ValueError:
                    if v.lower() in ("true", "false"):
                        hp[k] = v.lower() == "true"
                    else:
                        hp[k] = v

    device = torch.device(args.device)
    out = train_one(hp, device)
    save_run(out["model"], hp, out["metrics"], base_dir=args.save_dir)


if __name__ == "__main__":
    main()