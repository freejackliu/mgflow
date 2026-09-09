#!/usr/bin/env python3
"""Train ensemble of PyTorch NN models for uncertainty quantification.

Trains M models per property with different random seeds, then saves them
as an ensemble that can be loaded by EnsemblePredictor.

Usage:
    python -m mgflow.ensemble_train  --n_models 20
    # or
    from mgflow.ensemble_train import train_ensemble_all
    train_ensemble_all(n_models=20)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..models.neural_net import ARCH_CONFIGS

_DATA_BASE = Path(__file__).resolve().parent.parent.parent.parent / "Matlab 程序文件夹" / "TgTxTl_E_H" / "TgTxTl_E_H"
_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "pytorch_models"

FEAT25_DROP_IDX = [4, 5, 7, 24]

REDUCTION = {
    "Tg": {
        "i_min": np.array([6, 14, 2, 2, 1, 2, 5, 4, 12, 8, 8, 6, 3, 2, 3, 3, 5, 1, 3, 2]),
        "k_min": 9,
    },
    "Tx": {
        "i_min": np.array([3, 1, 15, 4, 7, 2, 10, 5, 2, 3, 11, 10, 7, 6, 4, 2, 1, 4, 2, 2]),
        "k_min": 9,
    },
    "Tl": {
        "i_min": np.array([15, 10, 7, 5, 3, 11, 11, 4, 7, 2, 7, 3, 6, 5, 1, 1, 3, 1, 3, 2]),
        "k_min": 13,
    },
    "E": {
        "i_min": np.array([17, 2, 3, 6, 14, 1, 9, 10, 1, 11, 6, 1, 1, 2, 3, 3, 4, 2, 3, 1]),
        "k_min": 11,
    },
    "H": {
        "i_min": np.array([21, 1, 2, 17, 9, 2, 8, 9, 7, 4, 1, 1, 5, 3, 1, 3, 1, 3, 2, 1]),
        "k_min": 13,
    },
}


def _apply_reduction(imin_1based, k_min, original_n=21):
    cols = list(range(original_n))
    for k in range(k_min):
        idx = int(imin_1based[k]) - 1
        cols.pop(idx)
    return cols


KEEP_COLS = {}
for prop, info in REDUCTION.items():
    KEEP_COLS[prop] = _apply_reduction(info["i_min"], info["k_min"])


def _load_dataset(prop):
    """Load features and labels for a given property (same as train_pytorch.py)."""
    import openpyxl
    from ..features.compute import compute_all_features

    excel_config = {
        "Tg": ("Tg_comb.xlsx", 25),
        "Tx": ("Tx_comb.xlsx", 26),
        "Tl": ("Tl_comb.xlsx", 28),
        "E":  ("E_comb.xlsx", 27),
        "H":  ("H_comb.xlsx", 28),
    }
    fname, label_col = excel_config[prop]
    path = _DATA_BASE / fname
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    data_rows = list(ws.iter_rows(min_row=2, values_only=True))

    X_21_list, y_list = [], []
    for row in data_rows:
        name, comp_str, frac_str = row[0], row[2], row[3]
        label_val = row[label_col]
        if name is None or comp_str is None or frac_str is None or label_val is None:
            continue
        if isinstance(label_val, str) and label_val.startswith("="):
            continue
        syms = str(comp_str).strip().split()
        fracs = np.array([float(f) for f in str(frac_str).strip().split()], dtype=np.float64)
        fracs = fracs / fracs.sum()
        comp = {s: f for s, f in zip(syms, fracs)}
        feat_25 = compute_all_features(comp)
        feat_21 = feat_25[[i for i in range(25) if i not in FEAT25_DROP_IDX]]
        X_21_list.append(feat_21)
        y_list.append(float(label_val))

    X_21 = np.array(X_21_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    good = ~np.isnan(X_21).any(axis=1) & ~np.isnan(y)
    X_21 = X_21[good]
    y = y[good]
    keep = KEEP_COLS[prop]
    X = X_21[:, keep]
    return X, y, keep


def train_ensemble_property(prop, n_models=20, device="cpu", output_dir=None):
    """Train M models for one property with different random seeds.

    Returns
    -------
    models : list of nn.Module
    scalers : list of dict
    metrics : dict with 'rmse_list', 'mean_rmse', 'std_rmse'
    """
    from ..models.neural_net import PropertyNN, PropertyNN_Small

    X_full, y_full, keep_cols = _load_dataset(prop)
    n_features = X_full.shape[1]
    cfg = ARCH_CONFIGS[prop]
    n_total = len(X_full)

    models = []
    scalers = []
    rmse_list = []

    if output_dir is None:
        output_dir = _OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for m in range(n_models):
        seed = 42 + m * 7
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 80/20 train/val split (different each model due to seed)
        indices = np.random.permutation(n_total)
        split = int(0.8 * n_total)
        train_idx = indices[:split]
        val_idx = indices[split:]

        X_train, y_train = X_full[train_idx], y_full[train_idx]
        X_val, y_val = X_full[val_idx], y_full[val_idx]

        # Standardize
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std < 1e-12] = 1.0
        X_train_scaled = (X_train - X_mean) / X_std
        X_val_scaled = (X_val - X_mean) / X_std

        y_mean = y_train.mean()
        y_std = y_train.std()
        y_train_scaled = (y_train - y_mean) / y_std
        y_val_scaled = (y_val - y_mean) / y_std

        # DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_scaled, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val_scaled, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds))

        # Model
        model_class = cfg["class"]
        model = model_class(n_features, hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"])
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e-6)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(cfg["epochs"]):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break

        model.load_state_dict(best_state)
        model.eval()

        # Compute validation RMSE in original scale
        with torch.no_grad():
            X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            y_pred_scaled = model(X_val_t).cpu().numpy()
            y_pred = y_pred_scaled * y_std + y_mean
            y_val_orig = y_val
            rmse = float(np.sqrt(np.mean((y_pred - y_val_orig) ** 2)))
            rmse_list.append(rmse)

        models.append(model)
        scalers.append({
            "X_mean": X_mean, "X_std": X_std,
            "y_mean": float(y_mean), "y_std": float(y_std),
            "keep_cols_21": keep_cols,
        })

        if (m + 1) % 5 == 0 or m == 0 or m == n_models - 1:
            print(f"  [{prop}] Model {m+1:3d}/{n_models}: val RMSE = {rmse:.4f}")

    # Save ensemble
    ensemble_dir = output_dir / "ensembles"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Save all model weights
    for m, model in enumerate(models):
        model_path = ensemble_dir / f"{prop}_model_{m:03d}.pt"
        torch.save(model.state_dict(), model_path)

    # Save all scalers
    scaler_path = ensemble_dir / f"{prop}_scalers.npz"
    scaler_dict = {}
    for m, s in enumerate(scalers):
        scaler_dict[f"X_mean_{m}"] = s["X_mean"]
        scaler_dict[f"X_std_{m}"] = s["X_std"]
        scaler_dict[f"y_mean_{m}"] = np.array([s["y_mean"]])
        scaler_dict[f"y_std_{m}"] = np.array([s["y_std"]])
        if m == 0:
            scaler_dict["keep_cols_21"] = np.array(s["keep_cols_21"])
            scaler_dict["n_models"] = np.array([n_models])
    np.savez_compressed(scaler_path, **scaler_dict)

    # Save config
    config_path = ensemble_dir / f"{prop}_config.json"
    import json
    with open(config_path, "w") as f:
        json.dump({
            "property": prop,
            "n_models": n_models,
            "n_features": n_features,
            "n_training_samples": n_total,
            "hidden_dims": cfg["hidden_dims"],
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "epochs": cfg["epochs"],
            "model_class": cfg["class"].__name__,
            "rmse_list": [round(r, 4) for r in rmse_list],
            "mean_rmse": round(float(np.mean(rmse_list)), 4),
            "std_rmse": round(float(np.std(rmse_list)), 4),
        }, f, indent=2)

    metrics = {
        "rmse_list": rmse_list,
        "mean_rmse": float(np.mean(rmse_list)),
        "std_rmse": float(np.std(rmse_list)),
    }

    print(f"  [{prop}] Ensemble of {n_models} models: mean RMSE = {metrics['mean_rmse']:.4f} ± {metrics['std_rmse']:.4f}")
    return models, scalers, metrics


def train_ensemble_all(n_models=20, device=None, save=True):
    """Train ensemble models for all 5 properties."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training ensemble of {n_models} models per property")
    print()

    results = {}
    for prop in ["Tg", "Tx", "Tl", "E", "H"]:
        print(f"{'='*50}")
        print(f"Training ensemble for {prop}...")
        print(f"{'='*50}")
        models, scalers, metrics = train_ensemble_property(prop, n_models=n_models, device=device)
        results[prop] = {"models": models, "scalers": scalers, "metrics": metrics}
        print()

    print("All ensemble models trained!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ensemble of NN models for uncertainty quantification")
    parser.add_argument("--n_models", type=int, default=20, help="Number of models per property (default: 20)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    parser.add_argument("--props", type=str, nargs="+", default=["Tg", "Tx", "Tl", "E", "H"],
                        help="Properties to train")
    args = parser.parse_args()
    train_ensemble_all(n_models=args.n_models, device=args.device)
