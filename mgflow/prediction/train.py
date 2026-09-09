#!/usr/bin/env python3
"""Train PyTorch NN models for all 5 properties, replacing MATLAB GPR.

Usage:
    python -m mgflow.train_pytorch
    # or
    from mgflow.train_pytorch import train_all
    train_all()
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import openpyxl

from ..models.neural_net import ARCH_CONFIGS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_BASE = Path(__file__).resolve().parent.parent.parent.parent / "Matlab 程序文件夹" / "TgTxTl_E_H" / "TgTxTl_E_H"
_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "pytorch_models"

# ---------------------------------------------------------------------------
# 25-feat (Python) → 21-feat (Excel) column mapping
# Python  compute_all_features() returns 25 features.
# The Excel datasets only have 21 features (4 of the 25 were not used).
# Indices of 25-feat array to drop to get the 21 Excel features:
FEAT25_DROP_IDX = [4, 5, 7, 24]  # dn_dc5(sd_dc), dn_dc6(sd_dn), ye_elastic_energy, Eig_Hessian

# ---------------------------------------------------------------------------
# Reduction indices (from MATLAB's local_training.m)
# Each entry: (i_min_1based, k_min_1based)
# i_min: which column to remove at each step (MATLAB 1-indexed, within CURRENT matrix)
# k_min: number of removal steps that gives lowest RMSE
# After removing k_min columns from the 21-feat space, we keep the rest.
# ---------------------------------------------------------------------------
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
    """Determine which of the original 21 columns survive after k_min removals.

    Returns a list of 0-indexed column indices (into the 21-feat space) to KEEP.
    """
    cols = list(range(original_n))
    for k in range(k_min):
        idx = int(imin_1based[k]) - 1  # MATLAB → 0-indexed (current matrix)
        cols.pop(idx)
    return cols  # indices into 21-feat space


# Pre-compute keep-columns for each property
KEEP_COLS = {}
for prop, info in REDUCTION.items():
    KEEP_COLS[prop] = _apply_reduction(info["i_min"], info["k_min"])


def _excel_col(coords):
    """Read numeric data from an Excel cell range.

    coords: str like 'E2:Y844' or 'E2:Y328'
    """
    import openpyxl.utils
    return coords


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset(prop):
    """Load features and labels for a given property.

    Features are computed via compute_all_features() (Python), not from Excel,
    so that training and prediction use the same feature computation.

    Returns
    -------
    X : ndarray (n_samples, n_features_selected)
    y : ndarray (n_samples,)
    scaler : dict with 'mean' and 'std' (computed on X)
    feature_names : list of str
    """
    # Map property → Excel file, label column (0-indexed)
    # We read composition from these files, then compute features ourselves
    excel_config = {
        "Tg": ("Tg_comb.xlsx", 25),   # Tg label
        "Tx": ("Tx_comb.xlsx", 26),   # Tx label
        "Tl": ("Tl_comb.xlsx", 28),   # Tl label
        "E":  ("E_comb.xlsx", 27),    # E (GPa) label
        "H":  ("H_comb.xlsx", 28),    # Hv (GPa) label
    }

    fname, label_col = excel_config[prop]
    path = _DATA_BASE / fname

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active

    # Read composition strings & labels from Excel
    data_rows = list(ws.iter_rows(min_row=2, values_only=True))
    n_samples = len(data_rows)

    from ..features.compute import compute_all_features

    # Indices of 25-feat array to drop to get 21 Excel-matching features
    FEAT25_DROP_IDX = [4, 5, 7, 24]

    X_21_list = []
    y_list = []

    for row in data_rows:
        name = row[0]
        comp_str = row[2]
        frac_str = row[3]
        label_val = row[label_col]

        if name is None or comp_str is None or frac_str is None or label_val is None:
            continue
        if isinstance(label_val, str) and label_val.startswith("="):
            continue

        # Parse composition
        syms = str(comp_str).strip().split()
        fracs = np.array([float(f) for f in str(frac_str).strip().split()], dtype=np.float64)
        fracs = fracs / fracs.sum()
        comp = {s: f for s, f in zip(syms, fracs)}

        # Compute features with Python (same as prediction)
        feat_25 = compute_all_features(comp)
        feat_21 = feat_25[[i for i in range(25) if i not in FEAT25_DROP_IDX]]

        X_21_list.append(feat_21)
        y_list.append(float(label_val))

    X_21 = np.array(X_21_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    # Filter out any NaN rows
    good = ~np.isnan(X_21).any(axis=1) & ~np.isnan(y)
    if not good.all():
        print(f"  [{prop}] Dropped {np.sum(~good)} rows with NaN features")
        X_21 = X_21[good]
        y = y[good]

    # Apply reduction: keep only selected columns
    keep = KEEP_COLS[prop]
    X = X_21[:, keep]

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0  # avoid division by zero
    X_scaled = (X - mean) / std

    # Standardize labels too
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std

    scaler = {
        "X_mean": mean, "X_std": std,
        "y_mean": y_mean, "y_std": y_std,
        "keep_cols_21": keep,  # indices into 21-feat space
    }

    feature_names = [f"feat_{i}" for i in keep]

    return X_scaled, y_scaled, scaler, feature_names


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_property(prop, device="cpu", seed=42):
    """Train a PyTorch NN for one property.

    Returns
    -------
    model : nn.Module (trained, in eval mode)
    scaler : dict
    history : dict with 'train_loss', 'val_loss', 'best_epoch'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y, scaler, _ = _load_dataset(prop)
    n_features = X.shape[1]
    cfg = ARCH_CONFIGS[prop]

    # Train/val split (80/20)
    n = len(X)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
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
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  [{prop}] Epoch {epoch+1:3d}/{cfg['epochs']}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if patience_counter >= cfg["patience"]:
            print(f"  [{prop}] Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_state)
    model.eval()

    # Compute RMSE on validation set in original scale
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_pred_scaled = model(X_val_t).cpu().numpy()
        y_pred = y_pred_scaled * scaler["y_std"] + scaler["y_mean"]
        y_val_orig = y_val * scaler["y_std"] + scaler["y_mean"]
        rmse = np.sqrt(np.mean((y_pred - y_val_orig) ** 2))

    print(f"  [{prop}] Done. Best val RMSE (original scale): {rmse:.4f}  "
          f"(n_features={n_features}, n_train={len(train_idx)}, n_val={len(val_idx)})")

    history["best_epoch"] = np.argmin(history["val_loss"])
    history["best_val_rmse"] = float(rmse)

    return model, scaler, history


def train_all(device=None, seed=42, save=True):
    """Train models for all 5 properties."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for prop in ["Tg", "Tx", "Tl", "E", "H"]:
        print(f"{'='*50}")
        print(f"Training {prop}...")
        print(f"{'='*50}")
        model, scaler, history = train_property(prop, device=device, seed=seed)
        results[prop] = {"model": model, "scaler": scaler, "history": history}

        if save:
            # Save model weights
            model_path = _OUTPUT_DIR / f"model_{prop}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  Saved model to {model_path}")

            # Save scaler
            scaler_path = _OUTPUT_DIR / f"scaler_{prop}.npz"
            np.savez_compressed(
                scaler_path,
                X_mean=scaler["X_mean"],
                X_std=scaler["X_std"],
                y_mean=scaler["y_mean"],
                y_std=scaler["y_std"],
                keep_cols_21=np.array(scaler["keep_cols_21"]),
            )
            print(f"  Saved scaler to {scaler_path}")

        # Full dataset training for final model (retrain on ALL data)
        print(f"  Retraining {prop} on full dataset...")
        model_full, scaler_full, _ = train_property_full(prop, device=device)
        if save:
            model_path_full = _OUTPUT_DIR / f"model_{prop}_full.pt"
            torch.save(model_full.state_dict(), model_path_full)
            scaler_path_full = _OUTPUT_DIR / f"scaler_{prop}_full.npz"
            np.savez_compressed(
                scaler_path_full,
                X_mean=scaler_full["X_mean"],
                X_std=scaler_full["X_std"],
                y_mean=scaler_full["y_mean"],
                y_std=scaler_full["y_std"],
                keep_cols_21=np.array(scaler_full["keep_cols_21"]),
            )
            print(f"  Saved full model to {model_path_full}")

        print()

    print("All models trained!")
    return results


def train_property_full(prop, device="cpu", seed=42):
    """Train on the FULL dataset (no validation split)."""
    torch.manual_seed(seed + 10)
    np.random.seed(seed + 10)

    X, y, scaler, _ = _load_dataset(prop)
    n_features = X.shape[1]
    cfg = ARCH_CONFIGS[prop]

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=min(32, len(X)), shuffle=True)

    model_class = cfg["class"]
    model = model_class(n_features, hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    max_epochs = cfg["epochs"] // 2  # fewer epochs on full data

    for epoch in range(max_epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"] // 2:
            break

    model.load_state_dict(best_state)
    model.eval()

    # Compute training RMSE in original scale
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_pred_scaled = model(X_t).cpu().numpy()
        y_pred = y_pred_scaled * scaler["y_std"] + scaler["y_mean"]
        y_orig = y * scaler["y_std"] + scaler["y_mean"]
        rmse = np.sqrt(np.mean((y_pred - y_orig) ** 2))

    print(f"  [{prop}] Full-dataset RMSE: {rmse:.4f}")

    return model, scaler, {"train_rmse": float(rmse)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_all()
