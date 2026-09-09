#!/usr/bin/env python3
"""Residual Resistivity (R) training and prediction for Ta-Nb-Ni-Co-Hf system.

Self-contained script: uses mgflow's feature computation pipeline but manages
its own dataset and models independently.

Usage:
    # Train ensemble model (from resistivity_results.json)
    python train_predict_resistivity.py --train --n_models 30

    # Predict compositions
    python train_predict_resistivity.py --predict --comp "Nb50Ta25Ni15Co10"

    # Batch predict from JSON
    python train_predict_resistivity.py --predict --input comps.json --output results.json

    # Full pipeline: train + grid search + visualize
    python train_predict_resistivity.py --all
"""

import sys, os, re, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Add parent to path for mgflow imports ──
_HERE = Path(__file__).resolve().parent
_MGFLOW = _HERE.parent
if str(_MGFLOW) not in sys.path:
    sys.path.insert(0, str(_MGFLOW))

from mgflow.features.compute import compute_all_features, FEATURE_NAMES_25
from mgflow.data.loader import composition_vector

# ── Paths ──
DATA_DIR = _HERE / "data"
MODELS_DIR = _HERE / "models"
PRED_DIR = _HERE / "predictions"
FIG_DIR = _HERE / "figures"
RAW_DATA = DATA_DIR / "resistivity_results.json"
DATASET_NPZ = DATA_DIR / "dataset_R.npz"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PRED_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ── Constants ──
FEAT25_DROP_IDX = [4, 5, 7, 24]   # 25 → 21
FULL_KEEP_COLS = list(range(21))   # No further reduction for resistivity
ELEMENTS = ["Ta", "Nb", "Ni", "Co", "Hf"]

# ── Model ──
class PropertyNN(nn.Module):
    def __init__(self, n_input, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = n_input
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════

def parse_composition_string(comp_str):
    """Parse 'La55Cu20Al20Ag5' -> {'La': 0.55, 'Cu': 0.20, ...}"""
    matches = re.findall(r'([A-Z][a-z]?)(\d+\.?\d*)', comp_str)
    comp = {sym: float(v)/100.0 for sym, v in matches}
    total = sum(comp.values())
    return {k: v/total for k, v in comp.items()}


def build_dataset_from_json(json_path=RAW_DATA):
    """Read resistivity_results.json, compute features, save NPZ dataset."""
    with open(json_path) as f:
        raw = json.load(f)

    names, labels = [], []
    X_25_list = []
    failed = []

    for comp_str, val in raw.items():
        try:
            comp = parse_composition_string(comp_str)
            feat = compute_all_features(comp)
            X_25_list.append(feat)
            labels.append(val)
            names.append(comp_str)
        except Exception as e:
            failed.append((comp_str, str(e)))

    X_25 = np.array(X_25_list)
    y = np.array(labels)
    names = np.array(names)

    # 25 → 21 → full keep
    X_21 = X_25[:, [i for i in range(25) if i not in FEAT25_DROP_IDX]]
    X = X_21[:, FULL_KEEP_COLS]

    # Pad elem_nos/fracs for consistency with other datasets
    from mgflow.data.loader import composition_vector
    MAX_ELEMS = 8
    elem_nos_list, fracs_list = [], []
    for name in names:
        comp = parse_composition_string(name)
        elem, frac = composition_vector(comp)
        if len(elem) < MAX_ELEMS:
            elem = np.pad(elem, (0, MAX_ELEMS - len(elem)), constant_values=0)
            frac = np.pad(frac, (0, MAX_ELEMS - len(frac)), constant_values=0.0)
        elem_nos_list.append(elem)
        fracs_list.append(frac)

    np.savez_compressed(
        DATASET_NPZ,
        names=names,
        elem_nos=np.array(elem_nos_list, dtype=np.int16),
        fracs=np.array(fracs_list, dtype=np.float64),
        labels=y,
        X_21=X_21,
        X=X,
    )

    print(f"Dataset built: {len(names)} samples → {DATASET_NPZ}")
    return X, y, names


def load_dataset():
    """Load pre-built NPZ dataset."""
    data = np.load(DATASET_NPZ, allow_pickle=True)
    return data["X"], data["labels"], data["names"]


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

def train_ensemble(X, y, n_models=30, device=None):
    """Train M models, save ensemble to models/ directory."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_total, n_features = X.shape
    models, scalers, rmse_list = [], [], []

    cfg = {"hidden_dims": (128, 64), "dropout": 0.2, "lr": 1e-3,
           "weight_decay": 1e-5, "epochs": 500, "patience": 50}

    for m in range(n_models):
        seed = 42 + m * 7
        torch.manual_seed(seed)
        np.random.seed(seed)

        idx = np.random.permutation(n_total)
        split = int(0.8 * n_total)
        train_idx, val_idx = idx[:split], idx[split:]

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx], y[val_idx]

        X_mean = X_tr.mean(axis=0)
        X_std = X_tr.std(axis=0); X_std[X_std < 1e-12] = 1.0
        y_mean = y_tr.mean(); y_std = y_tr.std()

        X_tr_s = (X_tr - X_mean) / X_std
        X_vl_s = (X_vl - X_mean) / X_std
        y_tr_s = (y_tr - y_mean) / y_std
        y_vl_s = (y_vl - y_mean) / y_std

        train_ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32),
                                 torch.tensor(y_tr_s, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_vl_s, dtype=torch.float32),
                               torch.tensor(y_vl_s, dtype=torch.float32))
        train_loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=len(val_ds))

        model = PropertyNN(n_features, **{k: cfg[k] for k in ("hidden_dims", "dropout")})
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20, min_lr=1e-6)
        criterion = nn.MSELoss()

        best_loss, best_state, patience = float("inf"), None, 0
        for _ in range(cfg["epochs"]):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            vloss = np.mean([criterion(model(xb.to(device)), yb.to(device)).item()
                             for xb, yb in val_loader])
            sched.step(vloss)

            if vloss < best_loss:
                best_loss = vloss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    break

        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            y_pred = model(torch.tensor(X_vl_s, dtype=torch.float32).to(device)).cpu().numpy()
            y_pred = y_pred * y_std + y_mean
            rmse = float(np.sqrt(np.mean((y_pred - y_vl) ** 2)))
            rmse_list.append(rmse)

        models.append(model)
        scalers.append({"X_mean": X_mean, "X_std": X_std,
                        "y_mean": float(y_mean), "y_std": float(y_std),
                        "keep_cols_21": FULL_KEEP_COLS})

        if (m + 1) % 5 == 0:
            print(f"  Model {m+1:3d}/{n_models}: val RMSE = {rmse:.4f}")

    # Save
    for m, model in enumerate(models):
        torch.save(model.state_dict(), MODELS_DIR / f"R_model_{m:03d}.pt")

    scaler_dict = {"n_models": np.array([n_models]),
                   "keep_cols_21": np.array(FULL_KEEP_COLS)}
    for m, s in enumerate(scalers):
        scaler_dict[f"X_mean_{m}"] = s["X_mean"]
        scaler_dict[f"X_std_{m}"] = s["X_std"]
        scaler_dict[f"y_mean_{m}"] = np.array([s["y_mean"]])
        scaler_dict[f"y_std_{m}"] = np.array([s["y_std"]])
    np.savez_compressed(MODELS_DIR / "R_scalers.npz", **scaler_dict)

    with open(MODELS_DIR / "R_config.json", "w") as f:
        json.dump({"property": "R", "n_models": n_models,
                   "n_features": n_features, "n_training_samples": n_total,
                   "hidden_dims": list(cfg["hidden_dims"]),
                   "dropout": cfg["dropout"], "lr": cfg["lr"],
                   "weight_decay": cfg["weight_decay"],
                   "epochs": cfg["epochs"],
                   "model_class": "PropertyNN",
                   "mean_rmse": round(float(np.mean(rmse_list)), 4),
                   "std_rmse": round(float(np.std(rmse_list)), 4),
                   "rmse_list": [round(r, 4) for r in rmse_list]}, f, indent=2)

    print(f"  Ensemble saved: mean RMSE = {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    return models, scalers


# ══════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════

def load_ensemble(device=None):
    """Load trained ensemble from models/ directory."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(MODELS_DIR / "R_config.json") as f:
        config = json.load(f)

    scaler_data = np.load(MODELS_DIR / "R_scalers.npz", allow_pickle=True)
    n_models = config["n_models"]
    n_features = config["n_features"]
    hidden_dims = tuple(config["hidden_dims"])

    scalers = []
    for m in range(n_models):
        scalers.append({
            "X_mean": scaler_data[f"X_mean_{m}"],
            "X_std": scaler_data[f"X_std_{m}"],
            "y_mean": float(scaler_data[f"y_mean_{m}"][0]),
            "y_std": float(scaler_data[f"y_std_{m}"][0]),
            "keep_cols_21": scaler_data["keep_cols_21"],
        })

    models = []
    for m in range(n_models):
        model = PropertyNN(n_features, hidden_dims=hidden_dims, dropout=0.0)
        model.load_state_dict(torch.load(MODELS_DIR / f"R_model_{m:03d}.pt",
                                          map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        models.append(model)

    print(f"Loaded {n_models}-model ensemble ({n_features} features)")
    return models, scalers


def predict_one(composition, models, scalers, device=None):
    """Predict R with uncertainty for a single composition."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    feat_25 = compute_all_features(composition)
    feat_21 = feat_25[[i for i in range(25) if i not in FEAT25_DROP_IDX]]
    keep = scalers[0]["keep_cols_21"]
    feat_reduced = feat_21[keep]

    preds = []
    for model, scaler in zip(models, scalers):
        x_scaled = (feat_reduced - scaler["X_mean"]) / scaler["X_std"]
        x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            y_scaled = model(x_t).item()
        y_val = y_scaled * scaler["y_std"] + scaler["y_mean"]
        preds.append(y_val)

    return float(np.mean(preds)), float(np.std(preds, ddof=1)), preds


def predict_ta_nb_ni_co_hf(models, scalers, grid_step=0.1, n_random=500):
    """Generate and predict systematic Ta-Nb-Ni-Co-Hf compositions."""
    results = []

    # Grid: 3-element (step 0.1, each ≥ 0.1)
    for step in [0.1, 0.2]:
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    sub = [ELEMENTS[i], ELEMENTS[j], ELEMENTS[k]]
                    for a in np.arange(step, 1.0+1e-9, step):
                        for b in np.arange(step, 1.0-a+1e-9, step):
                            c = round(1.0 - a - b, 1)
                            if c >= step and abs(a+b+c-1.0) < 1e-6:
                                results.append(dict(zip(sub, [a, b, c])))

    # Grid: 4-element (step 0.2)
    for i in range(5):
        for j in range(i+1, 5):
            for k in range(j+1, 5):
                for l in range(k+1, 5):
                    sub = [ELEMENTS[i], ELEMENTS[j], ELEMENTS[k], ELEMENTS[l]]
                    for a in np.arange(0.1, 1.0+1e-9, 0.2):
                        for b in np.arange(0.1, 1.0-a+1e-9, 0.2):
                            for c in np.arange(0.1, 1.0-a-b+1e-9, 0.2):
                                d = round(1.0-a-b-c, 1)
                                if d >= 0.1 and abs(a+b+c+d-1.0) < 1e-6:
                                    results.append(dict(zip(sub, [a, b, c, d])))

    # Random 5-element
    rng = np.random.RandomState(42)
    for _ in range(n_random):
        raw = rng.dirichlet(np.ones(5) * 2)
        comp = {el: round(v, 3) for el, v in zip(ELEMENTS, raw)}
        if all(v >= 0.03 for v in comp.values()):
            results.append(comp)

    print(f"Predicting {len(results)} compositions...")
    output = []
    for comp in results:
        try:
            mean_r, std_r, _ = predict_one(comp, models, scalers)
            comp_str = "".join(f"{el}{v*100:.1f}" for el, v in
                              sorted(comp.items(), key=lambda x: -x[1]) if v > 0)
            output.append({
                "composition": {k: round(v, 4) for k, v in comp.items()},
                "composition_str": comp_str,
                "R (µΩ·cm)": round(mean_r, 2),
                "R_std": round(std_r, 4),
            })
        except Exception:
            pass

    return output


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Resistivity predictor")
    parser.add_argument("--train", action="store_true", help="Build dataset and train")
    parser.add_argument("--n_models", type=int, default=30, help="Ensemble size")
    parser.add_argument("--predict", action="store_true", help="Run prediction")
    parser.add_argument("--comp", type=str, help="Single composition, e.g. Nb50Ta25Ni15Co10")
    parser.add_argument("--input", type=str, help="JSON input file with compositions array")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output JSON path (default: predictions/results.json)")
    parser.add_argument("--grid", action="store_true", help="Grid search Ta-Nb-Ni-Co-Hf")
    parser.add_argument("--all", action="store_true", help="Full pipeline")
    args = parser.parse_args()

    if args.train or args.all:
        print("Building dataset from resistivity_results.json...")
        if not RAW_DATA.exists():
            print(f"ERROR: {RAW_DATA} not found. Place resistivity_results.json in data/")
            sys.exit(1)
        X, y, names = build_dataset_from_json()
        print(f"Training {args.n_models} models...")
        train_ensemble(X, y, n_models=args.n_models)

    if args.predict or args.all or args.grid or args.comp:
        if not (MODELS_DIR / "R_config.json").exists():
            print("Models not found. Run with --train first.")
            sys.exit(1)
        models, scalers = load_ensemble()

    if args.comp:
        comp = parse_composition_string(args.comp)
        mean_r, std_r, preds = predict_one(comp, models, scalers)
        print(f"\n{args.comp}")
        print(f"  R = {mean_r:.2f} ± {std_r:.4f} µΩ·cm")

    if args.grid or args.all:
        print("\nGrid searching Ta-Nb-Ni-Co-Hf...")
        results = predict_ta_nb_ni_co_hf(models, scalers)
        out_path = PRED_DIR / (args.output if args.output else "TaNbNiCoHf_predictions.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} predictions to {out_path}")

        # Summary
        r_vals = [r["R (µΩ·cm)"] for r in results]
        print(f"\nR range: {min(r_vals):.2f} - {max(r_vals):.2f}")
        print(f"R mean: {np.mean(r_vals):.2f} µΩ·cm")
        results.sort(key=lambda x: x["R (µΩ·cm)"])
        print("\nTop 5 lowest:")
        for r in results[:5]:
            print(f"  {r['R (µΩ·cm)']:.2f}  {r['composition_str']}")

    if args.input:
        with open(args.input) as f:
            comps = json.load(f)
        results = []
        for item in comps:
            comp = item if isinstance(item, dict) else parse_composition_string(item)
            try:
                mean_r, std_r, _ = predict_one(comp, models, scalers)
                results.append({"composition": comp, "R": round(mean_r, 2), "R_std": round(std_r, 4)})
            except Exception as e:
                results.append({"composition": comp, "error": str(e)})
        out_path = PRED_DIR / args.output
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
