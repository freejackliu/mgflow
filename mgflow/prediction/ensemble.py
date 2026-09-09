"""Ensemble-based predictor with uncertainty quantification.

Provides an EnsemblePredictor that trains/loads M models per property
and returns both mean prediction and uncertainty (std across ensemble).

Usage:
    from mgflow.ensemble_predict import EnsemblePredictor

    # Train-first mode (if models don't exist, they're auto-trained)
    ep = EnsemblePredictor(n_models=20)

    # Predict with uncertainty
    result = ep.predict({"Zr": 0.5, "Cu": 0.3, "Ni": 0.2})
    # Returns: {"Tg": val, "Tg_std": unc, "Tx": ..., ...}

    # Batch prediction
    results = ep.predict_batch([comp1, comp2, ...])
"""

import json
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from ..models.neural_net import PropertyNN, PropertyNN_Small, ARCH_CONFIGS
from ..features.compute import compute_all_features
from ..data.loader import composition_vector
from .predictor import FEAT25_DROP_IDX, _KEEP_COLS_21, _BMGClassifier

_ENSEMBLE_DIR = Path(__file__).resolve().parent / "artifacts" / "ensembles"


class _EnsembleModel:
    """Container for M models of one property with their scalers."""

    def __init__(self, prop, models, scalers):
        self.prop = prop
        self.models = models
        self.scalers = scalers
        self.n_models = len(models)
        self.device = next(models[0].parameters()).device

        # All scalers share the same keep_cols_21
        self.keep_cols_21 = scalers[0]["keep_cols_21"]

    @classmethod
    def load(cls, prop, device="cpu", ensemble_dir=None):
        """Load an ensemble from saved files."""
        if ensemble_dir is None:
            ensemble_dir = _ENSEMBLE_DIR
        ensemble_dir = Path(ensemble_dir)

        # Load config
        config_path = ensemble_dir / f"{prop}_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Ensemble config not found: {config_path}")
        with open(config_path) as f:
            config = json.load(f)

        n_models = config["n_models"]
        n_features = config["n_features"]
        hidden_dims = tuple(config["hidden_dims"])
        dropout = config["dropout"]
        model_class_name = config["model_class"]

        # Load scalers
        scaler_path = ensemble_dir / f"{prop}_scalers.npz"
        scaler_data = np.load(scaler_path)
        keep_cols_21 = scaler_data["keep_cols_21"]

        scalers = []
        for m in range(n_models):
            scalers.append({
                "X_mean": scaler_data[f"X_mean_{m}"],
                "X_std": scaler_data[f"X_std_{m}"],
                "y_mean": float(scaler_data[f"y_mean_{m}"][0]),
                "y_std": float(scaler_data[f"y_std_{m}"][0]),
                "keep_cols_21": keep_cols_21,
            })

        # Load model weights
        model_class = PropertyNN if model_class_name == "PropertyNN" else PropertyNN_Small
        models = []
        for m in range(n_models):
            model_path = ensemble_dir / f"{prop}_model_{m:03d}.pt"
            model = model_class(n_features, hidden_dims=hidden_dims, dropout=0.0)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            models.append(model)

        print(f"  Loaded {prop} ensemble: {n_models} models")
        return cls(prop, models, scalers)

    def predict(self, feat_25):
        """Predict property value and uncertainty from 25-feature vector.

        Returns
        -------
        mean_val : float
        std_val : float
        individual_preds : list of float
        """
        # 25 → 21 → reduced
        feat_21 = feat_25[[i for i in range(25) if i not in FEAT25_DROP_IDX]]
        feat_reduced = feat_21[self.keep_cols_21]

        preds = []
        for model, scaler in zip(self.models, self.scalers):
            feat_scaled = (feat_reduced - scaler["X_mean"]) / scaler["X_std"]
            x = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                y_scaled = model(x).cpu().numpy().ravel()
            y_val = float(y_scaled[0] * scaler["y_std"] + scaler["y_mean"])
            preds.append(y_val)

        return float(np.mean(preds)), float(np.std(preds, ddof=1)), preds


class EnsemblePredictor:
    """Predictor with uncertainty quantification via model ensemble.

    Uses M independently trained NN models per property to provide:
      - mean prediction (more accurate than single model)
      - uncertainty estimate (std across ensemble)

    Parameters
    ----------
    n_models : int
        Number of models per property (default: 20).
        If saved ensemble exists, loads them; otherwise trains automatically.
    device : str, optional
        "cuda" or "cpu". Auto-detects if None.
    ensemble_dir : str or Path, optional
        Directory with ensemble models.
    force_retrain : bool
        If True, retrain even if saved models exist.
    """

    PROPERTIES = ["Tg", "Tx", "Tl", "E", "H"]

    def __init__(self, n_models=20, device=None, ensemble_dir=None, force_retrain=False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.n_models = n_models
        self._ensemble_models = {}
        self._bmg_clf = None

        if ensemble_dir is None:
            ensemble_dir = _ENSEMBLE_DIR
        self.ensemble_dir = Path(ensemble_dir)

        # Check if ensemble exists
        config_paths = [self.ensemble_dir / f"{p}_config.json" for p in self.PROPERTIES]
        all_exist = all(cp.exists() for cp in config_paths)

        if not all_exist or force_retrain:
            print("Ensemble models not found or force_retrain=True. Training now...")
            self._train_ensembles()

        # Load all ensembles
        self._load_ensembles()
        self._load_bmg_classifier()

    def _train_ensembles(self):
        """Auto-train ensemble models."""
        from .ensemble_train import train_ensemble_all
        train_ensemble_all(n_models=self.n_models, device=self.device)

    def _load_ensembles(self):
        """Load all ensemble models."""
        for prop in self.PROPERTIES:
            try:
                self._ensemble_models[prop] = _EnsembleModel.load(
                    prop, device=self.device, ensemble_dir=self.ensemble_dir
                )
            except FileNotFoundError as e:
                print(f"  [WARN] Could not load {prop} ensemble: {e}")

        if not self._ensemble_models:
            raise FileNotFoundError(
                f"No ensemble models found in {self.ensemble_dir}. "
                f"Run `python -m mgflow.ensemble_train` first."
            )
        print(f"  Ready to predict with uncertainty: {list(self._ensemble_models.keys())}")

    def _load_bmg_classifier(self):
        """Load the BMG/Ribbon RandomForest classifier."""
        clf_dir = Path(__file__).resolve().parent / "artifacts" / "pytorch_models"
        clf_path = clf_dir / "classifier_bmg_rf.pkl"
        if clf_path.exists():
            self._bmg_clf = _BMGClassifier(clf_path)
            print(f"  Loaded BMG/Ribbon classifier")
        else:
            print(f"  [WARN] BMG classifier not found: {clf_path}")

    def _predict_classification(self, feat_25):
        """Predict MG, BMG, Ribbon from 25-feature vector."""
        result = {"MG": 1}
        if self._bmg_clf is not None:
            bmg, bmg_prob = self._bmg_clf.predict(feat_25)
            result["BMG"] = bmg
            result["Ribbon"] = 1 - bmg
            result["BMG_prob"] = round(bmg_prob, 4)
        else:
            result["BMG"] = None
            result["Ribbon"] = None
            result["BMG_prob"] = None
        return result

    def predict(self, composition, verbose=False):
        """Predict properties with uncertainty for a given composition.

        Parameters
        ----------
        composition : dict or list of tuples
            e.g. {"Zr": 0.5, "Cu": 0.3, "Ni": 0.2}
        verbose : bool
            Print feature values.

        Returns
        -------
        dict : {
            "Tg": float, "Tg_std": float, "Tg_preds": [float, ...],
            "Tx": float, "Tx_std": float, ...
            "Tl": float, "Tl_std": float, ...
            "E": float, "E_std": float, ...
            "H": float, "H_std": float, ...
            "BMG_prob": float, ...
        }
        """
        from ..data.loader import ELEMENT_SYMBOLS
        from ..features.compute import FEATURE_NAMES_25

        elements, fractions = composition_vector(composition)
        feat_all = compute_all_features(composition)

        if verbose:
            comp_str = {ELEMENT_SYMBOLS[e]: float(f) for e, f in zip(elements, fractions)}
            print(f"\nComposition: {comp_str}")
            print(f"{'Feature':<25s} {'Value':>12s}")
            print("-" * 38)
            for name, val in zip(FEATURE_NAMES_25, feat_all):
                print(f"{name:<25s} {val:>12.6f}")
            print()

        result = self._predict_classification(feat_all)

        for prop in self.PROPERTIES:
            ensemble = self._ensemble_models.get(prop)
            if ensemble is None:
                result[prop] = None
                result[f"{prop}_std"] = None
                result[f"{prop}_preds"] = None
                continue

            mean_val, std_val, preds = ensemble.predict(feat_all)
            result[prop] = round(mean_val, 2)
            result[f"{prop}_std"] = round(std_val, 4)
            result[f"{prop}_preds"] = [round(p, 2) for p in preds]

        return result

    def predict_batch(self, compositions, verbose=False):
        """Predict properties with uncertainty for multiple compositions."""
        return [self.predict(c, verbose=verbose) for c in compositions]

    def suggest_exploration(self, composition, explore_weight=0.1):
        """Compute an exploration score for a composition.

        Higher score = more valuable to explore (high uncertainty).

        Score = (1 - explore_weight) * (-mean_Tl) + explore_weight * mean_uncertainty
        where mean_uncertainty = average of std across properties.

        Parameters
        ----------
        composition : dict
        explore_weight : float (0 to 1)
            0 = pure exploitation, 1 = pure exploration.

        Returns
        -------
        dict with "exploitation_score", "exploration_score", "combined_score"
        """
        result = self.predict(composition)
        exploitation = -result.get("Tl", 0) if result.get("Tl") else 0

        uncertainties = []
        for prop in self.PROPERTIES:
            std_val = result.get(f"{prop}_std")
            if std_val is not None:
                uncertainties.append(std_val)
        exploration = float(np.mean(uncertainties)) if uncertainties else 0.0

        combined = (1 - explore_weight) * exploitation + explore_weight * exploration
        return {
            "exploitation_score": round(exploitation, 2),
            "exploration_score": round(exploration, 4),
            "combined_score": round(combined, 4),
        }

    def predict_with_individuals(self, composition):
        """Predict and return all individual model predictions (for analysis)."""
        from ..data.loader import composition_vector
        from ..features.compute import compute_all_features

        elements, fractions = composition_vector(composition)
        feat_all = compute_all_features(composition)

        result = {}
        for prop in self.PROPERTIES:
            ensemble = self._ensemble_models.get(prop)
            if ensemble is None:
                continue
            mean_val, std_val, preds = ensemble.predict(feat_all)
            result[prop] = {"mean": mean_val, "std": std_val, "preds": preds}
        return result

    def __repr__(self):
        return (
            f"EnsemblePredictor(n_models={self.n_models}, "
            f"properties={list(self._ensemble_models.keys())})"
        )
