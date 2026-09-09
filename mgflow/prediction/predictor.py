"""Predict metallic glass properties from composition.

Supports three prediction modes:
  1. "pytorch" — neural network prediction using trained PyTorch models
  2. "python" — GPR prediction using exported MATLAB model parameters
  3. "matlab" — calls MATLAB directly via subprocess

Usage:
    from mgflow import Predictor

    # Mode 1: PyTorch (recommended, no MATLAB needed)
    pred = Predictor("pytorch")
    result = pred.predict({"Zr": 0.5, "Cu": 0.3, "Ni": 0.2})

    # Mode 2: Python GPR (after export from MATLAB)
    pred = Predictor("python", model_dir="exported_models/")
    result = pred.predict({"Zr": 0.5, "Cu": 0.3, "Ni": 0.2})

    # BMG / Ribbon classification
    result = pred.predict({"Zr": 0.5, "Cu": 0.3, "Ni": 0.2})
    # Returns: {"Tg": ..., "Tx": ..., "Tl": ..., "E": ..., "H": ...,
    #           "MG": ..., "BMG": ..., "Ribbon": ..., "BMG_prob": ...}
"""

import json
import pickle
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from ..features.compute import compute_all_features
from ..data.loader import composition_vector

# ---------------------------------------------------------------------------
# 25-feat (Python) → 21-feat (training data) mapping
# ---------------------------------------------------------------------------
# Python compute_all_features() returns 25 features.
# The Excel training datasets only have 21 features.
FEAT25_DROP_IDX = [4, 5, 7, 24]  # dn_dc5, dn_dc6, ye_elastic_energy, Eig_Hessian
FEAT25_KEEP_IDX = [i for i in range(25) if i not in FEAT25_DROP_IDX]  # 21 indices

# ---------------------------------------------------------------------------
# Reduction info (from MATLAB local_training.m feature selection)
# ---------------------------------------------------------------------------
_REDUCTION = {
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

    Returns list of 0-indexed column indices (into the 21-feat space) to KEEP.
    """
    cols = list(range(original_n))
    for k in range(k_min):
        idx = int(imin_1based[k]) - 1  # MATLAB → 0-indexed
        cols.pop(idx)
    return cols  # 0-indexed columns into 21-feat array


# Pre-compute keep-columns for each property
_KEEP_COLS_21 = {}
for prop, info in _REDUCTION.items():
    _KEEP_COLS_21[prop] = _apply_reduction(info["i_min"], info["k_min"])

# ---------------------------------------------------------------------------
# Mode 1: Python GPR prediction (MATLAB export)
# ---------------------------------------------------------------------------


def _rational_quadratic_kernel(x1, x2, sigma_l, sigma_f, alpha):
    dist2 = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
    r2 = dist2 / sigma_l ** 2
    return sigma_f ** 2 * (1 + r2 / (2 * alpha)) ** (-alpha)


class _GPRModel:
    """Minimal GPR predictor matching MATLAB's fitrgp with rational quadratic kernel."""

    def __init__(self, param_dict):
        self.Alpha = param_dict["Alpha"].ravel()
        self.X = param_dict["ActiveSetVectors"]
        self.Sigma = float(param_dict["Sigma"].ravel()[0])
        self.Beta = param_dict["Beta"].ravel()
        ki = param_dict["KernelInfo"]
        kp = ki["KernelParameters"].ravel()
        if "StandardizeMu" in param_dict and param_dict["StandardizeMu"].size > 0:
            self.standardize_mu = param_dict["StandardizeMu"].ravel()
            self.standardize_sigma = param_dict["StandardizeSigma"].ravel()
        else:
            self.standardize_mu = None
            self.standardize_sigma = None
        self.sigma_l = float(kp[0])
        self.sigma_f = float(kp[1])
        self.alpha_k = float(kp[2]) if len(kp) >= 3 else 1.0

    def predict(self, X_new):
        if self.standardize_mu is not None:
            X_new = (X_new - self.standardize_mu) / self.standardize_sigma
        K_star = _rational_quadratic_kernel(X_new, self.X, self.sigma_l, self.sigma_f, self.alpha_k)
        H_new = np.ones((X_new.shape[0], 1))
        return (H_new @ self.Beta + K_star @ self.Alpha).ravel()


# ---------------------------------------------------------------------------
# Mode 2: PyTorch NN prediction
# ---------------------------------------------------------------------------


class _PyTorchModel:
    """Wrapper for a trained PyTorch NN model with its scaler."""

    def __init__(self, prop, model_path, scaler_path, keep_cols_21):
        import torch

        self.prop = prop
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keep_cols_21 = keep_cols_21

        # Load scaler
        scaler_data = np.load(scaler_path)
        self.X_mean = scaler_data["X_mean"]
        self.X_std = scaler_data["X_std"]
        self.y_mean = float(scaler_data["y_mean"])
        self.y_std = float(scaler_data["y_std"])

        # Build model with correct input size
        n_input = len(self.keep_cols_21)
        from ..models.neural_net import ARCH_CONFIGS

        cfg = ARCH_CONFIGS[prop]
        model_class = cfg["class"]
        self.model = model_class(n_input, hidden_dims=cfg["hidden_dims"], dropout=0.0)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, feat_25):
        """Predict from 25-feature vector."""
        import torch

        # 25 → 21 → reduced
        feat_21 = feat_25[FEAT25_KEEP_IDX]
        feat_reduced = feat_21[self.keep_cols_21]

        # Standardize
        feat_scaled = (feat_reduced - self.X_mean) / self.X_std

        # NN forward
        x = torch.tensor(feat_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y_scaled = self.model(x).cpu().numpy().ravel()

        # Unscale
        return float(y_scaled[0] * self.y_std + self.y_mean)


# ---------------------------------------------------------------------------
# BMG / Ribbon classifier (RandomForest)
# ---------------------------------------------------------------------------


class _BMGClassifier:
    """RandomForest classifier for BMG vs Ribbon, trained on 25 features."""

    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, feat_25):
        """Predict BMG probability and class."""
        proba = self.model.predict_proba(feat_25.reshape(1, -1))[0]
        # Class 1 = BMG, Class 0 = Ribbon
        bmg_prob = float(proba[1])
        bmg = 1 if bmg_prob >= 0.5 else 0
        return bmg, bmg_prob


# ---------------------------------------------------------------------------
# Reduction matrices (for "python" mode with exported MATLAB models)
# ---------------------------------------------------------------------------

_REDUCTION_FILES = {
    "Tg": "reduction_Tg.mat",
    "Tx": "reduction_Tx.mat",
    "Tl": "reduction_Tl.mat",
    "E":  "reduction_E.mat",
    "H":  "reduction_H.mat",
}

_MODEL_FILES = {
    "Tg": "gpr_Tg.mat",
    "Tx": "gpr_Tx.mat",
    "Tl": "gpr_Tl.mat",
    "E":  "gpr_E.mat",
    "H":  "gpr_H.mat",
}


# ---------------------------------------------------------------------------
# Predictor — main interface
# ---------------------------------------------------------------------------


class Predictor:
    """Predict metallic glass properties from composition.

    Parameters
    ----------
    mode : str
        "pytorch" — use trained PyTorch NN models (recommended, no MATLAB needed)
        "python"  — use exported GPR models (run matlab_export_models.m first)
        "matlab"  — call MATLAB directly (requires MATLAB on PATH)
    model_dir : str or Path, optional
        Path to model directory. For "pytorch" mode, defaults to
        mgflow/pytorch_models/. For "python" mode, defaults to exported_models/.
    """

    PROPERTIES = ["Tg", "Tx", "Tl", "E", "H"]
    PROP_UNITS = {"Tg": "K", "Tx": "K", "Tl": "K", "E": "GPa", "H": "GPa"}

    def __init__(self, mode="pytorch", model_dir=None):
        self.mode = mode
        self._models = {}
        self._bmg_clf = None

        if mode == "pytorch":
            if model_dir is None:
                model_dir = Path(__file__).parent / "artifacts" / "pytorch_models"
            self.model_dir = Path(model_dir)
            self._load_pytorch_models()
            self._load_bmg_classifier()
        elif mode == "python":
            if model_dir is None:
                model_dir = Path.cwd() / "exported_models"
            self.model_dir = Path(model_dir)
            self._load_gpr_models()
        elif mode == "matlab":
            try:
                subprocess.run(
                    ["matlab", "-batch", "disp('ok')"], capture_output=True, timeout=10
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                raise RuntimeError(
                    "MATLAB not found on PATH. Install MATLAB or use mode='pytorch'."
                )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'pytorch', 'python', or 'matlab'.")

    def _load_bmg_classifier(self):
        """Load the BMG/Ribbon RandomForest classifier."""
        clf_path = self.model_dir / "classifier_bmg_rf.pkl"
        if clf_path.exists():
            self._bmg_clf = _BMGClassifier(clf_path)
            print(f"  Loaded BMG/Ribbon classifier")
        else:
            print(f"  [WARN] BMG classifier not found: {clf_path} — run train_classifier.py first")

    def _load_pytorch_models(self):
        """Load all trained PyTorch models."""
        for prop in self.PROPERTIES:
            model_path = self.model_dir / f"model_{prop}_full.pt"
            scaler_path = self.model_dir / f"scaler_{prop}_full.npz"

            if not model_path.exists():
                model_path = self.model_dir / f"model_{prop}.pt"
            if not model_path.exists():
                print(f"  [WARN] PyTorch model not found: {model_path} — skipping {prop}")
                continue
            if not scaler_path.exists():
                print(f"  [WARN] Scaler not found: {scaler_path} — skipping {prop}")
                continue

            keep_cols = _KEEP_COLS_21[prop]
            self._models[prop] = _PyTorchModel(prop, model_path, scaler_path, keep_cols)
            print(f"  Loaded {prop}: PyTorch NN ({len(keep_cols)} features)")

        if not self._models:
            raise FileNotFoundError(
                f"No PyTorch models found in {self.model_dir}. "
                f"Run `python -m mgflow.train_pytorch` first."
            )
        print(f"  Ready to predict: {list(self._models.keys())}")

    def _load_gpr_models(self):
        """Load all exported GPR models (from MATLAB)."""
        from scipy.io import loadmat

        for prop in self.PROPERTIES:
            modelf = self.model_dir / _MODEL_FILES[prop]
            reducf = self.model_dir / _REDUCTION_FILES[prop]

            if not modelf.exists():
                print(f"  [WARN] GPR model not found: {modelf} — skipping {prop}")
                continue

            data = loadmat(str(modelf))
            self._models[prop] = {"gpr": _GPRModel(data), "reduction": None}

            if reducf.exists():
                rdata = loadmat(str(reducf))
                if "i_min" in rdata:
                    i_min = rdata["i_min"].ravel().astype(int) - 1
                    self._models[prop]["reduction"] = i_min
                    print(f"  Loaded {prop}: GPR + reduction")

        if not self._models:
            raise FileNotFoundError(
                f"No exported GPR models found in {self.model_dir}. "
                f"Run matlab_export_models.m in MATLAB first."
            )
        print(f"  Ready to predict: {list(self._models.keys())}")

    def predict(self, composition, verbose=False):
        """Predict properties and classification for a given composition.

        Parameters
        ----------
        composition : dict or list of tuples
            e.g. {"Zr": 0.5, "Cu": 0.3, "Ni": 0.2}
        verbose : bool
            Print feature values.

        Returns
        -------
        dict : {
            "Tg": float, "Tx": float, "Tl": float, "E": float, "H": float,
            "MG": int, "BMG": int, "Ribbon": int, "BMG_prob": float
        }
        """
        if self.mode == "pytorch":
            return self._predict_pytorch(composition, verbose)
        elif self.mode == "python":
            return self._predict_gpr(composition, verbose)
        else:
            return self._predict_matlab(composition, verbose)

    def predict_batch(self, compositions, verbose=False):
        """Predict properties for multiple compositions."""
        return [self.predict(c, verbose=verbose) for c in compositions]

    # ----- BMG classification -----

    def _predict_classification(self, feat_25):
        """Predict MG, BMG, Ribbon from 25-feature vector."""
        result = {"MG": 1}  # All labeled samples are MGs
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

    # ----- PyTorch prediction -----

    def _predict_pytorch(self, composition, verbose=False):
        """Predict using PyTorch NN models + BMG classifier."""
        from ..data.loader import ELEMENT_SYMBOLS

        elements, fractions = composition_vector(composition)
        feat_all = compute_all_features(composition)  # (25,)

        if verbose:
            from ..features.compute import FEATURE_NAMES_25

            comp_str = {ELEMENT_SYMBOLS[e]: float(f) for e, f in zip(elements, fractions)}
            print(f"\nComposition: {comp_str}")
            print(f"{'Feature':<25s} {'Value':>12s}")
            print("-" * 38)
            for name, val in zip(FEATURE_NAMES_25, feat_all):
                print(f"{name:<25s} {val:>12.6f}")
            print()

        result = self._predict_classification(feat_all)
        for prop in self.PROPERTIES:
            model = self._models.get(prop)
            if model is None:
                result[prop] = None
                continue
            result[prop] = model.predict(feat_all)

        return result

    # ----- GPR prediction (from exported MATLAB models) -----

    def _predict_gpr(self, composition, verbose=False):
        from ..data.loader import ELEMENT_SYMBOLS

        elements, fractions = composition_vector(composition)
        feat_all = compute_all_features(composition)

        if verbose:
            from ..features.compute import FEATURE_NAMES_25

            comp_str = {ELEMENT_SYMBOLS[e]: float(f) for e, f in zip(elements, fractions)}
            print(f"\nComposition: {comp_str}")
            print(f"{'Feature':<25s} {'Value':>12s}")
            print("-" * 38)
            for name, val in zip(FEATURE_NAMES_25, feat_all):
                print(f"{name:<25s} {val:>12.6f}")
            print()

        result = self._predict_classification(feat_all)
        for prop in self.PROPERTIES:
            model_data = self._models.get(prop)
            if model_data is None:
                result[prop] = None
                continue

            feat = feat_all.copy()
            if model_data["reduction"] is not None:
                idx_remove = model_data["reduction"]
                mask = np.ones(len(feat), dtype=bool)
                for ii in idx_remove:
                    if ii < len(mask):
                        mask[ii] = False
                feat = feat[mask]

            gpr = model_data["gpr"]
            y = gpr.predict(feat.reshape(1, -1))
            result[prop] = float(y[0])

        return result

    # ----- MATLAB bridge prediction -----

    def _predict_matlab(self, composition, verbose=False):
        elements, fractions = composition_vector(composition)
        feat = compute_all_features(composition)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            feat_path = tmpdir / "features_in.csv"
            out_path = tmpdir / "predictions.json"

            np.savetxt(feat_path, feat.reshape(1, -1), delimiter=",")

            matlab_script = f"""
            addpath('{Path.cwd()}/Matlab 程序文件夹/TgTxTl_E_H/TgTxTl_E_H');
            xx = csvread('{feat_path}');

            load('RQGPR_Tg_r.mat');
            load('RQGPR_Tx_r.mat');
            load('RQGPR_Tl_r8_new.mat');
            load('RQGPR_E_r10.mat');
            load('H_8_reduced_byziqing.mat');

            load('Tg_reduction.mat');
            load('Tx_reduction.mat');
            load('Tl_reduced_r_new_0.mat');
            load('E_reduction.mat');
            load('H_reduce_new_byziqing.mat');

            xx_Tg=xx; xx_Tx=xx; xx_Tl=xx; xx_E=xx; xx_H=xx;

            for k=1:find(RMSE_mean_min==min(min(RMSE_mean_min)))
                xx_E(:,i_min(k))=[];
            end
            load('H_reduce_new_byziqing.mat');
            for k=1:find(RMSE_mean_min==min(min(RMSE_mean_min)))
                xx_H(:,i_min(k))=[];
            end
            load('Tg_reduction.mat');
            for k=1:find(RMSE_mean_min==min(min(RMSE_mean_min)))
                xx_Tg(:,i_min(k))=[];
            end
            load('Tx_reduction.mat');
            for k=1:find(RMSE_mean_min==min(min(RMSE_mean_min)))
                xx_Tx(:,i_min(k))=[];
            end
            load('Tl_reduced_r_new_0.mat');
            for k=1:find(RMSE_mean_min==min(min(RMSE_mean_min)))
                xx_Tl(:,i_min(k))=[];
            end

            Tg = RQGPR_Tg_r.predictFcn(xx_Tg);
            Tx = RQGPR_Tx_r.predictFcn(xx_Tx);
            Tl = RQGPR_Tl_r8_new.predictFcn(xx_Tl);
            E  = RQGPR_E_r10.predictFcn(xx_E);
            H  = H_8.predictFcn(xx_H);

            result = jsonencode(struct('Tg',Tg,'Tx',Tx,'Tl',Tl,'E',E,'H',H));
            fid = fopen('{out_path}','w');
            fprintf(fid, '%%s', result);
            fclose(fid);
            exit;
            """

            script_path = tmpdir / "predict_bridge.m"
            with open(script_path, "w") as f:
                f.write(matlab_script)

            try:
                subprocess.run(
                    ["matlab", "-batch", f"run('{script_path}')"],
                    capture_output=True, timeout=60,
                    cwd=Path.cwd()
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("MATLAB prediction timed out (60s).")

            if out_path.exists():
                result = json.loads(out_path.read_text())
                # Add classification (matlab mode doesn't have BMG classifier)
                result["MG"] = 1
                result["BMG"] = None
                result["Ribbon"] = None
                result["BMG_prob"] = None
                return result
            else:
                raise RuntimeError("MATLAB prediction failed. Check MATLAB output.")

    def __repr__(self):
        return (
            f"Predictor(mode={self.mode!r}, "
            f"models={list(self._models.keys()) if hasattr(self, '_models') else 'N/A'})"
        )
