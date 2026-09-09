"""
VAE-based generator for High Residual Resistivity (R) compositions.

Architecture
------------
  VAE learns a smooth latent manifold over 6-property composition space
  (Tg/Tx/Tl/E/H/R ≈ 3800 samples, 53 elements).

  High-R search strategy:
    1. Randomly sample latent vectors z ~ N(0, I)
    2. Decode → composition (53-dim, sum-to-1 via softmax)
    3. Predict R using the R ensemble (from mgflow/resistivity/models/)
    4. Rank by R, keep top candidates
    5. (Optional) CMA-ES evolution in latent space to climb the R gradient

Usage
-----
  # Train VAE on all 6 properties (run once)
  python -c "from mgflow.resistivity.inverse_high_r import train_vae_r; train_vae_r()"

  # Generate high-R compositions
  python -c "
from mgflow.resistivity.inverse_high_r import generate_high_R
results = generate_high_R(n_samples=5000)
print(results['best'])
"

  # Full pipeline
  from mgflow.resistivity.inverse_high_r import train_vae_r, generate_high_R
  train_vae_r()
  results = generate_high_R(n_samples=10000, top_k=50)
"""

import time, json, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Paths ──
_HERE = Path(__file__).resolve().parent
_MGFLOW = _HERE.parent
_R_MODELS = _HERE / "models"
_VAE_MODELS = _HERE / "vae_models"
_VAE_MODELS.mkdir(parents=True, exist_ok=True)

import sys
if str(_MGFLOW.parent) not in sys.path:
    sys.path.insert(0, str(_MGFLOW.parent))

from mgflow.data.loader import ELEMENT_SYMBOLS, SYMBOL_TO_NO, load_dataset
from mgflow.features.compute import compute_all_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[vae_r_high] Device: {DEVICE}")

# ════════════════════════════════════════════════════════════════
# 1. VAE element set (same as vae.py, but also covers R)
# ════════════════════════════════════════════════════════════════
VAE_R_MODEL_KEY = "vae_r_lat32_b0.01"

def _build_elements():
    """53 elements from the 5 existing properties."""
    all_set = set()
    for p in ["Tg", "Tx", "Tl", "E", "H"]:
        data = load_dataset(p, include_metadata=False)
        for row in data["elem_nos"]:
            for e in row:
                if e > 0:
                    all_set.add(int(e))
    return sorted(all_set)

VAE_ELEMENTS = _build_elements()
VAE_SYMBOLS = [ELEMENT_SYMBOLS[e] for e in VAE_ELEMENTS]
N_ELEMENTS = len(VAE_ELEMENTS)
ELEM_TO_VAE_IDX = {e: i for i, e in enumerate(VAE_ELEMENTS)}
print(f"[vae_r_high] {N_ELEMENTS} elements: {VAE_SYMBOLS}")


# ════════════════════════════════════════════════════════════════
# 2. Dataset (extends vae.py to include R compositions)
# ════════════════════════════════════════════════════════════════

def _parse_r_compositions():
    """Parse resistivity_results.json → list of VAE vectors."""
    src = _HERE / "data" / "resistivity_results.json"
    with open(src) as f:
        raw = json.load(f)

    vectors = []
    for cstr in raw:
        vec = np.zeros(N_ELEMENTS, dtype=np.float32)
        total = 0.0
        for sym, val in re.findall(r'([A-Z][a-z]?)(\d+\.?\d*)', cstr):
            an = SYMBOL_TO_NO.get(sym)
            if an is not None and an in ELEM_TO_VAE_IDX:
                f = float(val) / 100.0
                vec[ELEM_TO_VAE_IDX[an]] = f
                total += f
        if total > 0:
            vec /= total
        vectors.append(vec)
    return vectors


class CompositionDatasetR(Dataset):
    """VAE dataset: all 6 properties including R."""

    def __init__(self):
        # 5 existing properties
        vectors = []
        for p in ["Tg", "Tx", "Tl", "E", "H"]:
            data = load_dataset(p, include_metadata=False)
            for en, fr in zip(data["elem_nos"], data["fracs"]):
                vec = np.zeros(N_ELEMENTS, dtype=np.float32)
                total = 0.0
                for e, f in zip(en, fr):
                    if e > 0:
                        vec[ELEM_TO_VAE_IDX[int(e)]] = float(f)
                        total += float(f)
                if total > 0:
                    vec /= total
                vectors.append(vec)

        # R data
        r_vecs = _parse_r_compositions()
        vectors.extend(r_vecs)

        self.vectors = np.array(vectors, dtype=np.float32)
        print(f"[CompositionDatasetR] {len(self.vectors)} total samples "
              f"(incl. {len(r_vecs)} from R)  dim={N_ELEMENTS}")

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]


# ════════════════════════════════════════════════════════════════
# 3. VAE (same architecture as vae.py)
# ════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, n_input, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev = n_input
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)])
            prev = h
        self.shared = nn.Sequential(*layers)
        self.mu = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, n_output):
        super().__init__()
        rev = list(reversed(hidden_dims))
        layers = []
        prev = latent_dim
        for h in rev:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)])
            prev = h
        layers.append(nn.Linear(prev, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)  # logits


class CompositionVAE(nn.Module):
    def __init__(self, n_elements=N_ELEMENTS, latent_dim=32, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.n_elements = n_elements
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.encoder = Encoder(n_elements, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, n_elements)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return torch.softmax(self.decoder(z), dim=-1)

    def decode_logits(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar

    def generate(self, n_samples, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        with torch.no_grad():
            return torch.softmax(self.decoder(z), dim=-1).cpu().numpy()

    @property
    def device(self):
        return next(self.parameters()).device


def vae_loss(logits, x, mu, logvar, beta, kl_weight=1.0):
    log_probs = F.log_softmax(logits, dim=-1)
    recon_loss = -torch.sum(x * log_probs, dim=-1).mean()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    eff_beta = beta * kl_weight
    total = recon_loss + eff_beta * kl_loss
    return total, {"recon_loss": recon_loss.item(), "kl_loss": kl_loss.item(),
                   "beta_eff": float(eff_beta), "total": total.item()}


def train_vae_r(latent_dim=32, hidden_dims=None, beta=0.01, epochs=300,
                batch_size=128, lr=1e-3, kl_anneal_epochs=100, force=False):
    """Train VAE on all 6 properties (including R)."""
    model_path = _VAE_MODELS / f"{VAE_R_MODEL_KEY}.pt"
    if model_path.exists() and not force:
        print(f"[train_vae_r] Using existing VAE: {model_path}")
        return load_vae_r()

    if hidden_dims is None:
        hidden_dims = [256, 128, 64]

    dataset = CompositionDatasetR()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    vae = CompositionVAE(N_ELEMENTS, latent_dim, hidden_dims).to(DEVICE)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=20, min_lr=1e-6)

    t0 = time.time()
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        vae.train()
        losses = {"recon_loss": [], "kl_loss": [], "total": [], "beta_eff": []}
        kw = min(1.0, (epoch + 1) / kl_anneal_epochs) if kl_anneal_epochs > 0 else 1.0

        for batch in loader:
            x = batch.to(DEVICE)
            logits, mu, logvar = vae(x)
            loss, comps = vae_loss(logits, x, mu, logvar, beta, kw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0); opt.step()
            for k in losses:
                if k in comps: losses[k].append(comps[k])
                elif k == "beta_eff": losses[k].append(comps.get("beta_eff", float(beta * kw)))

        avg = {k: np.mean(v) for k, v in losses.items()}
        scheduler.step(avg["total"])

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            best_state = {k: v.cpu().clone() for k, v in vae.state_dict().items()}

        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"  [{epoch+1:3d}/{epochs}] recon={avg['recon_loss']:.6f}  "
                  f"kl={avg['kl_loss']:.6f}  β={avg['beta_eff']:.6f}  "
                  f"total={avg['total']:.6f}  [{time.time()-t0:.0f}s]")

    vae.load_state_dict(best_state)
    vae.eval()

    torch.save(vae.state_dict(), model_path)
    cfg = {"n_elements": N_ELEMENTS, "element_symbols": VAE_SYMBOLS,
           "latent_dim": latent_dim, "hidden_dims": hidden_dims, "beta": beta,
           "epochs": epochs, "n_training_samples": len(dataset)}
    with open(model_path.with_suffix(".json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[train_vae_r] Done. Saved to {model_path}")
    return vae


def load_vae_r():
    """Load the R-inclusive VAE."""
    model_path = _VAE_MODELS / f"{VAE_R_MODEL_KEY}.pt"
    cfg_path = model_path.with_suffix(".json")
    if not model_path.exists():
        raise FileNotFoundError(f"Train VAE first: {model_path}")

    with open(cfg_path) as f:
        cfg = json.load(f)
    vae = CompositionVAE(cfg["n_elements"], cfg["latent_dim"], cfg["hidden_dims"])
    vae.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    vae.to(DEVICE)
    vae.eval()
    print(f"[load_vae_r] Loaded VAE: lat={cfg['latent_dim']}, "
          f"hidden={cfg['hidden_dims']}, {cfg['n_training_samples']} samples")
    return vae


# ════════════════════════════════════════════════════════════════
# 4. R Predictor (uses _hpc_resistivity ensemble)
# ════════════════════════════════════════════════════════════════

class RPredictor:
    """Loads the R ensemble from _hpc_resistivity/models/."""

    def __init__(self, n_models=30):
        self.models = []
        self.scalers = []
        self.device = torch.device(DEVICE)
        self.FEAT25_DROP_IDX = [4, 5, 7, 24]

        config_path = _R_MODELS / "R_config.json"
        scaler_path = _R_MODELS / "R_scalers.npz"
        if not config_path.exists():
            raise FileNotFoundError(
                f"R ensemble not found in {_R_MODELS}. "
                f"Run _hpc_resistivity training first.")

        with open(config_path) as f:
            cfg = json.load(f)

        scaler_data = np.load(scaler_path, allow_pickle=True)

        # Build the same PropertyNN architecture as _hpc_resistivity training
        hidden_list = cfg.get("hidden_dims", [128, 64])
        n_feat = cfg["n_features"]

        # PropertyNN matching _hpc_resistivity train script
        class _PropertyNN(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev = n_feat
                for h in hidden_list:
                    layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.0)])
                    prev = h
                layers.append(nn.Linear(prev, 1))
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x).squeeze(-1)

        for m in range(n_models):
            net = _PropertyNN()
            net.load_state_dict(
                torch.load(_R_MODELS / f"R_model_{m:03d}.pt",
                           map_location=self.device, weights_only=True))
            net.to(self.device)
            net.eval()
            self.models.append(net)

            self.scalers.append({
                "X_mean": scaler_data[f"X_mean_{m}"],
                "X_std": scaler_data[f"X_std_{m}"],
                "y_mean": float(scaler_data[f"y_mean_{m}"][0]),
                "y_std": float(scaler_data[f"y_std_{m}"][0]),
                "keep_cols_21": scaler_data["keep_cols_21"],
            })

        self.n_models = n_models
        print(f"[RPredictor] Loaded {n_models}-model R ensemble")

    def predict(self, composition):
        """Predict R (µΩ·cm) with uncertainty.

        Parameters
        ----------
        composition : dict — e.g. {"Ta": 0.5, "Nb": 0.3, ...}

        Returns
        -------
        mean_R, std_R
        """
        feat_25 = compute_all_features(composition)
        feat_21 = feat_25[[i for i in range(25) if i not in self.FEAT25_DROP_IDX]]
        keep = self.scalers[0]["keep_cols_21"]
        feat = feat_21[keep]

        preds = []
        for model, sc in zip(self.models, self.scalers):
            x = (feat - sc["X_mean"]) / sc["X_std"]
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                y = model(xt).item()
            preds.append(y * sc["y_std"] + sc["y_mean"])

        return float(np.mean(preds)), float(np.std(preds, ddof=1))


# ════════════════════════════════════════════════════════════════
# 5. High-R latent optimization
# ════════════════════════════════════════════════════════════════

def vec_to_composition(vec, threshold=0.02):
    """53-dim VAE vector → {symbol: percent} dict."""
    vec = np.asarray(vec, dtype=np.float64)
    idx = np.argsort(vec)[::-1]
    keep = idx[vec[idx] >= threshold]
    if len(keep) == 0:
        return {}
    fracs = vec[keep] / vec[keep].sum()
    return {VAE_SYMBOLS[j]: round(float(fracs[i]) * 100, 2)
            for i, j in enumerate(keep)}


def generate_high_R(n_samples=5000, top_k=50, batch_size=200,
                    threshold=0.025, min_elements=3, max_elements=6,
                    lat_evo_steps=0, lat_evo_top=20,
                    elements=None):
    """Generate high-R compositions via VAE latent space sampling (+ optional evolution).

    Parameters
    ----------
    n_samples : int — total VAE samples to draw
    top_k : int — number of top candidates to return
    batch_size : int — batch size for VAE decode + R predict
    threshold : float — minimum element fraction
    min_elements, max_elements : int — filter by element count
    lat_evo_steps : int — if >0, do iterative latent evolution
        (sample → pick top → add noise → repeat)
    lat_evo_top : int — keep top N per evolution round
    elements : list or None — constrain to only these element symbols
        e.g. ["Ta", "Nb", "Ni", "Co", "Hf"]

    Returns
    -------
    dict with keys: candidates, best, history
    """
    t0 = time.time()

    # Load models
    vae = load_vae_r()
    r_pred = RPredictor()

    # Multi-round evolution or single-shot
    all_candidates = []

    if lat_evo_steps > 0:
        # Latent evolution: iterative refinement
        current_n = n_samples
        for step in range(lat_evo_steps + 1):
            if step == 0:
                print(f"\n[Evo round {step+1}] Sampling {current_n} random z...")
                z = torch.randn(current_n, vae.latent_dim, device=DEVICE)
            else:
                print(f"\n[Evo round {step+1}] Perturbing top {current_n} z's...")
                noise = torch.randn(current_n, vae.latent_dim, device=DEVICE) * 0.3
                z += noise  # Walk from previous best

            with torch.no_grad():
                decoded = vae.decode(z).cpu().numpy()

            # Filter + predict
            batch_comps = []
            for row in decoded:
                comp = vec_to_composition(row, threshold)
                if not (min_elements <= len(comp) <= max_elements):
                    continue
                if elements is not None:
                    comp = {k: v for k, v in comp.items() if k in elements}
                    if len(comp) < min_elements:
                        continue
                batch_comps.append(comp)

            # Predict R
            for idx_in_batch in range(0, len(batch_comps), batch_size):
                chunk = batch_comps[idx_in_batch:idx_in_batch + batch_size]
                frac_chunk = [{k: v / 100.0 for k, v in c.items()} for c in chunk]
                for comp, frac in zip(chunk, frac_chunk):
                    try:
                        r_mean, r_std = r_pred.predict(frac)
                        all_candidates.append((r_mean, r_std, comp))
                    except Exception:
                        continue

            if step < lat_evo_steps:
                # Keep top lat_evo_top → update z
                all_candidates.sort(key=lambda x: -x[0])
                top_hits = all_candidates[:lat_evo_top]
                # Get the z's that produced these
                # Need to map back — simpler: just regenerate from scratch
                top_z_list = []
                for r_val, r_s, comp in top_hits:
                    # Re-encode this composition into latent space
                    vec = np.zeros(N_ELEMENTS, dtype=np.float32)
                    for sym, pct in comp.items():
                        an = SYMBOL_TO_NO.get(sym)
                        if an is not None and an in ELEM_TO_VAE_IDX:
                            vec[ELEM_TO_VAE_IDX[an]] = pct / 100.0
                    if vec.sum() > 0:
                        vec /= vec.sum()
                    xt = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        mu, _ = vae.encode(xt)
                    top_z_list.append(mu)
                if top_z_list:
                    z = torch.cat(top_z_list, dim=0)
                    current_n = z.shape[0]
                else:
                    current_n = n_samples
                    z = torch.randn(current_n, vae.latent_dim, device=DEVICE)

            print(f"  → {len(all_candidates)} candidates so far")
    else:
        # Single-shot sampling
        print(f"\n[Generate] Sampling {n_samples} latent vectors...")
        all_decoded = vae.generate(n_samples)

        # Filter
        batch_comps = []
        for row in all_decoded:
            comp = vec_to_composition(row, threshold)
            if not (min_elements <= len(comp) <= max_elements):
                continue
            if elements is not None:
                comp = {k: v for k, v in comp.items() if k in elements}
                if len(comp) < min_elements:
                    continue
            batch_comps.append(comp)

        print(f"  → {len(batch_comps)} valid compositions after filtering")

        # Predict R
        for idx_in_batch in range(0, len(batch_comps), batch_size):
            chunk = batch_comps[idx_in_batch:idx_in_batch + batch_size]
            frac_chunk = [{k: v / 100.0 for k, v in c.items()} for c in chunk]
            for comp, frac in zip(chunk, frac_chunk):
                try:
                    r_mean, r_std = r_pred.predict(frac)
                    all_candidates.append((r_mean, r_std, comp))
                except Exception:
                    continue

    # ── Rank ──
    all_candidates.sort(key=lambda x: -x[0])
    top_candidates = all_candidates[:top_k]

    results_list = []
    for r_val, r_std, comp in top_candidates:
        comp_str = "".join(f"{s}{f:.1f}" for s, f in
                          sorted(comp.items(), key=lambda x: -x[1]))
        results_list.append({
            "R (µΩ·cm)": round(r_val, 2),
            "R_std": round(r_std, 4),
            "composition": comp,
            "composition_str": comp_str,
        })

    history = {
        "n_sampled": n_samples,
        "n_valid": len(all_candidates),
        "n_returned": len(results_list),
        "lat_evo_steps": lat_evo_steps,
        "time_seconds": round(time.time() - t0, 1),
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"HIGH-R GENERATION RESULTS")
    print(f"{'='*60}")
    print(f"  Samples: {n_samples}  Valid: {len(all_candidates)}")
    print(f"  Time: {history['time_seconds']:.1f}s")
    print(f"\n  Top {min(10, top_k)}:")
    print(f"  {'#':>3s}  {'R (µΩ·cm)':>10s}  {'Composition'}")
    print(f"  {'-'*50}")
    for i, r in enumerate(results_list[:10]):
        print(f"  {i+1:>3d}  {r['R (µΩ·cm)']:>8.2f}  ±{r['R_std']:.2f}  "
              f"{r['composition_str']}")

    # Save
    out_path = _VAE_MODELS.parent / "output" / "r_high_candidates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"candidates": results_list, "history": history}, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return {
        "candidates": results_list,
        "best": results_list[0] if results_list else None,
        "history": history,
    }


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VAE High-R Generator")
    parser.add_argument("--train", action="store_true", help="Train VAE")
    parser.add_argument("--generate", action="store_true", help="Generate high-R")
    parser.add_argument("--n", type=int, default=5000, help="Number of samples")
    parser.add_argument("--top", type=int, default=50, help="Top K to keep")
    parser.add_argument("--evo", type=int, default=0, help="Latent evolution rounds")
    parser.add_argument("--elements", type=str, default=None,
                        help="Comma-separated element symbols, e.g. Ta,Nb,Ni,Co,Hf")
    parser.add_argument("--force", action="store_true", help="Force retrain")
    args = parser.parse_args()

    elems = args.elements.split(",") if args.elements else None

    if args.train:
        train_vae_r(force=args.force)

    if args.generate:
        generate_high_R(n_samples=args.n, top_k=args.top, lat_evo_steps=args.evo,
                        elements=elems)

    if not args.train and not args.generate:
        # Default: train if needed, then generate
        if not (_VAE_MODELS / f"{VAE_R_MODEL_KEY}.pt").exists():
            train_vae_r()
        generate_high_R(n_samples=args.n, top_k=args.top, lat_evo_steps=args.evo,
                        elements=elems)
