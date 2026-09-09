"""
Variational Autoencoder (VAE) for composition generation.

Improvements over v1 (2026-07-08):
  - Softmax output (naturally sum-to-1)
  - KL-divergence reconstruction loss (better for probability vectors)
  - β-annealing: β ramps from 0 → target over first N epochs
  - Lower default β to prevent posterior collapse
"""

import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..data.loader import load_dataset, SYMBOL_TO_NO, ELEMENT_SYMBOLS


# ===================================================================
# Universal element ordering
# ===================================================================
def _build_vae_elements():
    all_sets = set()
    for p in ["Tg", "Tx", "Tl", "E", "H"]:
        data = load_dataset(p, include_metadata=False)
        for row in data["elem_nos"]:
            for e in row:
                if e > 0:
                    all_sets.add(int(e))
    return sorted(all_sets)

VAE_ELEMENTS = _build_vae_elements()
VAE_SYMBOLS = [ELEMENT_SYMBOLS[e] for e in VAE_ELEMENTS]
N_ELEMENTS = len(VAE_ELEMENTS)
ELEM_TO_VAE_IDX = {e: i for i, e in enumerate(VAE_ELEMENTS)}
print(f"[vae] {N_ELEMENTS} elements: {VAE_SYMBOLS}")


# ===================================================================
# Dataset
# ===================================================================
class CompositionDataset(Dataset):
    def __init__(self):
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
        self.vectors = np.array(vectors, dtype=np.float32)
        print(f"[CompositionDataset] {len(self.vectors)} samples, {N_ELEMENTS} dims")

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]


# ===================================================================
# VAE
# ===================================================================
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
        # NO activation — we take log_softmax in loss, softmax for generation
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

    def interpolate(self, z1, z2, steps=10):
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        zs = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
        with torch.no_grad():
            return torch.softmax(self.decoder(zs), dim=-1).cpu().numpy()

    def encode_to_z(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            mu, logvar = self.encode(x)
        return mu.cpu().numpy(), torch.exp(0.5 * logvar).cpu().numpy()

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return (f"CompositionVAE(n_elements={self.n_elements}, "
                f"latent_dim={self.latent_dim}, "
                f"hidden_dims={self.hidden_dims}, "
                f"params={sum(p.numel() for p in self.parameters())})")


# ===================================================================
# Loss with β-annealing
# ===================================================================
def vae_loss(logits, x, mu, logvar, beta, kl_anneal_weight=1.0):
    """ELBO: KL(recon ‖ target) + β * weight * KL(z ‖ prior).

    Parameters
    ----------
    logits : (batch, n_ele) — decoder logits
    x : (batch, n_ele) — target fractions (sums to 1)
    mu, logvar : (batch, latent_dim)
    beta : float — target KL weight
    kl_anneal_weight : float — 0→1 ramp-up factor

    Returns
    -------
    total, components
    """
    # Reconstruction: KL(target ‖ softmax(logits))
    # = cross_entropy(logits, target) - entropy(target)
    # = -sum(x * log(softmax(logits))) + sum(x * log(x))
    log_probs = F.log_softmax(logits, dim=-1)
    recon_loss = -torch.sum(x * log_probs, dim=-1).mean()

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    effective_beta = beta * kl_anneal_weight
    total = recon_loss + effective_beta * kl_loss

    return total, {
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "effective_beta": float(effective_beta),
        "total_loss": total.item(),
    }


# ===================================================================
# Training
# ===================================================================
def train_vae(latent_dim=32, hidden_dims=None, beta=0.01, epochs=300,
              batch_size=128, lr=1e-3, weight_decay=1e-5,
              kl_anneal_epochs=100, device=None, save_dir=None, verbose=True):
    """Train VAE with KL annealing.

    Parameters
    ----------
    beta : float — target KL weight. Start low (0.001~0.01) to avoid collapse.
    kl_anneal_epochs : int — ramp β from 0→beta over this many epochs.
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset = CompositionDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    vae = CompositionVAE(n_elements=N_ELEMENTS, latent_dim=latent_dim,
                         hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, min_lr=1e-6)

    t0 = time.time()
    history = {"recon_loss": [], "kl_loss": [], "total_loss": []}
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        vae.train()
        losses = {"recon_loss": [], "kl_loss": [], "total_loss": []}

        # KL annealing weight: linear ramp 0→1 over kl_anneal_epochs
        kl_weight = min(1.0, (epoch + 1) / kl_anneal_epochs) if kl_anneal_epochs > 0 else 1.0

        for batch in loader:
            x = batch.to(device)
            logits, mu, logvar = vae(x)
            loss, comps = vae_loss(logits, x, mu, logvar, beta=beta,
                                   kl_anneal_weight=kl_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            for k, v in comps.items():
                if k in losses:
                    losses[k].append(v)

        avg = {k: np.mean(v) for k, v in losses.items()}
        for k in history:
            history[k].append(avg[k])

        scheduler.step(avg["total_loss"])

        if avg["total_loss"] < best_loss:
            best_loss = avg["total_loss"]
            best_state = {k: v.cpu().clone() for k, v in vae.state_dict().items()}

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            el = time.time() - t0
            print(f"[{epoch+1:3d}/{epochs}] "
                  f"recon={avg['recon_loss']:.6f}  "
                  f"kl={avg['kl_loss']:.6f}  "
                  f"β_eff={comps['effective_beta']:.6f}  "
                  f"total={avg['total_loss']:.6f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"time={el:.0f}s")

    vae.load_state_dict(best_state)
    vae.eval()

    # Save
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "vae_models"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"vae_lat{latent_dim}_b{beta}.pt"
    torch.save(vae.state_dict(), model_path)
    config = {"n_elements": N_ELEMENTS, "element_symbols": VAE_SYMBOLS,
              "latent_dim": latent_dim, "hidden_dims": hidden_dims, "beta": beta,
              "epochs": epochs, "final_recon_loss": history["recon_loss"][-1],
              "final_kl_loss": history["kl_loss"][-1],
              "n_training_samples": len(dataset)}
    with open(model_path.with_suffix(".json"), "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"\nDone. Model saved to {model_path}")
        print(f"Final: recon={history['recon_loss'][-1]:.6f}  kl={history['kl_loss'][-1]:.6f}")
    return vae


def load_vae(latent_dim=32, beta=0.01, model_dir=None, device=None):
    if model_dir is None:
        model_dir = Path(__file__).resolve().parent / "vae_models"
    model_dir = Path(model_dir)
    model_path = model_dir / f"vae_lat{latent_dim}_b{beta}.pt"
    config_path = model_path.with_suffix(".json")

    if not model_path.exists():
        raise FileNotFoundError(f"VAE not found: {model_path}")

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        hidden_dims = cfg.get("hidden_dims", [256, 128, 64])
    else:
        hidden_dims = [256, 128, 64]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    vae = CompositionVAE(n_elements=N_ELEMENTS, latent_dim=latent_dim,
                         hidden_dims=hidden_dims)
    vae.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vae.to(device)
    vae.eval()
    return vae
