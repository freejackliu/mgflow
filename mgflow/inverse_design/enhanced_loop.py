"""Enhanced inverse-design loop with ranking-based selection and diversity-aware training.

Provides drop-in replacements for run_inverse_design() with:
  - Ranking-based fitness (α × rank(预测值) + β × rank(不确定性))
  - Diversity reward (pairwise distance penalty in generator loss)
  - Experience replay (maintains high-fitness candidate buffer)

All original interfaces in loop.py remain unchanged.
"""

import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    DEFAULT_ELEMENTS, LARGE_ELEMENT_POOL, DEFAULT_MODE,
    LATENT_DIM, GEN_HIDDEN_DIMS, GEN_DROPOUT, GEN_LEAKY_SLOPE,
    DEFAULT_N_GENERATIONS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_BETA,
    DEFAULT_KEEP_TOP_K, DEFAULT_THRESHOLD, DEFAULT_SAVE_TOP_N,
    ALL_TARGETS,
)
from .objectives import scalarize, summarize_objective_uncertainty
from .net import Generator
from .extract import extract_composition_batch, composition_to_dict, format_composition
from .loop import _is_valid_composition  # reuse basic validation


# ===================================================================
# Ranking-based fitness functions
# ===================================================================

def compute_rank_fitness(props: dict, target_cfg: dict,
                         uncertainty: dict = None,
                         alpha: float = 0.9, beta: float = 0.1) -> float:
    """Compute fitness via ranking-based scoring (Rabbe 2022 style).

    Score = α × rank(预测值) + β × rank(不确定性)

    Parameters
    ----------
    props : dict
        Prediction result from Predictor (Tg, Tx, Tl, E, H, BMG_prob, …).
    target_cfg : dict
        Target configuration: {"mode": "maximize"/"minimize", "key": property_key}.
    uncertainty : dict, optional
        Uncertainty dict with keys like "Tg_std", "Tl_std", etc.
        If None, defaults to beta=0 (pure exploitation).
    alpha : float
        Weight for exploitation (prediction rank). Default 0.9.
    beta : float
        Weight for exploration (uncertainty rank). Default 0.1.

    Returns
    -------
    score : float
        Higher is better.
    components : dict
        {"value_rank": ..., "uncertainty_rank": ..., "raw_value": ..., "raw_uncertainty": ...}
    """
    # ── 多目标：标量化得到 raw_value，不确定性取涉及属性的均值 ──
    if "objectives" in target_cfg:
        objectives = target_cfg["objectives"]
        val, _ = scalarize(
            props, objectives, normalize=target_cfg.get("normalize", "minmax"),
        )
        unc = summarize_objective_uncertainty(objectives, uncertainty)
        components = {
            "raw_value": float(val),
            "raw_uncertainty": float(unc),
            "alpha": alpha,
            "beta": beta,
        }
        return float(val), components

    key = target_cfg["key"]

    # Get raw value
    if key == "Tx_minus_Tg":
        val = props.get("Tx", 0) - props.get("Tg", 0)
    elif key == "BMG_prob":
        val = props.get("BMG_prob", 0) or 0.0
    else:
        val = props.get(key, 0) or 0.0

    mode = target_cfg["mode"]
    if mode == "minimize":
        val = -val  # flip so higher is better

    # Get uncertainty for the target property
    unc = None
    if uncertainty is not None:
        # Try to find matching uncertainty key
        unc_key = f"{key}_std"
        if unc_key in uncertainty and uncertainty[unc_key] is not None:
            unc = uncertainty[unc_key]
        # Fallback: mean of all available uncertainties
        if unc is None:
            unc_vals = []
            for k, v in uncertainty.items():
                if k.endswith("_std") and v is not None:
                    unc_vals.append(v)
            unc = float(np.mean(unc_vals)) if unc_vals else 0.0

    # NOTE: actual ranking happens after all candidates are evaluated.
    # Here we just return raw values; ranking is applied in _select_by_rank().
    components = {
        "raw_value": float(val),
        "raw_uncertainty": float(unc) if unc is not None else 0.0,
        "alpha": alpha,
        "beta": beta,
    }
    return float(val), components


def rank_transform(candidates: list, alpha: float = 0.9, beta: float = 0.1):
    """Convert raw values+uncertainties to ranking scores.

    Parameters
    ----------
    candidates : list of dict
        Each dict has "raw_value", "raw_uncertainty".
    alpha, beta : float
        Ranking weights.

    Returns
    -------
    scores : np.ndarray
        Combined ranking scores (higher = better).
    """
    n = len(candidates)
    if n == 0:
        return np.array([])

    values = np.array([c["raw_value"] for c in candidates])
    uncertainties = np.array([c["raw_uncertainty"] for c in candidates])

    # Rank 1 = best (highest value), rank n = worst
    # For values (higher is better): rank 1 = best
    value_ranks = np.argsort(np.argsort(-values)) + 1  # 1..n

    # For uncertainty (higher = more uncertain = more exploration value)
    # Higher uncertainty = better rank for exploration
    unc_ranks = np.argsort(np.argsort(-uncertainties)) + 1  # 1..n

    # Normalized ranks (0 to 1, lower = better)
    norm_value_ranks = value_ranks / n
    norm_unc_ranks = unc_ranks / n

    # Combined: lower rank = higher score
    # Score = -(α * norm_value_rank + β * norm_unc_rank)
    # So lower combined rank = higher score
    combined = -(alpha * norm_value_ranks + beta * norm_unc_ranks)

    return combined


# ===================================================================
# Experience replay buffer
# ===================================================================

class ExperienceReplay:
    """Buffer that stores high-fitness composition candidates.

    Used to provide diverse training targets for the generator,
    preventing mode collapse.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = []  # list of (fitness, composition_dict)

    def add(self, fitness: float, comp_dict: dict):
        """Add a candidate to the buffer."""
        self.buffer.append((fitness, comp_dict.copy()))
        # Sort by fitness descending and trim
        self.buffer.sort(key=lambda x: x[0], reverse=True)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[:self.capacity]

    def add_batch(self, candidates: list):
        """Add multiple candidates: list of (fitness, comp_dict)."""
        for fitness, comp_dict in candidates:
            self.add(fitness, comp_dict)

    def sample(self, k: int, elements: list, rng: np.random.Generator = None):
        """Sample k compositions as target vectors.

        Returns
        -------
        target_vectors : np.ndarray, shape (k, n_ele)
        """
        if rng is None:
            rng = np.random.default_rng()
        if len(self.buffer) == 0:
            return None

        # Weighted sampling: higher fitness = higher probability
        fitnesses = np.array([f for f, _ in self.buffer])
        # Softmax weighting for stable sampling
        weights = np.exp(fitnesses - fitnesses.max())
        weights = weights / weights.sum()

        n_ele = len(elements)
        n_sample = min(k, len(self.buffer))
        indices = rng.choice(len(self.buffer), size=n_sample, replace=False, p=weights)

        target_vectors = np.zeros((n_sample, n_ele), dtype=np.float32)
        for i, idx in enumerate(indices):
            _, comp_dict = self.buffer[idx]
            for j, el in enumerate(elements):
                target_vectors[i, j] = comp_dict.get(el, 0.0) / 100.0
            # Normalize
            s = target_vectors[i].sum()
            if s > 0:
                target_vectors[i] /= s

        return target_vectors

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"ExperienceReplay(capacity={self.capacity}, size={len(self.buffer)})"


# ===================================================================
# Diversity-aware generator loss
# ===================================================================

def diversity_penalty_loss(output: torch.Tensor, target: torch.Tensor,
                           lambda_div: float = 0.1) -> tuple:
    """Compute MSE loss with diversity reward.

    The diversity term penalizes the generator if different outputs
    in the batch are too similar (pairwise cosine similarity).

    total_loss = MSE(output, target) - lambda_div * diversity_reward

    where diversity_reward = mean pairwise distance between outputs.

    Parameters
    ----------
    output : torch.Tensor, shape (batch, n_ele)
        Generator output scores.
    target : torch.Tensor, shape (batch, n_ele)
        Target composition vectors.
    lambda_div : float
        Weight for diversity reward.

    Returns
    -------
    total_loss : torch.Tensor
    components : dict of float
    """
    mse_loss = nn.MSELoss()(output, target)

    # Pairwise cosine distance
    norm_out = output / (output.norm(dim=1, keepdim=True) + 1e-8)
    similarity = norm_out @ norm_out.T  # (batch, batch)
    # Exclude self-similarity
    batch_size = output.shape[0]
    mask = 1 - torch.eye(batch_size, device=output.device)
    pair_distances = (1 - similarity) * mask
    n_pairs = mask.sum()
    diversity = pair_distances.sum() / n_pairs if n_pairs > 0 else torch.tensor(0.0)

    total_loss = mse_loss - lambda_div * diversity

    return total_loss, {
        "mse_loss": mse_loss.item(),
        "diversity_reward": diversity.item(),
        "total_loss": total_loss.item(),
    }


# ===================================================================
# Enhanced inverse design loop
# ===================================================================

def run_enhanced_inverse_design(
    target: str = "BMG",
    elements: list = None,
    mode: str = None,
    n_generations: int = DEFAULT_N_GENERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    keep_top_k: int = DEFAULT_KEEP_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    save_top_n: int = DEFAULT_SAVE_TOP_N,
    device: str = None,
    verbose: bool = True,
    output_dir: str = None,
    # Enhanced features
    use_ensemble: bool = True,
    n_ensemble_models: int = 20,
    ranking_alpha: float = 0.9,
    ranking_beta: float = 0.1,
    lambda_diversity: float = 0.1,
    replay_capacity: int = 2000,
    replay_sample_k: int = None,
):
    """Enhanced inverse design loop with ranking selection, diversity, and experience replay.

    Parameters
    ----------
    (Same as run_inverse_design in loop.py, plus:)

    use_ensemble : bool
        Use EnsemblePredictor instead of single Predictor (default: True).
    n_ensemble_models : int
        Number of models in ensemble (default: 20).
    ranking_alpha : float
        Exploitation weight (0-1) in ranking (default: 0.9).
    ranking_beta : float
        Exploration weight (0-1) in ranking (default: 0.1).
    lambda_diversity : float
        Diversity reward weight in generator loss (default: 0.1).
    replay_capacity : int
        Max size of experience replay buffer (default: 2000).
    replay_sample_k : int, optional
        Number of candidates to sample from replay buffer each generation.
        Defaults to keep_top_k.

    Returns
    -------
    result : dict
        {"candidates": [...], "best": {...}, "history": [...]}
    """
    # ── Parse target ──
    if isinstance(target, str):
        if target not in ALL_TARGETS:
            raise ValueError(f"Unknown target '{target}'. Options: {list(ALL_TARGETS.keys())}")
        target_cfg = dict(ALL_TARGETS[target])
    else:
        target_cfg = dict(target)

    # Default replay_sample_k
    if replay_sample_k is None:
        replay_sample_k = keep_top_k

    # ── Mode & element set (same as loop.py) ──
    if mode is None:
        mode = DEFAULT_MODE
    if mode == "auto":
        pool = list(LARGE_ELEMENT_POOL)
        if elements is not None:
            elements = [el for el in elements if el in pool]
            if not elements:
                raise ValueError("No valid elements found in the large pool.")
        else:
            elements = pool
    else:
        if elements is None:
            elements = list(DEFAULT_ELEMENTS)
    elements = list(elements)
    n_ele = len(elements)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if mode == "auto":
        if n_ele <= 10:
            max_ele_per_comp = n_ele
        else:
            max_ele_per_comp = min(max(6, n_ele // 5), 10)
        if verbose:
            print(f"  Auto mode: max {max_ele_per_comp} elements per composition (pool={n_ele} elements)")
    else:
        max_ele_per_comp = None

    # ── Output dir ──
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build generator ──
    gen = Generator(
        n_elements=n_ele,
        latent_dim=LATENT_DIM,
        hidden_dims=GEN_HIDDEN_DIMS,
        dropout=GEN_DROPOUT,
        leaky_slope=GEN_LEAKY_SLOPE,
    ).to(device)
    gen.train()

    optimizer = optim.Adam(gen.parameters(), lr=lr, betas=DEFAULT_BETA)

    # ── Oracle: use EnsemblePredictor or single Predictor ──
    from ..features.compute import compute_all_features
    if use_ensemble:
        from ..prediction.ensemble import EnsemblePredictor
        predictor = EnsemblePredictor(n_models=n_ensemble_models, device=device)
    else:
        from ..prediction.predictor import Predictor
        predictor = Predictor("pytorch")

    # ── Experience replay buffer ──
    replay = ExperienceReplay(capacity=replay_capacity)

    # ── Tracking ──
    all_candidates = []          # (fitness, generation, comp_dict, props)
    history = []                 # per-generation stats

    start_time = time.time()
    rng = np.random.default_rng()

    # =================================================================
    # Main loop
    # =================================================================
    for gen_idx in range(n_generations):
        # 1) Generate
        z = torch.empty(batch_size, LATENT_DIM, device=device).uniform_(-1, 1)
        with torch.no_grad():
            scores = gen(z).cpu().numpy()

        # 2) Extract compositions
        comps = extract_composition_batch(
            scores, elements, threshold=threshold, max_elements=max_ele_per_comp,
        )

        # 3) Evaluate via oracle
        gen_candidates = []  # list of dicts for ranking
        gen_fitnesses = []
        valid_comps = []

        for comp_dict in comps:
            if not _is_valid_composition(comp_dict):
                continue
            try:
                frac_dict = composition_to_dict(comp_dict)
                props = predictor.predict(frac_dict)
            except Exception:
                continue

            # Extract uncertainty if available
            uncertainty = None
            if use_ensemble:
                uncertainty = {k: props.get(k) for k in props if k.endswith("_std")}

            # Compute raw fitness + ranking components
            raw_fit, rank_components = compute_rank_fitness(
                props, target_cfg, uncertainty=uncertainty,
                alpha=ranking_alpha, beta=ranking_beta,
            )

            gen_candidates.append({
                "comp_dict": comp_dict,
                "props": props,
                "raw_value": rank_components["raw_value"],
                "raw_uncertainty": rank_components["raw_uncertainty"],
                "raw_fitness": raw_fit,
            })
            gen_fitnesses.append((raw_fit, gen_idx, comp_dict, props))
            valid_comps.append(comp_dict)

        # 4) Apply ranking transform (if uncertainties available)
        if use_ensemble and len(gen_candidates) > 1:
            rank_scores = rank_transform(
                gen_candidates, alpha=ranking_alpha, beta=ranking_beta,
            )
            # Assign ranked fitness
            for i, score in enumerate(rank_scores):
                gen_fitnesses[i] = (score, gen_idx,
                                    gen_candidates[i]["comp_dict"],
                                    gen_candidates[i]["props"])

        # 5) Sort by fitness descending
        gen_fitnesses.sort(key=lambda x: x[0], reverse=True)

        # 6) Update experience replay buffer
        replay.add_batch([(f, c) for f, _, c, _ in gen_fitnesses[:keep_top_k]])

        # 7) Track all candidates
        all_candidates.extend(gen_fitnesses[:keep_top_k])

        # 8) Generator update: diversity-aware + experience replay
        targets_for_training = []

        # 8a) Top-k from current generation
        for fit, _, comp_dict, _ in gen_fitnesses[:keep_top_k]:
            target_vec = np.zeros(n_ele, dtype=np.float32)
            for i, el in enumerate(elements):
                target_vec[i] = comp_dict.get(el, 0.0) / 100.0
            if target_vec.sum() > 0:
                target_vec = target_vec / target_vec.sum()
            targets_for_training.append(target_vec)

        # 8b) Experience replay samples
        replay_targets = replay.sample(replay_sample_k, elements, rng)
        if replay_targets is not None:
            targets_for_training.extend([t for t in replay_targets])

        # 8c) Train generator
        if targets_for_training:
            target_tensor = torch.tensor(np.stack(targets_for_training), device=device)
            n_train = len(target_tensor)

            z_train = torch.empty(n_train, LATENT_DIM, device=device).uniform_(-1, 1)
            gen.train()
            optimizer.zero_grad()
            output = gen(z_train)

            # Diversity-aware loss
            loss, loss_components = diversity_penalty_loss(
                output, target_tensor, lambda_div=lambda_diversity,
            )
            loss.backward()
            optimizer.step()
            gen_loss_val = loss_components["total_loss"]
        else:
            gen_loss_val = 0.0

        # 9) Record history
        if gen_fitnesses:
            best_fit = gen_fitnesses[0][0]
            mean_fit = np.mean([f[0] for f in gen_fitnesses]).item()
            n_valid = len(valid_comps)
        else:
            best_fit = -np.inf
            mean_fit = -np.inf
            n_valid = 0

        history.append({
            "generation": gen_idx,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "n_valid": n_valid,
            "generator_loss": gen_loss_val,
            "replay_size": len(replay),
            "ranking_alpha": ranking_alpha,
            "ranking_beta": ranking_beta,
            "lambda_diversity": lambda_diversity,
        })

        # 10) Verbose progress
        if verbose and (gen_idx % 50 == 0 or gen_idx == n_generations - 1):
            elapsed = time.time() - start_time
            print(
                f"[{gen_idx:4d}/{n_generations}] "
                f"best={best_fit:+.4f}  mean={mean_fit:+.4f}  "
                f"valid={n_valid}/{batch_size}  "
                f"gen_loss={gen_loss_val:.6f}  "
                f"replay={len(replay)}  "
                f"time={elapsed:.0f}s"
            )

            if gen_fitnesses:
                _, _, best_comp, best_props = gen_fitnesses[0]
                comp_str = format_composition(best_comp)
                print(f"  Best: {comp_str}")
                for pk in ["Tg", "Tx", "Tl", "E", "H"]:
                    pv = best_props.get(pk)
                    if pv is not None:
                        print(f"    {pk}: {pv:.1f}", end=" ")
                bmg_p = best_props.get("BMG_prob")
                if bmg_p is not None:
                    print(f"BMG_prob: {bmg_p:.3f}", end="")
                # Show uncertainty if available
                if use_ensemble:
                    print("  | ", end="")
                    for pk in ["Tg", "Tx", "Tl"]:
                        pv_std = best_props.get(f"{pk}_std")
                        if pv_std is not None:
                            print(f"σ({pk})={pv_std:.2f}", end=" ")
                print()

    # ── Finalise ──
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    top_n = all_candidates[:save_top_n]

    results_list = []
    for fit, gen_idx, comp_dict, props in top_n:
        entry = {
            "fitness": round(fit, 4),
            "generation": gen_idx,
            "composition": comp_dict,
            "composition_str": format_composition(comp_dict),
            "properties": {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in props.items()
                if v is not None and not k.endswith("_preds")
            } if props else {},
        }
        results_list.append(entry)

    # Save to JSON
    output_path = output_dir / f"enhanced_inverse_design_{target}.json"
    with open(output_path, "w") as f:
        json.dump({
            "target": target,
            "mode": mode,
            "elements": elements,
            "config": {
                "n_generations": n_generations,
                "batch_size": batch_size,
                "threshold": threshold,
                "use_ensemble": use_ensemble,
                "n_ensemble_models": n_ensemble_models,
                "ranking_alpha": ranking_alpha,
                "ranking_beta": ranking_beta,
                "lambda_diversity": lambda_diversity,
                "replay_capacity": replay_capacity,
                "replay_sample_k": replay_sample_k,
            },
            "total_time_seconds": round(time.time() - start_time, 1),
            "top_candidates": results_list,
        }, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nDone in {time.time() - start_time:.0f}s. Top {save_top_n} saved to {output_path}")

    return {
        "candidates": results_list,
        "best": results_list[0] if results_list else None,
        "history": history,
    }
