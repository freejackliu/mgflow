#!/usr/bin/env python3
"""④ 反向设计示例：探索（Generator）+ 微调（VAE）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mgflow import run_inverse_design, refine_composition

print("=" * 60)
print("④ 反向设计：探索（Generator）")
print("=" * 60)
result = run_inverse_design(
    target="BMG",
    elements=["Zr", "Cu", "Al", "Ni"],
    n_generations=100, batch_size=50, verbose=False,
)
best = result["best"]
if best:
    print(f"  Best: {best['composition_str']}")
    print(f"  Fitness: {best['fitness']}")

print()
print("=" * 60)
print("④ 反向设计：微调（VAE）")
print("=" * 60)
refined = refine_composition(
    seed_compositions=[{"Zr": 0.5, "Cu": 0.3, "Ni": 0.2}],
    n_variants=100, perturbation_scale=0.3, target="Tl_high", verbose=False,
)
if refined["best"]:
    print(f"  Best: {refined['best']['composition_str']}")
    print(f"  Fitness: {refined['best']['fitness']}")
