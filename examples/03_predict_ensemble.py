#!/usr/bin/env python3
"""③ 正向预测示例：集成预测（含不确定性）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mgflow import EnsemblePredictor

comp = {"Zr": 0.5, "Cu": 0.3, "Ni": 0.2}

ep = EnsemblePredictor(n_models=20)
result = ep.predict(comp)

print("Ensemble prediction (mean ± std):")
for prop in EnsemblePredictor.PROPERTIES:
    mean = result.get(prop)
    std = result.get(f"{prop}_std")
    print(f"  {prop:>4s} = {mean} ± {std}")
