#!/usr/bin/env python3
"""③ 正向预测示例：PyTorch NN 单模型预测。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mgflow import Predictor

comp = {"Zr": 0.5, "Cu": 0.3, "Ni": 0.2}

pred = Predictor(mode="pytorch")
result = pred.predict(comp, verbose=False)

print("Predicted properties:")
for prop in Predictor.PROPERTIES:
    unit = Predictor.PROP_UNITS[prop]
    val = result.get(prop)
    print(f"  {prop:>4s} = {val if val is None else round(val, 2):>8} {unit}")
print(f"  BMG_prob = {result.get('BMG_prob')}")
