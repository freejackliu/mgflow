#!/usr/bin/env python3
"""① 特征构建示例：从成分计算 25 / 11 维特征。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mgflow import compute_all_features, print_features, FEATURE_NAMES_25

comp = {"Zr": 0.5, "Cu": 0.3, "Ni": 0.2}

print("=" * 60)
print("① 特征构建（25 维）")
print("=" * 60)
feat = print_features(comp)
print(f"\n25-feature vector shape: {feat.shape}")

print("\n" + "=" * 60)
print("① 特征构建（11 维，不含 physics）")
print("=" * 60)
feat11 = compute_all_features(comp, include_physics=False)
print(f"11-feature vector: {feat11}")
