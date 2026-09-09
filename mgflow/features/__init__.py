"""① 特征构建 (feature construction) 子包。

从合金成分出发，计算 25 维（或 11 维）物理化学特征：
  - dn_dc (6)  : 原子尺度应变/刚度失配
  - ye (2)     : RMS 原子应变、存储弹性能
  - scorr (1)  : 混合熵 + 弹性/化学涨落修正
  - hu (2)     : 内聚能偏差、相互作用能比
  - physics (14): r / Tm / EN / VEC / Bulk 的平均与偏差、混合焓、Hessian 特征值
"""
from .compute import compute_all_features, FEATURE_NAMES_25, FEATURE_NAMES_11, print_features
from . import dn_dc, ye, scorr, hu, physics

__all__ = [
    "compute_all_features", "FEATURE_NAMES_25", "FEATURE_NAMES_11", "print_features",
    "dn_dc", "ye", "scorr", "hu", "physics",
]
