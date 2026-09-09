"""MGflow — 金属玻璃性能预测与反向设计（清晰模块化重构版）。

四大功能模块：
  ① features/        特征构建 —— 从成分计算 25 / 11 维理化特征
  ② data/            数据条目存放 —— 元素属性、训练数据集、查询工具
  ③ prediction/      正向预测 —— PyTorch NN / GPR / MATLAB 桥接 / 集成预测
  ④ inverse_design/  反向设计 —— 生成器探索 + VAE 微调

常用入口：
    from mgflow import compute_all_features, Predictor, EnsemblePredictor
    from mgflow import run_inverse_design, refine_composition
"""
from .features.compute import (
    compute_all_features, FEATURE_NAMES_25, FEATURE_NAMES_11, print_features,
)
from .prediction.predictor import Predictor
from .prediction.ensemble import EnsemblePredictor
from .models.neural_net import PropertyNN, PropertyNN_Small
from .prediction.train_ensemble import train_ensemble_all

from . import inverse_design
from .inverse_design.loop import run_inverse_design
from .inverse_design.enhanced_loop import run_enhanced_inverse_design
from .inverse_design.vae import CompositionVAE, load_vae, train_vae
from .inverse_design.vae_refine import refine_composition, interpolate_compositions

__version__ = "2.0.0"

__all__ = [
    # 特征构建
    "compute_all_features", "FEATURE_NAMES_25", "FEATURE_NAMES_11", "print_features",
    # 正向预测
    "Predictor", "EnsemblePredictor", "PropertyNN", "PropertyNN_Small", "train_ensemble_all",
    # 反向设计
    "run_inverse_design", "run_enhanced_inverse_design",
    "CompositionVAE", "load_vae", "train_vae",
    "refine_composition", "interpolate_compositions",
]
