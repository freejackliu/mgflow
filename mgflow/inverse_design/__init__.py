"""④ 反向设计 (inverse design) 子包。

两种互补策略：
  - 探索 (Explore)：Generator 网络自由生成成分（loop.py / enhanced_loop.py）
  - 微调 (Refine) ：VAE 在已知成分流形上局部变分（vae.py / vae_refine.py）
"""
from .net import Generator
from .extract import extract_composition, extract_composition_batch, format_composition
from .loop import run_inverse_design
from .enhanced_loop import run_enhanced_inverse_design, ExperienceReplay, rank_transform
from .vae import CompositionVAE, load_vae, train_vae, VAE_SYMBOLS
from .vae_refine import refine_composition, interpolate_compositions
from .objectives import scalarize, summarize_objective_uncertainty
from .config import TARGETS, MULTI_TARGETS, ALL_TARGETS
from . import config

__all__ = [
    # 探索
    "Generator", "run_inverse_design", "run_enhanced_inverse_design",
    "ExperienceReplay", "rank_transform",
    # 微调
    "CompositionVAE", "load_vae", "train_vae", "VAE_SYMBOLS",
    "refine_composition", "interpolate_compositions",
    # 目标（单目标 + 多目标）
    "TARGETS", "MULTI_TARGETS", "ALL_TARGETS",
    "scalarize", "summarize_objective_uncertainty",
    # 公共
    "extract_composition", "extract_composition_batch", "format_composition",
    "config",
]
