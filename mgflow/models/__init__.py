"""③ 模型定义 (model definitions) 子包。

- neural_net.py : 前馈神经网络架构 + 各属性预设配置
- （可选扩展）gpr.py / classifier.py
"""
from .neural_net import PropertyNN, PropertyNN_Small, ARCH_CONFIGS

__all__ = ["PropertyNN", "PropertyNN_Small", "ARCH_CONFIGS"]
