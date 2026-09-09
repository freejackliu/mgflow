"""② 数据条目存放 (data storage) 子包。

- elemental/ : 元素属性与相互作用矩阵 (.npy)
- datasets/  : 解析好的训练数据集 (.npz，含成分/标签/元数据)
- loader.py  : 数据加载接口
- viewer.py  : 数据查询/浏览工具
"""
from .loader import (
    ELEMENT_SYMBOLS, SYMBOL_TO_NO, composition_vector,
    get_property, get_mixing_enthalpy, get_interaction_energy, get_orthogonalization_matrix,
    load_dataset, list_available_datasets,
)

__all__ = [
    "ELEMENT_SYMBOLS", "SYMBOL_TO_NO", "composition_vector",
    "get_property", "get_mixing_enthalpy", "get_interaction_energy",
    "get_orthogonalization_matrix", "load_dataset", "list_available_datasets",
]
