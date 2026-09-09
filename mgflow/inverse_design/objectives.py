"""多目标 (multi-objective) 解析与标量化工具。

在反向设计里同时优化多个属性（加权求和标量化），单目标是其特例。

目标配置格式（target dict 或 config.py 中 MULTI_TARGETS 的预设）：

    {
        "objectives": [
            {"key": "Tg", "mode": "maximize", "weight": 1.0},
            {"key": "Tx", "mode": "maximize", "weight": 1.0},
            {"key": "Tl", "mode": "minimize", "weight": 0.5},
            {"key": "E",  "mode": "target",    "target_value": 100.0},
        ],
        "normalize": "minmax",   # "minmax"（推荐）或 "none"
    }

- key   : Tg / Tx / Tl / E / H / BMG_prob / Tx_minus_Tg（过冷液相区 ΔT = Tx − Tg）
- mode  : maximize / minimize / target
- weight: 该目标的权重（默认 1.0），最终 fitness 为加权平均
- normalize: "minmax" 用训练数据各属性的 min/max 归一到 [0,1] 再加权；
             "none" 直接对原始值加权（仅当各目标量纲一致时使用）

对 mode="target"，效用 = 1 − |value − target_value| / 属性范围（截断到 [0,1]）。
"""

import numpy as np

from ..data.loader import load_dataset


# 属性归一化范围（惰性加载，取自训练数据 min/max）
_RANGES = None


def get_property_ranges():
    """返回各目标属性在训练数据中的 (min, max)，用于 min-max 归一化。"""
    global _RANGES
    if _RANGES is None:
        r = {}
        for prop in ["Tg", "Tx", "Tl", "E", "H"]:
            d = load_dataset(prop, include_metadata=False)
            labels = d["labels"]
            r[prop] = (float(labels.min()), float(labels.max()))
        r["BMG_prob"] = (0.0, 1.0)
        # 过冷液相区 ΔT = Tx − Tg 的粗略范围
        r["Tx_minus_Tg"] = (
            r["Tx"][0] - r["Tg"][1],
            r["Tx"][1] - r["Tg"][0],
        )
        _RANGES = r
    return _RANGES


def extract_objective_value(props: dict, key: str) -> float:
    """从预测结果中取出目标属性的原始值。"""
    if key == "Tx_minus_Tg":
        return float(props.get("Tx", 0.0) or 0.0) - float(props.get("Tg", 0.0) or 0.0)
    if key == "BMG_prob":
        return float(props.get("BMG_prob", 0.0) or 0.0)
    return float(props.get(key, 0.0) or 0.0)


def _normalize(value, lo, hi, mode, target_value):
    """把原始值映射到 [0,1] 的"效用"（越大越好）。"""
    span = hi - lo
    if span <= 0:
        return 0.5
    if mode == "maximize":
        return float(np.clip((value - lo) / span, 0.0, 1.0))
    if mode == "minimize":
        return float(np.clip((hi - value) / span, 0.0, 1.0))
    if mode == "target":
        dist = abs(value - (target_value if target_value is not None else (lo + hi) / 2))
        return float(np.clip(1.0 - dist / span, 0.0, 1.0))
    raise ValueError(f"Unknown mode: {mode}")


def scalarize(props: dict, objectives: list, normalize: str = "minmax"):
    """多目标标量化：加权平均各目标归一化后的效用。

    Parameters
    ----------
    props : dict — 预测结果
    objectives : list of dict — 目标列表
    normalize : str — "minmax" 或 "none"

    Returns
    -------
    fitness : float（越大越好）
    components : dict — 各目标的原始值 / 缩放后效用 / 权重，便于调试
    """
    total = 0.0
    total_weight = 0.0
    components = {}
    ranges = get_property_ranges() if normalize == "minmax" else None

    for obj in objectives:
        key = obj["key"]
        mode = obj.get("mode", "maximize")
        weight = float(obj.get("weight", 1.0))
        val = extract_objective_value(props, key)

        if normalize == "minmax" and ranges is not None:
            lo, hi = ranges[key]
            s = _normalize(val, lo, hi, mode, obj.get("target_value"))
        else:
            # 不归一化：minimize 取负，target 取负距离
            if mode == "minimize":
                s = -val
            elif mode == "target":
                s = -abs(val - obj.get("target_value", 0.0))
            else:
                s = val

        total += weight * s
        total_weight += weight
        components[key] = {"raw": val, "mode": mode, "weight": weight, "scaled": s}

    if total_weight <= 0:
        return 0.0, components
    # 加权平均，避免目标数量改变 fitness 量级
    return float(total / total_weight), components


def summarize_objective_uncertainty(objectives: list, uncertainty: dict) -> float:
    """汇总多目标涉及属性的不确定性（取平均，复合 key 用误差传播近似）。"""
    vals = []
    for obj in objectives:
        key = obj["key"]
        std = None
        if key in ("Tg", "Tx", "Tl", "E", "H"):
            std = uncertainty.get(f"{key}_std") if uncertainty else None
        elif key == "Tx_minus_Tg" and uncertainty is not None:
            tx = uncertainty.get("Tx_std")
            tg = uncertainty.get("Tg_std")
            if tx is not None and tg is not None:
                std = float(np.sqrt(tx ** 2 + tg ** 2))
            elif tx is not None:
                std = float(tx)
            elif tg is not None:
                std = float(tg)
        # BMG_prob 无集成不确定性 → 跳过
        if std is not None:
            vals.append(float(std))
    return float(np.mean(vals)) if vals else 0.0
