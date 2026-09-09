#!/usr/bin/env python3
"""训练并保存 BMG / Ribbon 分类器（RandomForest）。

Usage:
    python -m mgflow.prediction.train_classifier

Output:
    mgflow/prediction/artifacts/pytorch_models/classifier_bmg_rf.pkl
"""
import pickle
from pathlib import Path

import numpy as np

from ..data.loader import ELEMENT_SYMBOLS, load_dataset
from ..features.compute import compute_all_features

NO_TO_SYMBOL = {i: s for i, s in enumerate(ELEMENT_SYMBOLS)}
_OUT_DIR = Path(__file__).resolve().parent / "artifacts" / "pytorch_models"


def prepare_data():
    """收集特征与 BMG 标签（来自 Tg/Tx/Tl 数据集，按合金名去重）。"""
    X_list, y_list, names_list = [], [], []
    for prop in ["Tg", "Tx", "Tl"]:
        d = load_dataset(prop)  # 默认含元数据 (mg, bmg, ribbon, D, base_element)
        for i, name in enumerate(d["names"]):
            if d["bmg"][i] == -1:          # 无 BMG 标签，跳过
                continue
            if name in names_list:         # 跨属性去重
                continue
            names_list.append(name)

            nos = d["elem_nos"][i]
            fracs = d["fracs"][i]
            comp = {NO_TO_SYMBOL.get(int(n)): float(f)
                    for n, f in zip(nos, fracs) if f > 0}
            try:
                feat = compute_all_features(comp)
            except Exception:
                continue
            X_list.append(feat)
            y_list.append(int(d["bmg"][i]))

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  Samples: {len(X)}  BMG=1:{np.sum(y == 1)}  Ribbon:{np.sum(y == 0)}")
    return X, y, names_list


def train_and_save():
    """训练 RandomForest 分类器并保存到磁盘。"""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing training data...")
    X, y, _ = prepare_data()

    print("Training RandomForest classifier...")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    path = _OUT_DIR / "classifier_bmg_rf.pkl"
    with open(path, "wb") as f:
        pickle.dump(rf, f)
    print(f"  Saved to {path}")

    # 快速交叉验证评估
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    acc = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    print(f"  5-fold CV: AUC={auc.mean():.4f}±{auc.std():.4f}  Acc={acc.mean():.4f}±{acc.std():.4f}")


if __name__ == "__main__":
    train_and_save()
