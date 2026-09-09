# MGflow — 金属玻璃性能预测与反向设计（重构版）

从**合金成分**出发，完成四大任务：**特征构建 → 数据存储 → 正向预测 → 反向设计**。
本仓库是原 `MGflow` 的清晰模块化重构，代码按功能划分到四个子包中。

## 目录结构

```
MGflow_refactored/
├── README.md
├── setup.py
├── examples/                      # 可运行示例（按四大功能组织）
└── mgflow/
    ├── __init__.py                # 统一入口，导出公共 API
    │
    ├── features/                  # ① 特征构建
    │   ├── compute.py             #   特征聚合入口（25 / 11 维）
    │   ├── dn_dc.py               #   6 个 dn/dc 特征（原子尺度应变/刚度失配）
    │   ├── ye.py                  #   2 个 Ye 特征（RMS 应变、弹性能）
    │   ├── scorr.py               #   1 个 Scorr 特征（混合熵 + 涨落修正）
    │   ├── hu.py                  #   2 个 Hu 特征（内聚能偏差、相互作用能比）
    │   └── physics.py             #   14 个理化特征（r/Tm/EN/VEC/Bulk/混合焓/Hessian）
    │
    ├── data/                      # ② 数据条目存放
    │   ├── loader.py              #   元素属性 & 训练集加载接口
    │   ├── viewer.py              #   数据查询/浏览工具
    │   ├── elemental/             #   元素属性与相互作用矩阵 (.npy，9 个)
    │   └── datasets/              #   训练数据集 (.npz，Tg/Tx/Tl/E/H，含成分+标签+元数据)
    │
    ├── models/                    # ③ 模型定义
    │   └── neural_net.py          #   PropertyNN / PropertyNN_Small / ARCH_CONFIGS
    │
    ├── prediction/                # ③ 正向预测
    │   ├── predictor.py           #   Predictor（pytorch / python-GPR / matlab）
    │   ├── ensemble.py            #   EnsemblePredictor（集成 + 不确定性）
    │   ├── train.py               #   单模型训练
    │   ├── train_ensemble.py      #   集成训练
    │   ├── train_classifier.py    #   BMG/Ribbon 分类器训练
    │   └── artifacts/             #   训练好的权重（pytorch_models / ensembles / 分类器）
    │
    ├── inverse_design/            # ④ 反向设计
    │   ├── config.py              #   元素池 / 目标 / 超参数
    │   ├── net.py                 #   Generator 网络（噪声 → 元素分数）
    │   ├── extract.py             #   分数 → 成分
    │   ├── loop.py                #   探索循环 run_inverse_design
    │   ├── enhanced_loop.py       #   增强探索（排名 + 多样性 + 经验回放）
    │   ├── vae.py                 #   CompositionVAE + 训练
    │   ├── vae_refine.py          #   VAE 微调 / 插值
    │   ├── run.py                 #   命令行入口
    │   ├── vae_models/            #   VAE 权重
    │   └── output/                #   生成结果
    │
    ├── resistivity/               # （扩展）残余电阻率 R 预测与反向设计
    │   ├── train_predict.py       #   R 集成模型训练/预测
    │   ├── inverse_high_r.py      #   VAE 反向设计高 R 成分
    │   ├── visualize.py           #   可视化
    │   └── data/ models/ predictions/ figures/
    │
    └── scripts/matlab/            # MATLAB 桥接辅助脚本
        ├── matlab_export_models.m
        └── predict_matlab_bridge.m
```

## ① 特征构建（features/）

从成分字典计算 25 维（或 11 维）特征：

```python
from mgflow import compute_all_features, print_features

comp = {"Zr": 0.5, "Cu": 0.3, "Ni": 0.2}
feat = compute_all_features(comp)     # shape (25,)
feat11 = compute_all_features(comp, include_physics=False)  # shape (11,)
print_features(comp)                  # 打印每维特征
```

| 组 | 数量 | 说明 |
|----|------|------|
| dn/dc | 6 | 原子尺度应变/刚度失配 |
| Ye | 2 | RMS 原子应变、存储弹性能 |
| Scorr | 1 | 混合熵 + 弹性/化学涨落修正 |
| Hu | 2 | 内聚能偏差、相互作用能比 |
| Physics | 14 | r、Tm、EN、VEC、Bulk 的平均/偏差、混合焓、Hessian 特征值 |

## ② 数据条目存放（data/）

- `data/elemental/`：9 个 `.npy`，元素属性（原子尺寸、熔点、电负性、VEC、杨氏/体积模量）与相互作用矩阵（混合焓、相互作用能、正交化矩阵）。
- `data/datasets/`：5 个 `.npz`，每个含 `names / elem_nos / fracs / labels` 及元数据 `mg / bmg / ribbon / D / base_element`。

```python
from mgflow.data import load_dataset, list_available_datasets

list_available_datasets()                 # ['Tg', 'Tx', 'Tl', 'E', 'H']
d = load_dataset("Tg")                    # dict of arrays
# 查询工具
from mgflow.data.viewer import show_stats, query_element
show_stats()
```

## ③ 正向预测（prediction/）

三种预测模式 + 集成：

```python
from mgflow import Predictor, EnsemblePredictor

comp = {"Zr": 0.5, "Cu": 0.3, "Ni": 0.2}

# 单模型 PyTorch NN（推荐，无需 MATLAB）
pred = Predictor(mode="pytorch")
result = pred.predict(comp)
# => {"Tg":..., "Tx":..., "Tl":..., "E":..., "H":..., "BMG":..., "BMG_prob":...}

# 集成预测（含不确定性 _std）
ep = EnsemblePredictor(n_models=20)
result2 = ep.predict(comp)   # 额外含 Tg_std / Tx_std / ...
```

训练（首次需要，从 Excel 读取原始数据；Excel 路径默认指向上层 `Matlab 程序文件夹/TgTxTl_E_H/TgTxTl_E_H`）：

```bash
python -m mgflow.prediction.train             # 单模型
python -m mgflow.prediction.train_ensemble    # 集成
python -m mgflow.prediction.train_classifier  # BMG 分类器
```

## ④ 反向设计（inverse_design/）

两种互补策略：

- **探索（Explore）**：Generator 网络自由生成成分，无数据约束，可指定任意元素集。

```python
from mgflow import run_inverse_design, run_enhanced_inverse_design

result = run_inverse_design(
    target="BMG",                       # 或 Tg_high / Tl_low / E_high / supercooled_range ...
    elements=["Zr", "Cu", "Al", "Ni"],
    n_generations=200, batch_size=100,
)
result["best"]["composition"]           # 最优候选成分

# 增强版：排名选择 + 多样性 + 经验回放 + 集成不确定性
result2 = run_enhanced_inverse_design(target="Tl_high", elements=["Ta","Hf","Nb","Ni","Co"])
```

**多目标（加权求和标量化）**：除单目标外，`ALL_TARGETS` 还内置多目标预设，也支持自定义 dict：

```python
from mgflow import run_inverse_design

# 预设：同时提高 Tg 与 Tx（还有 high_GFA / thermal_stable / strong_glass）
r = run_inverse_design(target="TgTx_high", elements=["Zr", "Cu", "Al", "Ni"])

# 自定义多目标：大过冷液相区 ΔT + 低液相线温度 Tl + 目标杨氏模量
r = run_inverse_design(target={
    "objectives": [
        {"key": "Tx_minus_Tg", "mode": "maximize", "weight": 1.0},
        {"key": "Tl",          "mode": "minimize", "weight": 0.5},
        {"key": "E",           "mode": "target",    "target_value": 100.0},
    ],
    "normalize": "minmax",   # 各属性按训练数据 min/max 归一到 [0,1] 后加权
})
```

> 多目标用训练数据范围做 min-max 归一化后加权平均，`mode` 支持 `maximize / minimize / target`，`key` 支持 `Tg/Tx/Tl/E/H/BMG_prob/Tx_minus_Tg`。

- **微调（Refine）**：VAE 在已知 6453 条 BMG 成分流形上局部变分。

```python
from mgflow import refine_composition, interpolate_compositions

refined = refine_composition(
    seed_compositions=[{"Ta": 0.27, "Nb": 0.27, "Ni": 0.18, "Co": 0.17, "Hf": 0.11}],
    n_variants=200, perturbation_scale=0.3, target="Tl_high",
)
path = interpolate_compositions({"Zr":0.5,"Cu":0.3,"Ni":0.2}, {"Ti":0.4,"Zr":0.3,"Ni":0.3})
```

命令行：

```bash
python -m mgflow.inverse_design.run --target BMG --elements Zr,Cu,Al,Ni --generations 200
python -m mgflow.inverse_design.run --mode auto --target Tl_high
```

## 安装

```bash
cd MGflow_refactored
pip install -e .
pip install torch          # 预测需要
pip install scikit-learn openpyxl   # 分类器 / 重新训练需要
```

## 与原仓库的对应关系

| 功能 | 原位置 | 新位置 |
|------|--------|--------|
| 特征计算 | `mgflow/compute.py`, `mgflow/features/*` | `mgflow/features/*` |
| 数据加载/查询 | `mgflow/data_loader.py`, `mgflow/dataset_viewer.py` | `mgflow/data/loader.py`, `mgflow/data/viewer.py` |
| 模型定义 | `mgflow/pytorch_model.py` | `mgflow/models/neural_net.py` |
| 正向预测 | `mgflow/predict.py`, `mgflow/ensemble_predict.py` | `mgflow/prediction/predictor.py`, `mgflow/prediction/ensemble.py` |
| 训练脚本 | `mgflow/train_pytorch.py` 等 | `mgflow/prediction/train*.py` |
| 反向设计 | `mgflow/generator/*` | `mgflow/inverse_design/*` |
| 电阻率 | `mgflow/_hpc_resistivity/` | `mgflow/resistivity/` |
