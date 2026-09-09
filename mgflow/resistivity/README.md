# 残余电阻率 (Residual Resistivity, R) 预测模块（扩展）

自包含模块：从成分预测金属玻璃的残余电阻率 R（µΩ·cm），
使用在 955 条成分上训练的 PyTorch 神经网络集成。

## 目录结构
```
resistivity/
├── README.md              ← 本文件
├── train_predict.py       ← 训练 + 预测主脚本
├── inverse_high_r.py      ← VAE 反向设计高 R 成分
├── visualize.py           ← 出图（可选）
├── data/
│   └── resistivity_results.json   ← 原始数据（955 条）
├── models/                ← 训练好的集成（30 个模型 + 缩放参数）
├── predictions/           ← 预测结果
└── figures/               ← 可视化输出
```

## 快速开始

```bash
cd MGflow_refactored

# 1. 训练
python -m mgflow.resistivity.train_predict --train --n_models 30

# 2. 预测单个成分
python -m mgflow.resistivity.train_predict --predict --comp "Nb50Ta25Ni15Co10"

# 3. 批量预测（JSON 输入）
python -m mgflow.resistivity.train_predict --predict --input my_comp.json --output my_out.json

# 4. VAE 反向设计高 R 成分
python -m mgflow.resistivity.inverse_high_r --generate --top 50
```

输入 JSON 格式：
```json
[
  {"Ta": 0.5, "Nb": 0.3, "Ni": 0.2},
  {"Nb": 0.8, "Co": 0.1, "Hf": 0.1}
]
```

## 模型细节
- **架构**：2 隐层 NN（128 → 64 → 1），ReLU + Dropout
- **集成**：30 个独立训练模型（不同随机种子）
- **特征**：21 维理化特征（来自 mgflow.features.compute）
- **划分**：每模型 80/20 训练/验证（随机、种子相关）
- **性能**：平均验证 RMSE ≈ 11.7 ± 5.0 µΩ·cm
