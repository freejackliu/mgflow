"""PyTorch neural network models for metallic glass property prediction."""

import torch
import torch.nn as nn


class PropertyNN(nn.Module):
    """Flexible feed-forward neural network for regression.

    Architecture: [BN] → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear
    """

    def __init__(self, n_input, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = n_input
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PropertyNN_Small(nn.Module):
    """Smaller NN for datasets with few samples (e.g. H: 62 samples)."""

    def __init__(self, n_input, hidden_dims=(32, 16), dropout=0.3):
        super().__init__()
        layers = []
        prev = n_input
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Architecture presets per property based on dataset size
ARCH_CONFIGS = {
    "Tg": {"class": PropertyNN, "hidden_dims": (128, 64), "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-5, "epochs": 500, "patience": 50},
    "Tx": {"class": PropertyNN, "hidden_dims": (128, 64), "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-5, "epochs": 500, "patience": 50},
    "Tl": {"class": PropertyNN, "hidden_dims": (128, 64), "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-5, "epochs": 500, "patience": 50},
    "E":  {"class": PropertyNN, "hidden_dims": (64, 32),  "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 500, "patience": 60},
    "H":  {"class": PropertyNN_Small, "hidden_dims": (32, 16), "dropout": 0.4, "lr": 5e-4, "weight_decay": 1e-3, "epochs": 800, "patience": 100},
}
