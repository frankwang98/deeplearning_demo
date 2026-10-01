"""Model shared by training and inference for lesson 02."""

from torch import nn


CLASS_NAMES = ("探索者", "建设者", "守护者")


class TinyClassifier(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(CLASS_NAMES)),
        )

    def forward(self, features):
        return self.network(features)
