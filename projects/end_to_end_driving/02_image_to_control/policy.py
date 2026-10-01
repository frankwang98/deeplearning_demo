"""Tiny camera-to-control policy."""

import torch
from torch import nn


class VisionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.control = nn.Sequential(nn.Linear(24 * 3 * 4 + 1, 32), nn.ReLU(), nn.Linear(32, 2), nn.Tanh())

    def forward(self, image, speed):
        return self.control(torch.cat((self.features(image), speed), dim=1))
