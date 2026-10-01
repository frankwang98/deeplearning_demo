"""Image-to-waypoint network for the hybrid end-to-end stage."""

import torch
from torch import nn


LOOKAHEADS = torch.tensor((4.0, 8.0, 12.0))


class WaypointPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten(),
        )
        self.waypoints = nn.Sequential(
            nn.Linear(24 * 3 * 4 + 1, 32), nn.ReLU(), nn.Linear(32, len(LOOKAHEADS)), nn.Tanh()
        )

    def forward(self, image, speed):
        return self.waypoints(torch.cat((self.features(image), speed), dim=1))
