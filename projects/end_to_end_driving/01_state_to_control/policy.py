"""Neural policy shared by training and closed-loop evaluation."""

from torch import nn


class DrivingPolicy(nn.Module):
    INPUT_SIZE = 5
    OUTPUT_SIZE = 2

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.OUTPUT_SIZE),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.network(state)
