from __future__ import annotations

import torch
import torch.nn as nn


def hidden_dim_from_nb_bonds(d: int) -> int:
    """
    Heuristic to determine the hidden dimension of the actor and critic networks based on the number of bonds in the molecule.
    """
    if d <= 1:
        return 10
    if d <= 2:
        return 12
    if d <= 5:
        return 18
    if d <= 10:
        return 20
    return 30

class Actor(nn.Module):
    def __init__(self, state_dim, hidden=30):
        super().__init__()
        hidden = hidden_dim_from_nb_bonds(state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return torch.sigmoid(self.net(state))


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=30):
        super().__init__()
        hidden = hidden_dim_from_nb_bonds(state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state)