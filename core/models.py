import torch
from torch import nn
from config.constants import *

torch.manual_seed(42)    

class Actor(nn.Module):
    def __init__(self, state_dim, hidden= 30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state):
        return torch.sigmoid(self.net(state))
        
        
class Critic(nn.Module): 
    def __init__(self, state_dim, hidden= 30): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            
        )
    def forward(self, state): 
       return self.net(state)
   
   
class RandomPolicy(nn.Module):
    def forward(self, x):
        if x.dim() == 1:
            B = 1
        else:
            B = x.shape[0]
        u = torch.rand(B, device=x.device, dtype=torch.float32)
        return NU + (1.0 - 2.0 * NU) * u


class ConstantPolicy(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if x.dim() == 1:
            B = 1
        else:
            B = x.shape[0]
        return torch.full((B,), self.p, device=x.device, dtype=torch.float32)
