import torch
from torch import nn

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
   
   
