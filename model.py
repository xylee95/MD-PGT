import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.dense1 = nn.Linear(state_dim, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, action_dim)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x1 = torch.tanh(self.dense1(x))
        x2 = torch.tanh(self.dense2(x1))
        x3 = self.dense3(x2)
        dist = Categorical(logits=x3)
        return dist

def select_action(state, policy):
    try:
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
    except:
        pass
    dist = policy(state) 
    action = dist.sample() 
    policy.saved_log_probs.append(dist.log_prob(action)) 
    return action