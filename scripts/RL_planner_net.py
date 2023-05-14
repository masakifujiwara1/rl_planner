from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns
import RL_settings

# hyper param
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

SCAN_SAMPLE = 720
MIN_POOLING_K = 20

model_dir_path = Path('mdoel')
result_dir_path = Path('result')
if not model_dir_path.exists():
    model_dir_path.mkdir()
if not result_dir_path.exists():
    result_dir_path.mkdir()

class SharedNetwork(nn.Module):
    def __init__(self):
        super(SharedNetwork, self).__init__()

        self.fc1 = nn.Linear(38, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)

    def forward(self, x_s, x_t):
        output_s, _ = torch.min(x_s.view(int(SCAN_SAMPLE/MIN_POOLING_K), MIN_POOLING_K), dim=1)
        x1 = torch.cat([output_s, x_t], dim=0)
        x2 = F.relu(self.fc1(x1))
        x3 = F.relu(self.fc2(x2))
        x4 = F.relu(self.fc3(x3))
        return x4

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.shared_net = SharedNetwork()
        self.critic = nn.Linear(512, 1)
    
    def forward(self, x_s, x_t):
        shared_out = self.shared_net(x_s, x_t)
        critic_out = self.critic(shared_out)
        return critic_out

class ActorNet(nn.Module):
    def __init__(self, action_scale):
        super(ActorNet, self).__init__()
        self.shared_net = SharedNetwork()
        self. = nn.Linear(512, 2)
    
    def forward(self, x_s, x_t):
        shared_out = self.shared_net(x_s, x_t)
        critic_out = self.critic(shared_out)
        return critic_out

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        
        self.fc1 = nn.Linear(38, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)

        self.action_head = nn.Linear(512, 2)
        self.value_head = nn.Linear(512, 1)

        self.saved_actions = []
        self.saved_rewards = []

    def forward(self, x_s, x_t):
        output_s, _ = torch.min(x_s.view(int(SCAN_SAMPLE/MIN_POOLING_K), MIN_POOLING_K), dim=1)
        x1 = torch.cat([output_s, x_t], dim=0)
        x2 = F.relu(self.fc1(x1))
        x3 = F.relu(self.fc2(x2))
        x4 = F.relu(self.fc3(x3))

        action_prob = F.softplus(self.action_head(x4))
        state_values = self.value_head(x4)

        return action_prob, state_values

class PPO:
