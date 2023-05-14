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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SharedNetwork(nn.Module):
    def __init__(self):
        super(SharedNetwork, self).__init__()

        self.fc1 = nn.Linear(38, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)

    def forward(self, scan_data, target):
        output_s, _ = torch.min(scan_data.view(int(SCAN_SAMPLE/MIN_POOLING_K), MIN_POOLING_K), dim=1)
        x1 = torch.cat([output_s, target], dim=0)
        x2 = F.relu(self.fc1(x1))
        x3 = F.relu(self.fc2(x2))
        x4 = F.relu(self.fc3(x3))
        return x4

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.shared_net = SharedNetwork()
        self.critic = nn.Linear(512, 1)
    
    def forward(self, scan_data, target):
        shared_out = self.shared_net(scan_data, target)
        critic_out = self.critic(shared_out)
        return critic_out

class ActorNet(nn.Module):
    def __init__(self, action_scale):
        super(ActorNet, self).__init__()
        self.shared_net = SharedNetwork()
        self.mean_linear = nn.Linear(512, 2)
        self.log_std_linear = nn.Linear(512, 2)

        self.action_scale = torch.tensor(action_scale)
        self.action_bias = torch.tensor(0.)
    
    def forward(self, scan_data, target):
        shared_out = self.shared_net(scan_data, target)
        mean = F.softplus(self.mean_linear(shared_out))
        log_std = F.softplus(self.log_std_linear(shared_out))
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, scan_data, target):
        mean, log_std = self.forward(scan_data, target)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        return action, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

class ActorCriticModel(object):

    def __init__(self, action_scale, args, device):

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']

        self.target_update_interval = args['target_update_interval']

        self.device = device

        self.actor_net = ActorNet(action_scale=action_scale).to(self.device)
        self.critic_net = CriticNet().to(self.device)
        self.critic_net_target = CriticNet().to(self.device)

        hard_update(self.critic_net_target, self.critic_net)
        convert_network_grad_to_false(self.critic_net_target)

        self.actor_optim = optim.Adam(self.actor_net.parameters())
        self.critic_optim = optim.Adam(self.critic_net.parameters())

    def select_action(self, scan_data, target, evaluate=False):
        scan_data = torch.FloatTensor(scan_data).unsqueeze(0).to(self.device)
        target = torch.FloatTensor(target).unsqueeze(0).to(self.device)
        if not evaluate:
            action, _ = self.actor_net.sample(scan_data, target)
        else:
            _, action = self.actor_net.sample(scan_data, target)
        return action.detach().numpy().reshape(-1)

    def update_parameters(self, memory, batch_size, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_state_action, _ = self.actor_net.sample(next_state_batch)
            next_q_values_target = self.critic_net_target(next_state_batch, next_state_action)
            next_q_values = reward_batch + mask_batch * self.gamma * next_q_values_target

        q_values = self.critic_net(state_batch, action_batch)
        critic_loss = F.mse_loss(q_values, next_q_values)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        action, _ = self.actor_net.sample(state_batch)
        q_values = self.critic_net(state_batch, action)
        actor_loss = - q_values.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_net_target, self.critic_net, self.tau)

        return critic_loss.item(), actor_loss.item()