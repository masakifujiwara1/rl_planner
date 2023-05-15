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

class MakeState:
    def __init__(self):
        pass

    def toState(self, scan_data, target):
        scan_data_tensor, target_tensor = self.toTensor(scan_data, target)
        output_s, _ = torch.min(scan_data_tensor.view(int(SCAN_SAMPLE/MIN_POOLING_K), MIN_POOLING_K), dim=1)
        state = torch.cat([output_s, target_tensor], dim=0)
        return state

    def toTensor(self, scan_data, target):
        scan_data = scan_data.astype(np.float32)
        target = target.astype(np.float32)
        scan_data_tensor = torch.from_numpy(scan_data).clone()
        target_tensor = torch.from_numpy(target).clone()
        return scan_data_tensor, target_tensor

class SharedNetwork(nn.Module):
    def __init__(self):
        super(SharedNetwork, self).__init__()

        self.fc1 = nn.Linear(38, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)

    def forward(self, state):
        x2 = F.relu(self.fc1(state))
        x3 = F.relu(self.fc2(x2))
        x4 = F.relu(self.fc3(x3))
        return x4

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.shared_net = SharedNetwork()
        self.critic = nn.Linear(512, 1)
    
    def forward(self, state):
        shared_out = self.shared_net(state)
        critic_out = self.critic(shared_out)
        return critic_out

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.shared_net = SharedNetwork()
        self.mean_linear = nn.Linear(512, 2)
        self.log_std_linear = nn.Linear(512, 2)

        # self.action_scale = torch.tensor(action_scale)
        # self.action_bias = torch.tensor(0.)
    
    def forward(self, state):
        shared_out = self.shared_net(state)
        mean = self.mean_linear(shared_out)
        log_std = self.log_std_linear(shared_out)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        mean = torch.tanh(mean)
        return action, mean

    # def to(self, device):
    #     self.action_scale = self.action_scale.to(device)
    #     self.action_bias = self.action_bias.to(device)
    #     return super().to(device)

class ActorCriticModel(object):
    def __init__(self, args, device):

        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']

        self.target_update_interval = args['target_update_interval']

        self.device = device

        self.actor_net = ActorNet().to(self.device)
        self.critic_net = CriticNet().to(self.device)
        self.critic_net_target = CriticNet().to(self.device)

        hard_update(self.critic_net_target, self.critic_net)
        convert_network_grad_to_false(self.critic_net_target)

        self.actor_optim = optim.Adam(self.actor_net.parameters())
        self.critic_optim = optim.Adam(self.critic_net.parameters())

    def select_action(self, state, evaluate=False):
        scan_data = torch.FloatTensor(scan_data).unsqueeze(0).to(self.device)
        target = torch.FloatTensor(target).unsqueeze(0).to(self.device)
        if not evaluate:
            action, _ = self.actor_net.sample(state)
        else:
            _, action = self.actor_net.sample(state)
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

    def soft_update(target_net, source_net, tau):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(param.data)

    def convert_network_grad_to_false(network):
        for param in network.parameters():
            param.requires_grad = False

class ReplayMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.memory_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, mask)
        self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)