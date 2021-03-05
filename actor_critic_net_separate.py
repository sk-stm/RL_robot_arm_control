import torch
from torch import nn
import torch.nn.functional as F


class ActorNet(nn.Module):

    def __init__(self, state_size, action_size):
        super(ActorNet, self).__init__()

        self.actor_fc1 = nn.Linear(state_size, 128)
        self.actor_fc2 = nn.Linear(128, 256)
        self.actor_fc3 = nn.Linear(256, 512)
        self.actor_fc4 = nn.Linear(512, action_size)

    def forward(self, state, action=None):
        # get mean of action distribution
        x_a = F.relu(self.actor_fc1(state))
        x_a = F.relu(self.actor_fc2(x_a))
        x_a = F.relu(self.actor_fc3(x_a))
        return torch.tanh(self.actor_fc4(x_a))


class CriticNet(nn.Module):

    def __init__(self, state_size, action_size):
        super(CriticNet, self).__init__()

        self.critic_fc1 = nn.Linear(state_size + action_size, 128)
        self.critic_fc2 = nn.Linear(128, 256)
        self.critic_fc3 = nn.Linear(256, 512)
        self.critic_fc4 = nn.Linear(512, 1)

    def forward(self, state, action):

        # get critics opinion on the state
        x_c = F.relu(self.critic_fc1(torch.cat([state, action], dim=1)))
        x_c = F.relu(self.critic_fc2(x_c))
        x_c = F.relu(self.critic_fc3(x_c))
        return F.relu(self.critic_fc4(x_c))