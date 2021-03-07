import torch
from torch import nn
import torch.nn.functional as F


class ActorNet(nn.Module):

    def __init__(self, state_size, action_size):
        super(ActorNet, self).__init__()

        self.actor_fc1 = nn.Linear(state_size, 400)
        self.actor_fc2 = nn.Linear(400, 300)
        self.actor_fc3 = nn.Linear(300, action_size)

    def forward(self, state):
        # get mean of action distribution
        x_a = F.relu(self.actor_fc1(state))
        x_a = F.relu(self.actor_fc2(x_a))
        return torch.tanh(self.actor_fc3(x_a))


class CriticNet(nn.Module):

    def __init__(self, state_size, action_size):
        super(CriticNet, self).__init__()

        self.critic_fc1 = nn.Linear(state_size, 400)
        self.critic_fc2 = nn.Linear(400 + action_size, 300)
        self.critic_fc3 = nn.Linear(300, 1)

    # TODO maybe reset parameters uniformly

    def forward(self, state, action):

        # get critics opinion on the state
        x_c = F.relu(self.critic_fc1(state))
        x_c = F.relu(self.critic_fc2(torch.cat([x_c, action], dim=1)))
        return self.critic_fc3(x_c)
