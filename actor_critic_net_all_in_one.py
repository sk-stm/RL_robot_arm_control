import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from environment import device


class ActorCriticNet(nn.Module):

    def __init__(self, state_size, action_size, noise):
        super(ActorCriticNet, self).__init__()

        self.actor_fc1 = nn.Linear(state_size, 64)
        self.actor_fc2 = nn.Linear(64, 128)
        self.actor_fc3 = nn.Linear(128, 256)
        self.actor_fc4 = nn.Linear(256, action_size)

        self.critic_fc1 = nn.Linear(state_size, 64)
        self.critic_fc2 = nn.Linear(64, 128)
        self.critic_fc3 = nn.Linear(128, 256)
        self.critic_fc4 = nn.Linear(256, 1)

        # set std of action distribution
        #self.std = nn.Parameter(torch.zeros(action_size) + noise)
        self.std = np.zeros(action_size) + noise

    def forward(self, state, action=None):
        # get mean of action distribution
        x_a = F.relu(self.actor_fc1(state))
        x_a = F.relu(self.actor_fc2(x_a))
        x_a = F.relu(self.actor_fc3(x_a))
        mean_a = torch.tanh(self.actor_fc4(x_a))

        #TODO currently only the mean is changed, one could also change the std with another head

        # get critics opinion on the state
        x_c = F.relu(self.critic_fc1(state))
        x_c = F.relu(self.critic_fc2(x_c))
        x_c = F.relu(self.critic_fc3(x_c))
        critic_value = self.critic_fc4(x_c)

        #action_distribution = torch.distributions.Normal(mean_a, F.softplus(self.std))
        action_distribution = torch.distributions.Normal(mean_a, F.softplus(torch.from_numpy(self.std).to(device)))
        if action is None:
            action = action_distribution.sample()

        # get log probability of the action to implement REINFORCE: https://pytorch.org/docs/stable/distributions.html
        action_log_prob = action_distribution.log_prob(action).sum(-1).unsqueeze(-1)

        # get the entropy of the distribution
        entropy = action_distribution.entropy().sum(-1).unsqueeze(-1)

        return {'action': action,
                'log_prob_a': action_log_prob,
                'entropy': entropy,
                'mean_a': mean_a,
                'critic_value': critic_value}

