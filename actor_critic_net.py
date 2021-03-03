import torch
from torch import nn
import torch.nn.functional as F


class ActorCriticNet(nn.Module):

    def __init__(self, state_size, action_size):
        super(ActorCriticNet, self).__init__()

        self.actor_fc1 = nn.Linear(state_size, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, action_size)

        self.critic_fc1 = nn.Linear(state_size, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        # set std of action distribution
        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state, action=None):
        # get mean of action distribution
        x_a = F.relu(self.actor_fc1(state))
        x_a = F.relu(self.actor_fc2(x_a))
        mean_a = torch.tanh(self.actor_fc3(x_a))

        # get critics opinion on the state
        x_c = F.relu(self.critic_fc1(state))
        x_c = F.relu(self.critic_fc2(x_c))
        critic_value = F.relu(self.critic_fc3(x_c))

        action_distribution = torch.distributions.Normal(mean_a, F.softplus(self.std))
        if action is None:
            action = action_distribution.sample()

        # get log probability of the action
        action_log_prob = action_distribution.log_prob(action).sum(-1).unsqueeze(-1)

        # get the entropy of the distribution
        entropy = action_distribution.entropy().sum(-1).unsqueeze(-1)

        return {'action': action,
                'log_prob_a': action_log_prob,
                'entropy': entropy,
                'mean_a': mean_a,
                'critic_value': critic_value}
