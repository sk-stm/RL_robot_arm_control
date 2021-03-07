import random
from collections import deque

import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F

from actor_critic_net_separate import ActorNet, CriticNet
from ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from replay_buffer import ReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA = 0.999
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 1e-3
LR_CRITIC = 3e-3
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5
NOISE_REDUCTION_FACTOR = 0.9999


class DDPGAgent:

    def __init__(self, env, brain_name, state_size, action_size):
        self.local_actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.target_actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.actor_optimizer = optim.Adam(self.local_actor_network.parameters(), lr=LR_ACTOR)

        self.local_critic_network = CriticNet(state_size=state_size, action_size=action_size).to(device)
        self.target_critic_network = CriticNet(state_size=state_size, action_size=action_size).to(device)
        self.critic_optimizer = optim.Adam(self.local_critic_network.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random.seed(0))

        self.oup = OrnsteinUhlenbeckProcess(action_size=action_size)
        self.t_step = 0
        self.noise_weight = 1

    def act(self, state):
        state_tensor = torch.from_numpy(state).float().to(device)
        self.local_actor_network.eval()
        with torch.no_grad():
            action = self.local_actor_network(state_tensor)
            action = action.cpu().detach().numpy()

            self.noise_weight *= NOISE_REDUCTION_FACTOR
            action += self.oup.sample()*self.noise_weight
            action = np.clip(action, -1, 1)

        self.local_actor_network.train()

        return action

    def step(self, state, action, next_observed_state, observed_reward, done):
        # N-step TD error
        #for i in range(number_of_actor_runs):

        done_value = int(done[0] == True)
        self.memory.add(state, action, observed_reward, next_observed_state, done_value)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):

        state, actions, rewards, next_state, dones = experiences

        # run target actor to get next best action
        next_target_action = self.target_actor_network(next_state)
        next_state_value = self.target_critic_network(next_state, next_target_action)
        target_critic_value_for_next_state = rewards + GAMMA * (1-dones) * next_state_value

        # TODO this detach is maybe not needed because the target network is never optimized
        #target_critic_value_for_next_state = target_critic_value_for_next_state.detach()

        local_value_current_state = self.local_critic_network(state, actions)

        # important to put a - in front because optimizer will do gradient decent
        critic_loss = F.mse_loss(local_value_current_state, target_critic_value_for_next_state)

        self.local_critic_network.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action = self.local_actor_network(state)
        #policy_loss = -self.local_critic_network(state.detach(), action).mean()
        policy_loss = -self.local_critic_network(state, action).mean()

        self.local_actor_network.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.local_actor_network, self.target_actor_network, TAU)
        self.soft_update(self.local_critic_network, self.target_critic_network, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        :param tau: (float) interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def evaluate(self, states):
        scores_window = deque(maxlen=100)

        for i in range(100):

            # get action from actor network
            state_tensor = torch.from_numpy(states).float().to(device)
            action = self.target_actor_network(state_tensor)
            action = action.cpu().detach().numpy()

            # take action in environment
            env_info = self.env.step(action)[self.brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            reward = env_info.rewards  # get reward (for each agent)
            done = env_info.local_done  # see if episode finished

            scores_window.append(reward[0])

            states = next_states  # roll over states to next time step
            if done[0]:  # exit loop if episode finished
                break

        return scores_window
