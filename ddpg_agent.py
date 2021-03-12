import os
import random
import shutil

import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F

from actor_critic_net_separate import ActorNet, CriticNet
from environment import device
from ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from replay_buffer import ReplayBuffer
from DDPG_PARAMETERS import LR_ACTOR, LR_CRITIC, WEIGHT_DECAY, BUFFER_SIZE, BATCH_SIZE, NOISE_REDUCTION_FACTOR, UPDATE_EVERY, GAMMA, TAU
from save_and_plot import create_folder_structure_according_to_agent, save_score_plot


class DDPGAgent:

    def __init__(self, state_size, action_size, num_agents=1):
        """
        Initializes the agent with local and target networks for the actor and the critic.

        :param state_size:  dimensionality of the state space
        :param action_size: dimensionality of the action space
        """
        self.num_agents = num_agents
        # define actor networks
        self.local_actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.target_actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.actor_optimizer = optim.Adam(self.local_actor_network.parameters(), lr=LR_ACTOR)

        # define critic networks
        self.local_critic_network = CriticNet(state_size=state_size, action_size=action_size).to(device)
        self.target_critic_network = CriticNet(state_size=state_size, action_size=action_size).to(device)
        self.critic_optimizer = optim.Adam(self.local_critic_network.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random.seed(0))

        self.oup = OrnsteinUhlenbeckProcess(action_size=action_size)
        self.t_step = 0
        self.noise_weight = 1

    def act(self, state, add_noise: bool):
        """
        Retrieves an action from the local actor network given the current state.

        :param state:       state to get an action for
        :param add_noise:   if True, will add noise to an action given by the Ornstein-Uhlenbeck-Process
        :return:            chosen action
        """
        state_tensor = torch.from_numpy(state).float().to(device)

        # important to NOT create a gradient here because it's done later during learning and doing it twice corrupts
        # the gradients
        self.local_actor_network.eval()
        with torch.no_grad():

            # get the action
            action = self.local_actor_network(state_tensor)
            action = action.cpu().detach().numpy()

            if add_noise:
                self.noise_weight *= NOISE_REDUCTION_FACTOR
                action += self.oup.sample()*self.noise_weight
                # clip the action after noise adding to the boundaries for the environment
                # TODO make this a parameter of the environment that is chosen.
                action = np.clip(action, -1, 1)

        # change the network back to training mode to train it during the learning step
        self.local_actor_network.train()

        return action

    def step(self, state, action, next_observed_state, observed_reward, done):
        """
        Adds the current state, the taken action, next state, reward and done to the replay memory. Performs
        learning with the actor and critic networks.

        :param state:               Currently perceived state of the environment
        :param action:              Action performed in the environment
        :param next_observed_state: Next state observed in the environment
        :param observed_reward:     Observed reward after taken the action
        :param done:                Indicator if the episode is done or not
        """

        done_value = int(done[0] == True)
        self.memory.add(state, action, observed_reward, next_observed_state, done_value)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """
        Perform learning of the agent.

        :param experiences: Sample of size: batch size from the replay buffer.
        """
        state, actions, rewards, next_state, dones = experiences

        # run target actor to get next best action for the next_state
        next_target_action = self.target_actor_network(next_state)

        # evaluate the chosen next_action by the critic
        next_state_value = self.target_critic_network(next_state, next_target_action)
        target_critic_value_for_next_state = rewards + GAMMA * (1-dones) * next_state_value

        # obtain the local critics evaluation of the state
        local_value_current_state = self.local_critic_network(state, actions)

        # formulate a loss to drive the local critics estimation more towards the target critics evaluation including
        # the received reward
        critic_loss = F.mse_loss(local_value_current_state, target_critic_value_for_next_state)

        # train the critic
        self.local_critic_network.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # get forward pass for local actor on the current state to create an action.
        action = self.local_actor_network(state)
        # evaluate that action with the local critic to formulate a loss in the action according to its evaluation
        # important is the '-' in front because pytorch performs gradient decent but we want to maximize the value
        # of the critic
        policy_loss = -self.local_critic_network(state, action).mean()

        # learn the actor network
        self.local_actor_network.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # perform soft updates from the local networks to the targets to converge towards better evaluation values.
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

    def load_model_into_DDPG_agent(self, model_path):
        """
        Loads a pretrained network into the created agent.
        """
        self.local_actor_network.load_state_dict(torch.load(model_path))

    def save_current_agent(self, score_max, scores, i_episode):
        """
        Saves the current agent.

        :param agent:       agent to saved
        :param score_max:   max_score reached by the agent so far
        :param scores:      all scores of the agent reached so far
        :param i_episode:   number of current episode
        """
        new_folder_path = create_folder_structure_according_to_agent()

        os.makedirs(new_folder_path, exist_ok=True)
        torch.save(self.local_actor_network.state_dict(),
                   os.path.join(new_folder_path, f'checkpoint_{np.round(score_max, 2)}.pth'))
        save_score_plot(scores, i_episode, path=new_folder_path)
        shutil.copyfile("DDPG_PARAMETERS.py", os.path.join(new_folder_path, "DDPG_PARAMETERS.py"))