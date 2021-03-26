import os
import shutil

from actor_critic_net_all_in_one import ActorCriticNet
import torch.optim as optim
import numpy as np
import torch

from actor_critic_net_all_in_one_crawler import ActorCriticNetCrawler
from environment import device, ENV_NAME
from save_and_plot import create_folder_structure_according_to_agent, save_score_plot
from storage import Storage
import torch.nn as nn
from A2C_PARAMETERS import ACTIONS_BETWEEN_LEARNING, GAMMA, ENTROPY_WEIGHT, VALUE_LOS_WEIGHT, GRADIENT_CLIP, NOISE_REDUCTION_FACTOR, \
    NOISE_ON_THE_ACTIONS, LEARNING_RATE, WEIGHT_DECAY, USE_GAE, GAE_TAU


class A2CAgent:

    def __init__(self, state_size, action_size, num_agents=1, ):
        if ENV_NAME == "REACHER":
            self.network = ActorCriticNet(state_size=state_size,
                                          action_size=action_size,
                                          noise=NOISE_ON_THE_ACTIONS).to(device)
        elif ENV_NAME == "CRAWLER":
            self.network = ActorCriticNetCrawler(state_size=state_size,
                                                 action_size=action_size,
                                                 noise=NOISE_ON_THE_ACTIONS).to(device)

        self.mse = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.num_agents = num_agents
        self.action_size = action_size
        self.storage = Storage()
        self.t_step = 0

    def act(self, state):
        """
        Retrieves an action from the local actor network given the current state.

        :param state:       state to get an action for
        :param add_noise:   if True, will add noise to an action given by the Ornstein-Uhlenbeck-Process
        :return:            chosen action
        """
        state_tensor = torch.from_numpy(state).float().to(device)
        prediction = self.network(state_tensor)

        return prediction

    def step(self, state, prediction, observed_reward, done):
        # run the actor
        done_value = np.where(np.array(done) == True, 0, 1)
        self.storage.action.append(prediction['action'])
        self.storage.critic_value.append(prediction['critic_value'])
        self.storage.done.append(torch.from_numpy(np.array(done_value)).to(device))
        self.storage.entropy.append(prediction['entropy'])
        self.storage.log_prob_a.append(prediction['log_prob_a'])
        self.storage.mean_a.append(prediction['mean_a'])

        self.storage.reward.append(torch.from_numpy(np.array(observed_reward)).to(device))
        self.storage.advantage.append(None)
        self.storage.returns.append(None)

        self.t_step = (self.t_step + 1) % ACTIONS_BETWEEN_LEARNING
        if self.t_step == 0:
            self.learn(state)
            self.storage.empty()

    def learn(self, state):
        #with torch.autograd.detect_anomaly():
            # run actor once more
        state_tensor = torch.from_numpy(state).float().to(device)
        prediction = self.network(state_tensor)

        # reduce noise to the network std factor
        self.network.std *= NOISE_REDUCTION_FACTOR

        returns_of_next_state = prediction['critic_value'].squeeze().detach()

        advantages = torch.from_numpy(np.zeros((self.num_agents, 1))).to(device).squeeze()
        # reverse fill advantage and return
        for i in reversed(range(ACTIONS_BETWEEN_LEARNING)):
            returns_of_next_state = self.storage.reward[i] + GAMMA * self.storage.done[i] * returns_of_next_state
            self.storage.returns[i] = returns_of_next_state.detach()

            if not USE_GAE:
                advantages = returns_of_next_state - self.storage.critic_value[i].squeeze().detach()
            else:
                if i < ACTIONS_BETWEEN_LEARNING-1:
                    td_error = self.storage.reward[i] + GAMMA * self.storage.done[i] * self.storage.critic_value[i+1].squeeze() - self.storage.critic_value[i].squeeze()
                    advantages = advantages * GAE_TAU * GAMMA * self.storage.done[i] + td_error
                else:
                    td_error = self.storage.reward[i] + GAMMA * self.storage.done[i] * returns_of_next_state - self.storage.critic_value[i].squeeze()
                    advantages = advantages * GAE_TAU * GAMMA * self.storage.done[i] + td_error

            self.storage.advantage[i] = advantages.detach()

        # calc the loss
        log_prob_a_tensor = torch.cat(self.storage.log_prob_a).squeeze()
        advantage_tensor = torch.cat(self.storage.advantage).squeeze()
        policy_loss = -(log_prob_a_tensor * advantage_tensor).mean()

        return_tensor = torch.cat(self.storage.returns).squeeze()
        critic_value_tensor = torch.cat(self.storage.critic_value).squeeze()
        #value_loss = 0.5 * (return_tensor - critic_value_tensor).pow(2).mean()

        value_loss = 0.5 * self.mse(return_tensor.float(), critic_value_tensor.float())
        value_loss = value_loss.float()

        entropy_tensor = torch.cat(self.storage.entropy).squeeze()
        entropy_loss = entropy_tensor.mean()

        self.optimizer.zero_grad()
        (policy_loss + ENTROPY_WEIGHT * entropy_loss + VALUE_LOS_WEIGHT * value_loss).backward()

        nn.utils.clip_grad_norm_(self.network.parameters(), GRADIENT_CLIP)
        self.optimizer.step()

    def load_model_into_A3C_agent(self, model_path):
        self.network.load_state_dict(torch.load(model_path))

    def save_current_agent(self, score_max, scores, score_mean_list, i_episode):
        """
        Saves the current agent.

        :param agent:       agent to saved
        :param score_max:   max_score reached by the agent so far
        :param scores:      all scores of the agent reached so far
        :param i_episode:   number of current episode
        """
        new_folder_path = create_folder_structure_according_to_agent()

        os.makedirs(new_folder_path, exist_ok=True)
        torch.save(self.network.state_dict(),
                   os.path.join(new_folder_path, f'checkpoint_{np.round(score_max, 2)}.pth'))
        save_score_plot(scores, score_mean_list, i_episode, path=new_folder_path)
        shutil.copyfile("A2C_PARAMETERS.py", os.path.join(new_folder_path, "A2C_PARAMETERS.py"))