from actor_critic_net_all_in_one import ActorCriticNet
import torch.optim as optim
import numpy as np
import torch
from storage import Storage
import torch.nn as nn
from PARAMETERS import device, UPDATE_EVERY, GAMMA, ENTROPY_WEIGHT, VALUE_LOS_WEIGHT, GRADIENT_CLIP, NOISE_REDUCTION_FACTOR


class A2CAgent:

    def __init__(self, state_size, action_size, num_agents=1, ):
        self.network = ActorCriticNet(state_size=state_size,
                                      action_size=action_size,
                                      noise=1e-3).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0003)
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
        # important to NOT create a gradient here because it's done later during learning and doing it twice corrupts
        # the gradients
        state_tensor = torch.from_numpy(state).float().to(device)
        #self.network.eval()
        #with torch.no_grad():
        prediction = self.network(state_tensor)

        # change the network back to training mode to train it during the learning step
        #self.network.train()

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

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            self.learn(state)

    def learn(self, state):
        # run actor once more
        state_tensor = torch.from_numpy(state).float().to(device)
        prediction = self.network(state_tensor)

        #TODO reduce noise to the network std factor

        returns_of_next_state = prediction['critic_value'].squeeze().detach()

        # reverse fill advantage and return
        #for i in reversed(range(number_of_actor_runs)):
        for i in reversed(range(UPDATE_EVERY)):
            returns_of_next_state = self.storage.reward[i] + GAMMA * self.storage.done[i] * returns_of_next_state
            advantages = returns_of_next_state - self.storage.critic_value[i].squeeze().detach()

            self.storage.advantage[i] = advantages.detach()
            self.storage.returns[i] = returns_of_next_state.detach()

        # calc the loss
        log_prob_a_tensor = torch.cat(self.storage.log_prob_a).squeeze()
        advantage_tensor = torch.cat(self.storage.advantage).squeeze()
        policy_loss = -(log_prob_a_tensor * advantage_tensor).mean()

        return_tensor = torch.cat(self.storage.returns).squeeze()
        critic_value_tensor = torch.cat(self.storage.critic_value).squeeze()
        value_loss = 0.5 * (return_tensor - critic_value_tensor).pow(2).mean()

        entropy_tensor = torch.cat(self.storage.entropy).squeeze()
        entropy_loss = entropy_tensor.mean()

        self.optimizer.zero_grad()
        (policy_loss + ENTROPY_WEIGHT * entropy_loss + VALUE_LOS_WEIGHT * value_loss).backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), GRADIENT_CLIP)
        self.optimizer.step()

        # empty storage
        self.storage.empty()
