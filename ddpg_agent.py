import random
from collections import deque

from actor_critic_net_all_in_one import ActorCriticNet
import torch.optim as optim
import numpy as np
import torch

from actor_critic_net_separate import ActorNet, CriticNet
from prioritized_memory import PrioritizedReplayBuffer
from replay_buffer import ReplayBuffer
from storage import Storage
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

number_of_actor_runs = 5
num_workers = 1
discount_factor = 0.999
gradient_clip = 0.01
entropy_weight = 0.001
value_loss_weight = 1.0

#memory
BATCH_SIZE = 64
PROBABILITY_EXPONENT = 0.8
BUFFER_SIZE = int(1e4)
GAMMA = 0.99


class DDPGAgent:

    def __init__(self, env, brain_name, state_size, action_size, num_agents=1):
        self.local_actor_network = ActorNet(state_size=state_size,
                                            action_size=action_size).to(device)
        self.target_actor_network = ActorNet(state_size=state_size,
                                             action_size=action_size).to(device)
        self.local_critic_network = CriticNet(state_size=state_size,
                                              action_size=action_size).to(device)
        self.target_critic_network = CriticNet(state_size=state_size,
                                               action_size=action_size).to(device)
        self.actor_optimizer = optim.Adam(self.local_actor_network.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.local_critic_network.parameters(), lr=0.0003)

        self.env = env
        self.num_agents = num_agents
        self.action_size = action_size
        self.brain_name = brain_name
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random.seed(0))

    def noise_process_for_exploration(self):
        theta = 0.15
        mu = 0
        dt = 1e-2
        # TODO write linear schedule
        std = 0.2
        return theta * mu * dt + std * np.sqrt(dt) * np.random.randn(*self.action_size)

    def step(self, states):
        # run the actor
        storage = Storage()

        # N-step TD error
        #for i in range(number_of_actor_runs):
        #actions = np.random.randn(self.num_agents, self.action_size)  # select an action (for each agent)
        #actions = np.clip(actions, -1, 1)  # all actions between -1 and 1

        # get action from actor network
        state_tensor = torch.from_numpy(states).float().to(device)
        prediction = self.local_network(state_tensor)
        action = prediction['action'].cpu().detach().numpy()
        action += self.noise_process_for_exploration()

        # take action in environment
        env_info = self.env.step(action)[self.brain_name]  # send all actions to tne environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        reward = env_info.rewards  # get reward (for each agent)
        done = env_info.local_done  # see if episode finished

        # put all in storage
        storage.action.append(action)
        storage.critic_value.append(prediction['critic_value'])

        # needs to be a pytorch tensor just like everything else
        done_value = 1 - int(done is True)
        storage.done.append(torch.from_numpy(np.array(done_value)).to(device))
        storage.entropy.append(prediction['entropy'])
        storage.log_prob_a.append(prediction['log_prob_a'])
        storage.mean_a.append(prediction['mean_a'])

        # needs to be a pytorch tensor just like everything else
        storage.reward.append(torch.from_numpy(np.array(reward)).to(device))

        # init advantage and return values in storage
        storage.advantage.append(None)
        storage.returns.append(None)

        self.memory.add(states, action, reward, next_states, done)

        if done[0]:  # exit loop if episode finished
            #break
            return

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()

            states, actions, rewards, next_states, dones = experiences

            # run actor once more
            state_tensor = torch.from_numpy(states).float().to(device)
            next_action = self.target_actor_network(state_tensor)

            # TODO ab hier gehts weiter
            # reverse fill advantage and return
            #for i in reversed(range(number_of_actor_runs)):
            for i in reversed(range(len(storage.action))):
                critic_value_for_next_state = storage.reward[i] + discount_factor * storage.done[i] * critic_value_for_next_state
                advantages = critic_value_for_next_state - storage.critic_value[i].detach()

                storage.advantage[i] = advantages.detach()
                storage.returns[i] = critic_value_for_next_state.detach()

            # calc the loss
            log_prob_a_tensor = torch.cat(storage.log_prob_a).squeeze()
            advantage_tensor = torch.cat(storage.advantage).squeeze()
            policy_loss = -(log_prob_a_tensor * advantage_tensor).mean()

            return_tensor = torch.cat(storage.returns).squeeze()
            critic_value_tensor = torch.cat(storage.critic_value).squeeze()
            value_loss = 0.5 * (return_tensor - critic_value_tensor).pow(2).mean()

            entropy_tensor = torch.cat(storage.entropy).squeeze()
            entropy_loss = entropy_tensor.mean()

            self.optimizer.zero_grad()
            (policy_loss + entropy_weight * entropy_loss + value_loss_weight * value_loss).backward()
            nn.utils.clip_grad_norm(self.network.parameters(), gradient_clip)
            self.optimizer.step()

    def evaluate(self, states):
        scores_window = deque(maxlen=100)

        for i in range(100):
            #actions = np.random.randn(self.num_agents, self.action_size)  # select an action (for each agent)
            #actions = np.clip(actions, -1, 1)  # all actions between -1 and 1

            # get action from actor network
            state_tensor = torch.from_numpy(states).float().to(device)
            prediction = self.network(state_tensor)
            action = prediction['action'].cpu().detach().numpy()

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