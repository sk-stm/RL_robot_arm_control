from actor_critic_net import ActorCriticNet
import torch.optim as optim
import numpy as np
import torch
from storage import Storage
import torch.nn as nn


number_of_actor_runs = 5
num_workers = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
discount_factor = 0.99
gradient_clip = 0.1
entropy_weight = 0.01
value_loss_weight = 1.0


class A2CAgent:

    def __init__(self, env, brain_name, state_size, action_size, num_agents=1):
        self.network = ActorCriticNet(state_size=state_size,
                                      action_size=action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0003)
        self.env = env
        self.num_agents = num_agents
        self.action_size = action_size
        self.brain_name = brain_name

    def step(self, states):
        # run the actor
        storage = Storage()

        score = 0

        for i in range(number_of_actor_runs):
            # TODO get action from actor network
            #actions = np.random.randn(self.num_agents, self.action_size)  # select an action (for each agent)
            #actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
            state_tensor = torch.from_numpy(states).float().to(device)
            prediction = self.network(state_tensor)
            action = prediction['action'].cpu().detach().numpy()

            # take action in environment
            env_info = self.env.step(action)[self.brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            reward = env_info.rewards  # get reward (for each agent)
            done = env_info.local_done  # see if episode finished

            score += reward[0]

            # put all in storage
            storage.action.append(action)
            storage.critic_value.append(prediction['critic_value'])

            # needs to be a pytorch tensor just like everything else
            storage.done.append(torch.from_numpy(np.array(done)).to(device))
            storage.entropy.append(prediction['entropy'])
            storage.log_prob_a.append(prediction['log_prob_a'])
            storage.mean_a.append(prediction['mean_a'])

            # needs to be a pytorch tensor just like everything else
            storage.reward.append(torch.from_numpy(np.array(reward)).to(device))

            # init advantage and return values in storage
            storage.advantage.append(None)
            storage.returns.append(None)

            #scores += env_info.rewards  # update the score (for each agent)

            states = next_states  # roll over states to next time step
            if done[0]:  # exit loop if episode finished
                # TODO check if this is correct to return completely not just a break
                return score

        # run actor once more
        state_tensor = torch.from_numpy(states).float().to(device)
        prediction = self.network(state_tensor)

        # TODO maybe uder fp32 explicitly here before "from_numpy"
        # advantages = torch.from_numpy((np.zeros(num_workers, 1))).to(device)

        returns = prediction['critic_value'].detach()

        # reverse fill advantage and return
        for i in reversed(range(number_of_actor_runs)):
            returns = storage.reward[i] + discount_factor * storage.done[i] * returns
            advantages = returns - storage.critic_value[i]

            # TODO look if the detach is correct at all the places... this seems odd
            storage.advantage[i] = advantages  # .detach()
            storage.returns[i] = returns  # .detatch()

        # calc the loss
        # TODO check if this is correct
        log_prob_a_array = torch.cat(storage.log_prob_a).squeeze()
        advantage_array = torch.cat(storage.advantage).squeeze()
        policy_loss = -(log_prob_a_array * advantage_array).mean()

        return_array = torch.cat(storage.returns).squeeze()
        critic_value_array = torch.cat(storage.critic_value).squeeze()
        value_loss = 0.5 * (return_array - critic_value_array).pow(2).mean()

        entropy_array = torch.cat(storage.entropy).squeeze()
        entropy_loss = entropy_array.mean()

        self.optimizer.zero_grad()
        (policy_loss + entropy_weight * entropy_loss + value_loss_weight * value_loss).backward()
        nn.utils.clip_grad_norm(self.network.parameters(), gradient_clip)
        self.optimizer.step()

        return score