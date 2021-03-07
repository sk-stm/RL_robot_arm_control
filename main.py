from collections import deque
from typing import List

from unityagents import UnityEnvironment
import numpy as np

from a2c_agent import A2CAgent
from ddpg_agent import DDPGAgent

ENV_PATH = '/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux_single/Reacher.x86'
env = UnityEnvironment(file_name=ENV_PATH)


def main():
    brain_name, num_agents, agent_states, state_size, action_size = init_env()
    scores_window = run_environment(brain_name, num_agents, agent_states, state_size, action_size)
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores_window)))

    env.close()


def init_env():

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    agent_states = env_info.vector_observations
    state_size = agent_states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(agent_states.shape[0], state_size))
    print('The state for the first agent looks like:', agent_states[0])

    env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
    agent_states = env_info.vector_observations                  # get the current state (for each agent)

    return brain_name, num_agents, agent_states, state_size, action_size


num_episodes = 100000


def run_environment(brain_name, num_agents, agent_state, state_size, action_size):
    #agent = A2CAgent(env=env, brain_name=brain_name, state_size=state_size, action_size=action_size)
    agent = DDPGAgent(env=env, brain_name=brain_name, state_size=state_size, action_size=action_size)
    scores_window = deque(maxlen=100)

    for i_episode in range(1, num_episodes + 1):
        # TODO save after so many steps

        # TODO log results

        # TODO break if too many steps taken

        score = 0
        # teach the agent
        # reset environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get first state
        state = env_info.vector_observations[0]
        agent.oup.reset_process()

        # TODO make this a variable depending on the environment (episode length)
        for i_times in range(1000):
            action = agent.act(state)
            # take action in environment
            env_info = env.step(action)[brain_name]  # send all actions to tne environment
            next_observed_state = env_info.vector_observations  # get next state (for each agent)

            observed_reward = env_info.rewards  # get reward (for each agent)
            # The reward given was always ~ 0.02 which is not what the environment description explained.
            # So I changed it to 0.1 if it's greater than 0.
            observed_reward = [0.1 if rew > 0 else 0 for rew in observed_reward]

            done = env_info.local_done  # see if episode finished

            agent.step(state, action, next_observed_state, observed_reward, done)
            state = next_observed_state

            score += observed_reward[0]
            if done[0]:
                break

        scores_window.append(score)  # save most recent score

        # evaluate every so often
        if i_episode%10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="\n ")
        else:
            print('\rEpisode {}\tScor for this episode {:.2f}:'.format(i_episode, score), end="")


if __name__ == "__main__":
    main()
