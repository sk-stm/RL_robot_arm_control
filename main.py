from collections import deque
from typing import List

from unityagents import UnityEnvironment
import numpy as np

from a2c_agent import A2CAgent

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


num_episodes = 1000


def run_environment(brain_name, num_agents, agent_states, state_size, action_size):
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    agent = A2CAgent(env=env, brain_name=brain_name, state_size=state_size, action_size=action_size)

    scores_window = deque(maxlen=100)

    for i_episode in range(1, num_episodes + 1):
        # TODO save after so many steps

        # TODO log results

        # TODO break if too many steps taken

        score = agent.step(agent_states)
        scores_window.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

    return scores_window


if __name__ == "__main__":
    main()