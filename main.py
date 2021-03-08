import datetime
import os
from collections import deque
from typing import List
import shutil

from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import torch
from PARAMETERS import device

from a2c_agent import A2CAgent
from actor_critic_net_separate import ActorNet
from ddpg_agent import DDPGAgent

ENV_PATH = '/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux_single/Reacher.x86'
env = UnityEnvironment(file_name=ENV_PATH)
TRAIN_MODE = True
MODEL_TO_LOAD = '/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/DDPG/2021_03_08_12_58_06/checkpoint_19.95.pth'


def main():
    brain_name, num_agents, agent_states, state_size, action_size = init_env()

    # agent = A2CAgent(env=env, brain_name=brain_name, state_size=state_size, action_size=action_size)
    agent = DDPGAgent(state_size=state_size, action_size=action_size)

    if not TRAIN_MODE:
        load_model_into_agent(agent, state_size, action_size)

    run_environment(brain_name, agent)

    env.close()


def load_model_into_agent(agent, state_size, action_size):
    """
    Loads a pretrained network into the created agent.
    """
    actor_network = ActorNet(state_size=state_size, action_size=action_size).to(device)
    actor_network.load_state_dict(torch.load(MODEL_TO_LOAD))
    agent.local_actor_network = actor_network


def init_env():
    """
    Initialized the environment.

    :return: brain_name, num_agents, agent_states, state_size, action_size
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]

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

    agent_states = env_info.vector_observations                  # get the current state (for each agent)

    return brain_name, num_agents, agent_states, state_size, action_size


num_episodes = 100000


def run_environment(brain_name, agent):
    """
    Runs the environment and the agent.

    :param brain_name:  name of the brain of the environment
    :param agent:       the agent to act in this environment
    """
    # lists containing scores from each episode
    scores_window = deque(maxlen=100)
    score_max = 0
    scores = []

    for i_episode in range(1, num_episodes + 1):
        score = 0
        # get first state of environment
        env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]
        state = env_info.vector_observations[0]
        agent.oup.reset_process()

        # TODO make this a variable depending on the environment (episode length)
        for i_times in range(1000):
            # take action in environment and get its response
            action = agent.act(state, add_noise=TRAIN_MODE)
            env_info = env.step(action)[brain_name]
            next_observed_state = env_info.vector_observations
            observed_reward = env_info.rewards
            # The reward given was always ~ 0.02 which is not what the environment description explained.
            # So I changed it to 0.1 if it's greater than 0. Also stated in the Udacity peer chat:
            # https://hub.udacity.com/rooms/community:nd893:845401-project-503-smg-2?contextType=room
            observed_reward = [0.1 if rew > 0 else 0 for rew in observed_reward]
            done = env_info.local_done

            if TRAIN_MODE:
                agent.step(state, action, next_observed_state, observed_reward, done)

            state = next_observed_state

            score += observed_reward[0]
            if done[0]:
                break

        # save the obtained scores
        scores_window.append(score)
        scores.append(score)
        score_mean = np.mean(scores_window)

        plot_and_save_agent(agent, i_episode, score_max, scores, score_mean, score)

        if score_mean > score_max:
            score_max = score_mean


def plot_and_save_agent(agent, i_episode, score_max, scores, scores_mean, score):
    """
    Plots and saves the agent each 100th episode.
    Saves the agent, the current scores, the episode number, the trained parameters of the NN model and the hyper
    parameters of the agent to a folder with the current date and time if the mean average of the last 100 scores
    are > 13 and if a new maximum average was reached.

    :param agent:           agent to saved
    :param eps:             current value of epsilon
    :param i_episode:       number of current episode
    :param score_max:       max_score reached by the agent so far
    :param scores:          all scores of the agent reached so far
    :param scores_mean:     mean of the last 100 scores
    :param score:           score of the running episode
    """

    # evaluate every so often
    if i_episode % 10 == 0 and TRAIN_MODE:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores_mean), end="\n ")
    else:
        print('\rEpisode {}\tScor for this episode {:.2f}:'.format(i_episode, score), end="")
    if i_episode % 100 == 0 and TRAIN_MODE:
        print('\nEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores_mean), end="")
        save_score_plot(scores, i_episode)
    if scores_mean >= 20.0 and scores_mean > score_max and TRAIN_MODE:
        save_current_agent(agent, score_max, scores, i_episode)
        # TODO save replay buffer parameters as well if prioritized replay buffer was used
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} '.format(i_episode - 100, scores_mean))


def save_current_agent(agent, score_max, scores, i_episode):
    """
    Saves the current agent.

    :param agent:       agent to saved
    :param score_max:   max_score reached by the agent so far
    :param scores:      all scores of the agent reached so far
    :param i_episode:   number of current episode
    """
    new_folder_path = create_folder_structure_according_to_agent(agent)

    os.makedirs(new_folder_path, exist_ok=True)
    torch.save(agent.local_actor_network.state_dict(),
               os.path.join(new_folder_path, f'checkpoint_{np.round(score_max, 2)}.pth'))
    save_score_plot(scores, i_episode, path=new_folder_path)
    shutil.copyfile("PARAMETERS.py", os.path.join(new_folder_path, "PARAMETERS.py"))


def create_folder_structure_according_to_agent(agent):
    """
    Creates a folder structure to store the current experiment according to the type of agent that was run and the
    current date and time.

    :param agent: Agent to be stored
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    if type(agent) == DDPGAgent:
        new_folder_path = os.path.join('DDPG', f'{date_str}')
    else:
        raise NotImplementedError()
    return new_folder_path


def save_score_plot(scores: List, i_episode: int, path: str = ""):
    """
    Saves a plot of numbers to a folder path. The The i_episode number is added to the name of the file.

    :param scores:      All numbers to store
    :param i_episode:   Current number of episodes
    :param path:        Path to the folder to store the plot to.
    """
    fig = plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(os.path.join(path, f'score_plot_{i_episode}.jpg'))


if __name__ == "__main__":
    main()
