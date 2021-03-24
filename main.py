from collections import deque
from unityagents import UnityEnvironment
import numpy as np

from ddpg_agent import DDPGAgent
from a3c_agent import A3CAgent

from environment import ENV_PATH, MODEL_TO_LOAD, AGENT_TYPE, ENV_NAME, NEEDED_REWARD_FOR_SOLVING_ENV, SAVE_EACH_NEXT_BEST_REWARD
from save_and_plot import save_score_plot

env = UnityEnvironment(file_name=ENV_PATH)
TRAIN_MODE = True
LOAD_MODEL = False
num_episodes = 2000000


def main():
    brain_name, num_agents, agent_states, state_size, action_size = init_env()

    if AGENT_TYPE == 'DDPG':
        agent = DDPGAgent(state_size=state_size, action_size=action_size)
        if LOAD_MODEL or not TRAIN_MODE:
            agent.load_model_into_DDPG_agent(model_path=MODEL_TO_LOAD)
    elif AGENT_TYPE == 'A3C':
        agent = A3CAgent(state_size=state_size, action_size=action_size, num_agents=num_agents)
        if LOAD_MODEL or not TRAIN_MODE:
            agent.load_model_into_A3C_agent(model_path=MODEL_TO_LOAD)
    else:
        raise NotImplementedError()

    run_environment(brain_name, agent)

    env.close()


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
    score_mean_list = []
    saved_earliest_agent = False

    for i_episode in range(1, num_episodes + 1):
        score = 0
        # get first state of environment
        env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]

        # check 1 vs multi agent environment
        if agent.num_agents == 1:
            state = env_info.vector_observations[0]
            agent.oup.reset_process()
        else:
            state = env_info.vector_observations

        # check for environment type
        if ENV_NAME == 'REACHER':
            max_episode_length = 1000
        elif ENV_NAME == 'CRAWLER':
            max_episode_length = 1000
        else:
            raise NotImplementedError()

        # act in the environment
        for i_times in range(max_episode_length):
            # take action in environment and get its response
            action, prediction = act_in_environment(agent, state)

            env_info = env.step(action)[brain_name]
            next_observed_state = env_info.vector_observations
            done = env_info.local_done
            observed_reward = env_info.rewards

            if ENV_NAME == 'REACHER':
                # The reward given was always ~ 0.02 which is not what the environment description explained.
                # So I changed it to 0.1 if it's greater than 0. Also stated in the Udacity peer chat:
                # https://hub.udacity.com/rooms/community:nd893:845401-project-503-smg-2?contextType=room
                observed_reward = [0.1 if rew > 0 else 0 for rew in observed_reward]

            if TRAIN_MODE:
                train_agent(action, agent, done, next_observed_state, observed_reward, prediction, state)

            state = next_observed_state

            # collect rewards
            if agent.num_agents == 1:
                score += observed_reward[0]
            else:
                score += np.mean(observed_reward)

            if np.all(done):
                break

        # save the obtained scores
        scores_window.append(score)
        scores.append(score)
        score_mean = np.mean(scores_window)
        score_mean_list.append(score_mean)

        score_max, saved_earliest_agent = plot_and_save_agent(agent, i_episode, score_max, scores, score_mean, score_mean_list, score, saved_earliest_agent)


def train_agent(action, agent, done, next_observed_state, observed_reward, prediction, state):
    """
    Perform a training step on the agent
    :param state:               np.array, current state of the environment
    :param action:              np.array, action taken in the environment
    :param agent:               acting agent(s)
    :param done:                np array, indicator of one or more of the agent in the environment terminated
    :param next_observed_state: np.array, next state of the environment
    :param observed_reward:     np.array, observed_reward for the action taken in the current state
    :param prediction:          np.array or dict, full output of the actor network (for A2C it'S combined with the critic output
    """
    if type(agent) == DDPGAgent:
        agent.step(state, action, next_observed_state, observed_reward, done)
    elif type(agent) == A3CAgent:
        agent.step(state, prediction, observed_reward, done)
    else:
        raise NotImplementedError


def act_in_environment(agent, state):
    """
    Let the agent produce and action to act in the state.
    :param agent: agent to act
    :param state: state to act in
    :return: (action, full network prediction in case of a combined Network)
    """

    if type(agent) == DDPGAgent:
        action = agent.act(state, add_noise=TRAIN_MODE)
        prediction = action
    elif type(agent) == A3CAgent:
        prediction = agent.act(state)
        action = prediction['action'].cpu().detach().numpy()
    else:
        raise NotImplementedError

    return action, prediction


def plot_and_save_agent(agent, i_episode, score_max, scores, scores_mean, score_mean_list, score, saved_earliest_agent):
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
    #if i_episode % 10 == 0 and TRAIN_MODE:
    #    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores_mean), end="\n ")
    #else:
    #    print('\rEpisode {}\tScore for this episode {:.2f}:'.format(i_episode, score), end="")
    print('\rEpisode {}\tScore for this episode {:.2f}\tApplied noise {:.5f}:'.format(i_episode, score, agent.network.std[0]), end="")
    if i_episode % 10 == 0 and TRAIN_MODE:
        print('\nEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores_mean))
        save_score_plot(scores, score_mean_list, i_episode)

    if scores_mean >= score_max + SAVE_EACH_NEXT_BEST_REWARD and TRAIN_MODE:
        agent.save_current_agent(score_max, scores, score_mean_list, i_episode)
        score_max += SAVE_EACH_NEXT_BEST_REWARD
        print('\nSaved agent with reward: {:.2f}\t after {:d} episodes!'.format(scores_mean, i_episode))

    if i_episode % 100 == 0 and TRAIN_MODE:
        agent.save_current_agent(score_max, scores, score_mean_list, i_episode)

    if scores_mean >= NEEDED_REWARD_FOR_SOLVING_ENV and TRAIN_MODE and not saved_earliest_agent:
        agent.save_current_agent(score_max, scores, score_mean_list, i_episode)
        saved_earliest_agent = True
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} '.format(i_episode, scores_mean))

    return score_max, saved_earliest_agent


if __name__ == "__main__":
    main()
