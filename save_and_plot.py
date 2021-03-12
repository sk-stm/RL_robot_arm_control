import os
from typing import List
import datetime
import matplotlib.pyplot as plt
import numpy as np
from environment import AGENT_TYPE, ENV_NAME


def create_folder_structure_according_to_agent():
    """
    Creates a folder structure to store the current experiment according to the type of agent that was run and the
    current date and time.

    :param agent: Agent to be stored
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    new_folder_path = os.path.join(f'{AGENT_TYPE}_{ENV_NAME}', f'{date_str}')
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