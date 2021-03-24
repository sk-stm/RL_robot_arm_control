import torch

# ENV_PATH = '../Reacher_Linux_single/Reacher.x86'
# ENV_NAME = 'REACHER'
# MODEL_TO_LOAD = 'DDPG/best_model/checkpoint_97.0.pth'
# AGENT_TYPE = 'DDPG'
# NEEDED_REWARD_FOR_SOLVING_ENV = 30

# ENV_PATH = '../Reacher_Linux_multi/Reacher.x86'
# MODEL_TO_LOAD = 'A2C/earliest_model/checkpoint_30.64.pth'
# AGENT_TYPE = 'A3C'
# ENV_NAME = 'REACHER'
# NEEDED_REWARD_FOR_SOLVING_ENV = 30

ENV_PATH = '../Crawler_Linux/Crawler.x86'
MODEL_TO_LOAD = 'A3C_CRAWLER/2021_03_24_00_21_50/checkpoint_300.pth'
AGENT_TYPE = 'A3C'
ENV_NAME = 'CRAWLER'
NEEDED_REWARD_FOR_SOLVING_ENV = 2000



# DEVICE PARAMETERS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_EACH_NEXT_BEST_REWARD = 100
