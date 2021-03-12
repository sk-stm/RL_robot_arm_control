import torch

# Reacher
# LEARNING PARAMETERS
# GAMMA = 0.99
# BATCH_SIZE = 64
# BUFFER_SIZE = int(1e5)
# TAU = 1e-1
# LR_ACTOR = 5e-4
# LR_CRITIC = 5e-4
# WEIGHT_DECAY = 0.0001
# UPDATE_EVERY = 5
#
# # NOISE PARAMETERS
# NOISE_REDUCTION_FACTOR = 0.9999
# THETA = 0.15
# MU = 0
# SIGMA = 0.1
#
# # DEVICE PARAMETERS
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#DDPG CRAWLER
# LEARNING PARAMETERS
# GAMMA = 0.99
# BATCH_SIZE = 64
# BUFFER_SIZE = int(1e5)
# TAU = 1e-1
# LR_ACTOR = 5e-4
# LR_CRITIC = 5e-4
# WEIGHT_DECAY = 0.0001
# UPDATE_EVERY = 5
#
# # NOISE PARAMETERS
# NOISE_REDUCTION_FACTOR = 0.99999
# THETA = 0.15
# MU = 0
# SIGMA = 0.2
#
# # DEVICE PARAMETERS
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# A2C Reacher
# TODO remove those dependencies
THETA = 0.15
MU = 0
SIGMA = 0.2
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0001
NOISE_REDUCTION_FACTOR = 0.99999

GAMMA = 0.99
GRADIENT_CLIP = 0.01
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
UPDATE_EVERY = 5
ROLLOUT_LENGTH = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")