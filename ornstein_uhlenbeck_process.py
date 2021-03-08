import numpy as np
from PARAMETERS import THETA, MU, SIGMA


class OrnsteinUhlenbeckProcess:

    def __init__(self, action_size):
        self.prev_x = self.prev_x = np.zeros(action_size)
        self.action_size = action_size

    def sample(self):
        x = self.prev_x + THETA * (MU + self.prev_x) + SIGMA * np.random.randn(*(self.action_size,))
        self.prev_x = x
        return x

    def reset_process(self):
        self.prev_x = np.zeros(self.action_size)
