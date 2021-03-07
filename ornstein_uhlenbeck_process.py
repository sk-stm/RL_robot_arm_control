import numpy as np


class OrnsteinUhlenbeckProcess:

    def __init__(self, action_size):
        self.prev_x = self.prev_x = np.zeros(action_size)
        self.action_size = action_size

    def sample(self):
        theta = 0.15
        mu = 0
        # TODO write linear schedule
        std = 0.1
        x = self.prev_x + theta * (mu + self.prev_x) + std * np.random.randn(*(self.action_size,))
        self.prev_x = x
        return x

    def reset_process(self):
        self.prev_x = np.zeros(self.action_size)
