import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# NOTE: Inspiration of the concept was taken from: https://www.youtube.com/watch?v=MqZmwQoOXw4&t=503s


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, probability_exponent):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffersize = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.probability_exponent = probability_exponent

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        Since the new example is new, assign it with highest priority.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(max(self.priorities, default=1))

    def calc_probabilites(self):
        """
        Converts priorities to probabilities according to: priorities^a / sum (all priorities^a)
        :param a: factor to steer the priorisation. a=0: uniform picking of experiences
                                                    a=1: picking experiences according to priorities
        :return: probabilities for all experiences
        """
        scaled_priorities = np.array(self.priorities) ** self.probability_exponent
        probablilities = scaled_priorities / sum(scaled_priorities)
        return probablilities

    def set_priorities(self, indices, errors, regularization=0.01):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + regularization

    def get_importance_weights(self, probabilities):
        importance_weights = 1/self.buffersize * 1/probabilities
        normalized_importance_weights = importance_weights / sum(importance_weights)
        return normalized_importance_weights

    def sample(self):
        """Sample a batch of experiences from memory according to priorities."""
        experience_probabilitires = self.calc_probabilites()
        experience_indices = random.choices(range(len(self.memory)), k=self.batch_size, weights=experience_probabilitires)
        experiences = np.array(self.memory)[experience_indices]

        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        importance_weights = self.get_importance_weights(experience_probabilitires[experience_indices])

        return (states, actions, rewards, next_states, dones), experience_indices, importance_weights

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)