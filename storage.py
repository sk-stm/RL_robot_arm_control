class Storage:

    def __init__(self):
        self.action = []
        self.log_prob_a = []
        self.entropy = []
        self.mean_a = []
        self.critic_value = []
        self.reward = []
        self.done = []
        self.advantage = []
        self.returns = []

    def empty(self):
        self.action.clear()
        self.log_prob_a.clear()
        self.entropy.clear()
        self.mean_a.clear()
        self.critic_value.clear()
        self.reward.clear()
        self.done.clear()
        self.advantage.clear()
        self.returns.clear()
