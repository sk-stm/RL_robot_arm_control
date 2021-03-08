# Learning algorithm
The learning algorithm used was DDPG (deep deterministic policy gradient).
The agent acts in the environment to create an understanding of what actions create the most rewards in what states.
The agent perceives the state of the environment and acts according to a policy. For each action it takes,
it receives an reward and observes the next states it transitions to. Also it observes if the episode ended or not.

Each observation (state, action, reward, next state, done) is stored in a list, the replay memory. Once the memory
contains enough entries, a random sample of mini batch size is taken from the memory. These samples are used to train
two function approximators, in our case fully connected neural networks, specifically MLPs.

One of the two networks is an actor network, getting the environment state as input and produces 4 continuous outputs
in the range of (-1, 1).
**The structure of the actor network is:**
```
self.actor_fc1 = nn.Linear(state_size, 400)
self.actor_fc2 = nn.Linear(400, 300)
self.actor_fc3 = nn.Linear(300, action_size)
```
With ReLU non-linearities after fc1 and fc2 and a tanh after fc3.

The other is a critic network that evaluates the states of the agent and the actions the agent took.

**The critic network has the following structure:**
```
self.critic_fc1 = nn.Linear(state_size, 400)
self.critic_fc2 = nn.Linear(400 + action_size, 300)
self.critic_fc3 = nn.Linear(300, 1)
```
Note that in the only in the second fc layer the action is added to the state. This was also done by
[the DDPG paper](https://arxiv.org/pdf/1509.02971.pdf)
The goal is to approximate a function that maps states to values and with this knowledge steer the actor to
 chose actions that lead to rewarding states.

These networks (local networks) learn to act and judge the states and actions that the agent takes.
This procedure is repeated for UPDATE_EVERY (hyper parameter) steps, so the agent has acted in the environment
for some time.
After that, the learning of the networks takes place with i.i.d sampled states from the replay buffer.
 Then the parameters of the locel networks are copied to a second pair of networks with the same structure
 (traget networks).
The target networks are networks that parameters are kept constant during the training. By copying the parameters
from the local networks to the target networks, they capture the current state of the overall training. These networks
are then used to estimate the total discounted reward for the next state and propose the next best action respectively.

That is, in case of the critic:
For each state the value of the next state is estimated to approximate the value of the current state a little
bit better. Why is that necessary?

The local critic network is trained in a supervised manner. That is, for each state there are target values that
the network aims towards. They can be used to create a loss for the network to optimize its parameters.

Specifically this target is the estimated total discounted reward for the next state from the target network:
if the episode ended, the target == reward, else the target is:
targets = rewards + (gamma * best_qtarget_value)

Where best_qtarget_value is the best value for the next state that the target network contains. And `gamma == 0.99` is the
discount factor for using the estimated target value. This is useful because the target is only estimated and also changes
over time. The higher `gamma`, the more it is used to update. The lower gamma, the more the reward i trusted, and the less
the estimated value of the next state is used. The target state and action pair value is created by the target networks
and the prediction that is driven towards that target is created by the local networks. That way the prediction of each
state is improved evey step.

The actor network is trained with a loss created by the critic network that judges the picked action of the actor network.
If the action is evaluated positively by the critic, the loss is negative and if the action wis not evaluated well,
the loss is positive. This might be confusing but the pytorch library performs `gradient decent` and therefore tries to
minimize the loss. Therefor the smaller the loss is the better is the result.

Both networks are trained with the Adam optimizer, which is a standard first good choice of optimizers because it inherently
accounts for the update momentum and other update properties. The **learning rate** was set to `LR=5e-4` for both
actor and critic.
The update step of the parameters of the target networks is done softly. That is, the parameters are not just copied
over from the local networks, but are softly updated according to `TAU==1e-1`. That means that the old parameters of
the target networks are preserved by a factor of `1-TAU` and updated by the local networks parameters by `TAU==1e-1`.
This way the target networks don't change too quickly and the target state-action values stay different from the local
state-action values, to create a meaningful loss. The **batch size** for training samples was set to `BATCH_SIZE==64`


So with each learning step the local networks approximates the current state and actions a little bit better because
it is driven towards the reward + the expected reward at the next state according to the target networks.
The target networks are updated with the better approximation the local networks gained after some steps,
by copying their parameters.

This is repeated until a good approximation of state-action values and actions is achieved.

The actions the agent choses to interact with the environment are chosen according to the actors networks. To ensure
exploration some noise is added to the action. The noise is chosen according to the Ornstein-Uhlenbeck-process.
This process is a good fit because it preserves the initial movement and adds some noise to it. It's leaned from
 physics where it models the velocity of a massive Brownian particle under friction. The parameters chosen for this
 process are:
 ```
THETA = 0.15
MU = 0
SIGMA = 0.1
 ```
 where `SIGMA` was lowered with each acting in the environment by some amount. Specifically it was multiplied by a
 `NOISE_REDUCTION_FACTOR = 0.9999`.
 Therefore the noise encourage random exploration of actions at the start and use the best actions later during the
 training.

The rewards received by an agent of this type can be shown in this figure:
# TODO
![Best performance over all](DQN/best_model_overall/score_plot_1992.jpg)

# TODO
This agent was trained 2000 episodes and reached an average reward of 15.39.

The reward > 30 was achieved after 1110 episodes. The next figure shows the learning process of that agent.
![Earlies solution to the environment](file:///home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/DDPG/2021_03_08_15_32_53/score_plot_1110.jpg)

# Future ideas to improve performance
To improve the performance and make the agent train faster, one could tune the parameters a little bit more. It's clear
to see that the training starts off very slowly which could be because the noise applied in the beginning is too high,
preventing a stable search of good states. Furthermore the Learning rate and batch size could be increased to make learning
more efficient. If one does that, the replay memory size should be increased as well.

To improve performance beyond parameter tuning it would be nice to try out other actor critic methods or policy optimization techniques like
A2C, PPO. Both are very popular algorithms that might perform well on this kind of problem. It would also be nice to
explore the machanics of for example A2C (advantage actor critic) because the action creation work fundamentally
different to DDPG. Also TRPO (Trust Region Policy Optimization anr TNPG (Truncated Natural Prolicy Gradient) look
promising according to [this paper](https://arxiv.org/pdf/1604.06778.pdf).

 One could also apply prioritized experience replay which should make more the agent learn more efficiently in the
 regions it's unsure yet. And therefore contribute to a faster convergence.

# Best performace parameters for DQN:
# LEARNING PARAMETERS
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5

# NOISE PARAMETERS
NOISE_REDUCTION_FACTOR = 0.9999
THETA = 0.15
MU = 0
SIGMA = 0.1