# Start of project

Implementation is inspired from [this nice repo](https://github.com/ShangtongZhang/DeepRL/blob/master/examples.py#L384).

I use an Advatage actor Critic agent. And simple storage between the runs.
The agent runs a couple of steps collecting data with the current policy.
data = (reward, next_state, done)

To take a step the current version of the AC network is used pro produce a prediction and action for the given state:
preiction = (action to take, mean of the action value, logarithmic probability of the density function
that describes the action value (here normal distribution), entropy of that distribution,
 critic evaluation (value) of the current state).

The difference between actual reward and expected reward is used to update the critic, and the reverse cumulated reward
is used to update the probability of the action to be chosen in that state the next time by the critic.

The experience that the agent gained over time is used to update the neural networks (to choose a better action and evaluate the
state better) and then thrown away. -> The NN is the entity that keeps track of all the states and actions that work well.

TODO: implement a correct evaluation mechanism to see if the agent really makes progress.

# Experiements:
## 1
number_of_actor_runs = 5
num_workers = 1
discount_factor = 0.99
gradient_clip = 0.1
entropy_weight = 0.01
value_loss_weight = 1.0

-> constant rewards of 0. I think the agent doesnt find good action to chose because the time to interact with the world is
too small. + The std of the action distribution is 0, which might mess up the entropy, maybe a very small value is better.
It might also be necessary to learn the std as well.

## 2
Hyper parameter tuning doesn't work at all. The agent seems not to learn anything. This means back to the drawing board
and rethink the algorithm and search for bugs in the implementation.

# Conclusion 1

It seems that the training runs are not independent anymore. Therefore the learning of the critic is prone to
divergence. It's better to use a replay buffer to iid sample the input data from experience just like in the
banana project. Also: Exploration is not done correctly, according to the [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf)
there must be a noise term added to the action that is selected in order to also do other actions and explore more.
Else it just takes for ever, which is exactly what can be seen by current experiments.

# Next try
After watching [this](https://www.youtube.com/watch?v=08V9r3NgFSE&t=311s) I encapsulated the action sampling in a
no_Grad function in numpy, so that the action sampling doesn't create a gradient, which would mess up the whole
backprob according to the loss.
Also I changed the network of the critic to only use the state as input and not state+action concatenated. The action
concatenated to the second fc layer as also done in the [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf).
Still the agent doesn't get better at the task.

## Exp 1
GAMMA = 0.999
BATCH_SIZE = 64
BUFFER_SIZE = int(1e4)
TAU = 0.01
theta = 0.15
mu = 0
dt = 1e-2
std = 0.2

-> rewards after 240 runs: 0.57

## Exp 2
GAMMA = 0.999
BATCH_SIZE = 64
BUFFER_SIZE = int(1e4)
TAU = 0.1
theta = 0.15
mu = 0
dt = 1e-2
std = 0.02
self.actor_optimizer = optim.Adam(self.local_actor_network.parameters(), lr=0.03)
self.critic_optimizer = optim.Adam(self.local_critic_network.parameters(), lr=0.05)
Episode length = 10000

->Episode 200	Average Score: 0.47

## Exp3
same as Exp2 but with only 500 steps long episodes.

-> rewards are in general smaller but also increasing here and there.. hard to judge

## Exp 4
GAMMA = 0.999
BATCH_SIZE = 64
BUFFER_SIZE = int(1e4)
TAU = 0.1
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
WEIGHT_DECAY = 0.0001
theta = 0.15
mu = 0
std = 0.02

-> Episode 200	Average Score: 0.11;
 Episode 310	Average Score: 0.14
 still stalling also every second episode has a return of 0.0 for some reason.
Smalls like a bug.

## Exp 5
GAMMA = 0.999
BATCH_SIZE = 64
BUFFER_SIZE = int(1e4)
TAU = 0.01
LR_ACTOR = 1e-3
LR_CRITIC = 3e-3
WEIGHT_DECAY = 0.0001
theta = 0.15
mu = 0
std = 0.2

-> Episode 930	Average Score: 0.12
still not converging
Maybe the training is not working at all and all I'm seeing is just the noise of the taken actions.
I have to check if the gradient is calculated correctly or of it's corrupted somewhere. It certainly seems like it.

## Exp 6
Included the detach function for target_critic_value_for_next_state.
Also: Added code that the target net update happens only all UPDATE_EVERY steps. Maybe the problem is that
the target is too close to the local and therefore no real loss is created and the local networks can't evolve.

-> Episode 430	Average Score: 0.55
still not converging

## Exp 7
Didn't set back the actor network into training mode so it was all the same network as from the beginning to act
No wonder it didn't improve.

-> performance seems to increase!! very slowly but it raises. I think I got it or at least am close now! :D
Next step: find good parameters

## Exp 8
Exp. length: 1000
GAMMA = 0.999
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 1e-3
LR_CRITIC = 3e-3
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 10
std = 0.1

-> Episode 1370	Average Score: 0.41; still not converging