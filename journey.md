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

## Exp 9
GAMMA = 0.999
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 1e-3
LR_CRITIC = 3e-3
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5
NOISE_REDUCTION_FACTOR = 0.9999

-> no converge

## Exp 10
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 1e-3
LR_CRITIC = 3e-3
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5
NOISE_REDUCTION_FACTOR = 0.9999

-> diverged

## Exp 11
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
TAU = 1e-1
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 5
NOISE_REDUCTION_FACTOR = 0.9999
theta = 0.15
mu = 0
sigma = 0.1

-> works! now let's wait


# Crawler

## Exp 1
GAMMA = 0.99
GRADIENT_CLIP = 0.1
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-2
NOISE_REDUCTION_FACTOR = 0.9999
LEARNING_RATE = 0.0003

-> up till 13 looks reasonable but diverged,
 with high exploration in the beginning to understand that is needed -> refine

## Exp 2
GAMMA = 0.99
GRADIENT_CLIP = 0.01
ENTROPY_WEIGHT = 0.01
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 10
NOISE_ON_THE_ACTIONS = 1e-2
NOISE_REDUCTION_FACTOR = 0.9999
LEARNING_RATE = 0.0003

-> nicht über 12 ... divergiert
-> besser weniger Noise hohe update raten

## Exp 3
GAMMA = 0.99
GRADIENT_CLIP = 0.1
ENTROPY_WEIGHT = 0.01
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 10
NOISE_ON_THE_ACTIONS = 1e-5
NOISE_REDUCTION_FACTOR = 0.9999
LEARNING_RATE = 0.003

-> diverges from the beginning

## EXP 4
GAMMA = 0.99
GRADIENT_CLIP = 0.1
ENTROPY_WEIGHT = 0.01
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 10
NOISE_ON_THE_ACTIONS = 1e-5
NOISE_REDUCTION_FACTOR = 0.9999
LEARNING_RATE = 0.0003

-> rumgetanze, max 12 reward, dann diverged ~7
Sieht teilweise aber schon nach krabbeln aus

## Exp 5 Crawler

GAMMA = 0.99
GRADIENT_CLIP = 0.01
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-3
NOISE_REDUCTION_FACTOR = 0.9999
LEARNING_RATE = 0.0003

-> rumgetanze, sehr noisy rewards, wird nicht viel besser.
Vll schnellere noise reduction

## Exp 6

GAMMA = 0.99
GRADIENT_CLIP = 0.01
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-3
NOISE_REDUCTION_FACTOR = 0.999
LEARNING_RATE = 0.00003

-> learning doesn't start off as fast

## Exp 7

GAMMA = 0.99
GRADIENT_CLIP = 0.01
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-3
NOISE_REDUCTION_FACTOR = 0.9999
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-5

-> learning not starting really -> reduce learning rate
TODO maybe have a different learning rate for both networks?

# Exp 8

GAMMA = 0.99
GRADIENT_CLIP = 0.01
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-3
NOISE_REDUCTION_FACTOR = 0.9999
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> learning starts off great,  then stalls -> reduce noise faster


# Exp 9

GAMMA = 0.99
GRADIENT_CLIP = 0.01
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-3
NOISE_REDUCTION_FACTOR = 0.999
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> learning looks the same as Exp 8 -> larger gradient clip to exploit good actions more

# Exp 10

GAMMA = 0.99
GRADIENT_CLIP = 0.1
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-3
NOISE_REDUCTION_FACTOR = 0.999
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> less noise but still stalls -> larger gradient clip to exploit good slutions even more

# Exp 11

GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-3
NOISE_REDUCTION_FACTOR = 0.999
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> raising higher rewards but similar as before but with slightly more noise
if it stalls maybe longer noise application is needed
-> stalls and is noise as always -> more noise at the beginning

# Exp 12

GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-1
NOISE_REDUCTION_FACTOR = 0.999
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> same as before -> decay noise a bit less

# Exp 13

GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-1
NOISE_REDUCTION_FACTOR = 0.99999
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> stagnation at roughly the same level as before with similar noise
-> maybe higher learning rate to get out of local optimum?
Or smaller learning rate with higher noise to settle more stable in an optimum?
-> HAD POTENTIAL. Was killed by accident too early maybe


# Exp 14

GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-1
NOISE_REDUCTION_FACTOR = 0.99999
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-5

-> still noisy and no real success. It's ok in the beginning and then stagnates after ~400 steps. It's unclear if the
agent can't find a reasonable way to explore the environment to understand that forward is better, or if it resides
in a local optimum. The noise applied is still high ~0.09 after 2000 episodes. It would be nice to find a way to
keep training past the 400th episode.
-> I saw in https://github.com/ShangtongZhang/DeepRL/blob/master/examples.py#L554 that the gradient clip is set to 5!
not 0.x So let'S try that as well and reduce the noise a bit faster

# Exp 15

GAMMA = 0.99
GRADIENT_CLIP = 5
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-1
NOISE_REDUCTION_FACTOR = 0.9999
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-5

-> doesn't learn at all no improvement just noise -> reduce clipping a bit and reduce learning rate

# Exp 16

GAMMA = 0.99
GRADIENT_CLIP = 2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-1
NOISE_REDUCTION_FACTOR = 0.9999
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> same as always... no idea. Just random pick longer between learning


# Exp 17

GAMMA = 0.99
GRADIENT_CLIP = 2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 20
NOISE_ON_THE_ACTIONS = 1e-1
NOISE_REDUCTION_FACTOR = 0.9999
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> one sees a learning kaing place very slowly try out previous best params and let it run over night

# Exp 18

GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-1
NOISE_REDUCTION_FACTOR = 0.99999
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> increases slowly but doesn't find the right approach for the long run. After a day training
 still it looks reasonable in inference, so I use it as baseline to initialize a new training

# Exp 19

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-5
NOISE_REDUCTION_FACTOR = 0.999
LEARNING_RATE = 0.00003
WEIGHT_DECAY = 1e-5

-> small improvement to a reward of ~ 20 maybe because the noise + smaller learning rate kicked the exploration again?

# Exp 20

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 5
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-4
NOISE_REDUCTION_FACTOR = 0.999
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> small improvement to a reward of ~ 20 maybe because the noise + smaller learning rate kicked the exploration again?
let's try a smaller leaning rate with no noise. If this works better in the long run, it might be that the noise is
preventing finding the fine movement.

# Exp 21

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 5
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.000003
WEIGHT_DECAY = 1e-5

-> slow increase but still much noise -> no noise, higher learning rate

# Exp 22

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 5
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-7

-> same as before. I would like to reduce the noise of the actions (maybe the entropy needs to be reduced for that)
That is, give the entropy a higher weight

# Exp 23

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 5
ENTROPY_WEIGHT = 0.5
VALUE_LOS_WEIGHT = 1.0
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> after 7000 iterations, it looks similar to Exp 21. Maybe also reduce the weight of the value (critic)
Ok here is another plan:
1. start with low entropy weight and high value weight, so a large space of actions will be explored.
and a good understanding of the environment is created (what was done already)
2. After some time gradually reduce the value weight and increase the entropy weight of the loss, so that the action
that is chosen has more influence on the loss and therefore the weights are chosen more carefully for the actions
 -> leading to more stable actions
 + find a way to reduce the impact of the loss to the critic when it's good

# Exp 24

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 5
ENTROPY_WEIGHT = 0.7
VALUE_LOS_WEIGHT = 0.001
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

# Exp 25

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 5
ENTROPY_WEIGHT = 0.7
VALUE_LOS_WEIGHT = 0.001
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> stays the same as before. Still much noise in the actions. The action loss is also changing a lot.

# Exp 26

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 0.02
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 20
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5

-> same as before. Trying now to use GAE (generalized advantage estimation) to keep the variance of the TD error
under control. Maybe also a longer N-step bootstraping would do the trick.

# Exp 27

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 20
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95

-> Sames as before no real improvement. -> Try out lass long N-step boot straping to reduce the variance of the return

# Exp 28

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 20000
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95

-> same same ..

# Exp 29

Pretrained model parameters: /home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/A3C_CRAWLER/day_long_trained_model/checkpoint_0.pth
GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 200
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95


-> same same. Maybe start with the fresh initialized net and smaller noise to begin with

# Exp 30

GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 80
NOISE_ON_THE_ACTIONS = 1e-7
NOISE_REDUCTION_FACTOR = 0.99
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95

-> same same but different
Found the bug!! The break condition was set to if any of the agents terminated, that way all the agent had to work
at the same time else all agents were interrupted. That of course non sense and lead to agent trying to gain as
much reward at the beginning as possible. -> jumping

# Exp 31

GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-7
NOISE_REDUCTION_FACTOR = 0.99
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95

-> learn fast and reasonable ~ reward of 200 after 40 iterations
-> nan after some time. According to internet that's because the gradient or the action become out of bound

# Exp 32

GAMMA = 0.99
GRADIENT_CLIP = 0.1
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-7
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95

-> still nan after 800 episodes and degration / jumping before that -> more graient clipping?

# Exp 33

GAMMA = 0.99
GRADIENT_CLIP = 0.05
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-7
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95

-> still Nan after 10 iterations. -> included reaky relu
-> still the same

# Exp 34

GAMMA = 0.99
GRADIENT_CLIP = 0.01
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-7
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.00003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95

-> nan after 400 iterations.

# Exp 35

Pretrained model parameters: A3C_CRAWLER/2021_03_24_10_46_46/checkpoint_200.pth
GAMMA = 0.99
GRADIENT_CLIP = 0.01
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 3
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.00003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95

-> pretty noisy, nan after ~ 600 iterations so better?

# inbetween exp:
LEARNING_RATE = 0.03 -> diverging in the begining ~10 iterations
LEARNING_RATE = 0.003 -> not diverging but weired local minimum at ~40 iterations

# Exp 36

GAMMA = 0.99
GRADIENT_CLIP = 0.1
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-3
NOISE_REDUCTION_FACTOR = 0.99
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95
MAX_EPISODE_LENGTH = 500

-> slower learning w.r.t Exp 31, due to smaller gradient clipping let's see where it ends.
-> Maybe larger gradient cliping in the begining and smaller later on... again another paramter...
thought: maybe the agent learns to jump again so much because at the end of an episode it has to reach as far
as possible so extending the MAX_EPISODE_LENGTH might be another step for optimization

nan error trace:
```
Episode 698	Score for this episode 254.07	Applied noise 0.00000:[W python_anomaly_mode.cpp:104] Warning: Error detected in PowBackward0. Traceback of forward call that caused the error:
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/main.py", line 227, in <module>
    main()
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/main.py", line 32, in main
    run_environment(brain_name, agent)
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/main.py", line 123, in run_environment
    train_agent(action, agent, done, next_observed_state, observed_reward, prediction, state)
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/main.py", line 159, in train_agent
    agent.step(state, prediction, observed_reward, done)
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/a3c_agent.py", line 65, in step
    self.learn(state)
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/a3c_agent.py", line 101, in learn
    value_loss = 0.5 * (return_tensor - critic_value_tensor).pow(2).mean()
 (function _print_stack)
Traceback (most recent call last):
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/main.py", line 227, in <module>
    main()
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/main.py", line 32, in main
    run_environment(brain_name, agent)
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/main.py", line 123, in run_environment
    train_agent(action, agent, done, next_observed_state, observed_reward, prediction, state)
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/main.py", line 159, in train_agent
    agent.step(state, prediction, observed_reward, done)
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/a3c_agent.py", line 65, in step
    self.learn(state)
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/RL_robot_arm_control/a3c_agent.py", line 107, in learn
    (policy_loss + ENTROPY_WEIGHT * entropy_loss + VALUE_LOS_WEIGHT * value_loss).backward()
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/p2_env_36/lib/python3.6/site-packages/torch/tensor.py", line 221, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/home/shinchan/Projekte/Reinforcement_learning/Udacity/project_1/deep-reinforcement-learning/p2_continuous-control/p2_env_36/lib/python3.6/site-packages/torch/autograd/__init__.py", line 132, in backward
    allow_unreachable=True)  # allow_unreachable flag
RuntimeError: Function 'PowBackward0' returned nan values in its 0th output.

Process finished with exit code 1
```

**Seems an error in MSE. This is expected since I read about it in one of the intros to pytorch losses -> use the
pytorch internal MSE instead.**

# Exp 37

GAMMA = 0.99
GRADIENT_CLIP = 0.2
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-5
NOISE_REDUCTION_FACTOR = 0.99
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95
MAX_EPISODE_LENGTH = 500

-> didn't work. Still nan...


# Exp 38

GAMMA = 0.99
GRADIENT_CLIP = 0.1
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 1e-5
NOISE_REDUCTION_FACTOR = 0.99
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-5
USE_GAE = True
GAE_TAU = 0.95
MAX_EPISODE_LENGTH = 500

-> looks good until nan happened. So next will be fine tune from this position + less weight decay and smaller learning rate.

# Exp 39

MODEL_TO_LOAD = 'A3C_CRAWLER/Exp_38_best_model/checkpoint_230.pth'
GAMMA = 0.99
GRADIENT_CLIP = 0.1
ENTROPY_WEIGHT = 0.001
VALUE_LOS_WEIGHT = 1
ACTIONS_BETWEEN_LEARNING = 5
NOISE_ON_THE_ACTIONS = 0
NOISE_REDUCTION_FACTOR = 1
LEARNING_RATE = 0.00003
WEIGHT_DECAY = 1e-7
USE_GAE = True
GAE_TAU = 0.95
MAX_EPISODE_LENGTH = 1000

-> der Fahler wird sein, dass der Reward einer Episode von vorne beginnen muss wenn der Agent hingefallen ist
Momentan wird einfach weitergerechnet. Das irgendwie auch zum Ziel führen kann, aber nicht ganz so sauber und direkt.
Wahrscheinlich müsste man doch alle Agenten der Env in separaten Prozessen einzeln abfertigen und dann geballt auf
der collecteted Erfahrung trainieren. Meinetwegen mit batches. Die frage ist dann nur, wie füllt man die batches, die
Vorzeitig terminiert sind?