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
banana project. Also: Exploration is not done correctly, according to (DDPG)[https://arxiv.org/pdf/1509.02971.pdf]
there must be a noise term added to the action that is selected in order to also do other actions and explore more.
Else it just takes for ever, which is exactly what can be seen by current experiments.

