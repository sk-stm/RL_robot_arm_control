# REINFORCEMENT LEARNING FOR UNITY REACHER PROJECT

## Project description

An agent has to reach a moving target position in an simulated environment.
The objective is to keep the end effector of the robot arn inside the designated area for al long as possible.

The environment is written in UNITY which can be observed through a continuous 37 dimensional statespace.
The statespace contains information about the position of the agent arm in the environment, and about target.
Thus the world is completely observed by the agent.
Additionally the world keeps changing over time the target moves in a circle around the agent.

The agent can perform 4 continuous actions, that are applying a torque of (-1, 1) to each of it's 4 joints.

When the end effector of the agent is inside the target area, it get's a reward of 0.1, else a reward of 0 is given.

*NOTE*: In the linux environment the reward given by the environment is different from the reward in the documentation,
so I artificially changes the reward to what it should have been. It would be nice to eliminate that bug for this
environment in future versions.

## Project goal
The environment is considered solved if the agent get an **average reward of >=30** in 100 consecutive runs.

## Installation

1. Create virtual python environment and source it:
    - `python3.6 -m venv p1_env`
    - `source p1_env/bin/activate`
    - `pip install -U pip`
2. Clone repository:
    - `git clone https://github.com/sk-stm/RL_robot_arm_control.git`
    - `cd RL_robot_arm_control`
    - `pip install -r requirements.txt`
3. Follow the instructions to perform a minimal install of the environment for you system in https://github.com/openai/gym#installation
4. Select the environment according to your system:
    - Set the `ENV_PATH` variable in `main.py` to the path to the file that holds the appropriate environment for your system.
    You can download your specific environment:
        - Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Max OS: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows 32Bit: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows 64Bit: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## Run training:
1. Set the `TRAIN_MODE` variable in `main.py` to `True`
2. Set `LOAD_MODEL = False` in `main.py` if you want to start training from scratch
3. In `environment.py` uncomment the environment specific block of code you want to run
    1. If `LOAD_MODEL == True`, set the path to the network parameters you want to load to the varaible
    `MODEL_TO_LOAD` in `environment.py`.
4. Run main.py and the agent start training in the environment.

## Run inference:
1. Set the `TRAIN_MODE` variable in `main.py` to `False`
2. Set `LOAD_MODEL = False` in `main.py`
3. In `environment.py` uncomment the environment specific block of code you want to run
4. In `environment.py` set the path to the network parameters you want to load in the `MODEL_TO_LOAD` variable.
5. Run main.py and the agent start training in the environment.
