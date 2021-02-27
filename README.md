# REINFORCEMENT LEARNING FOR UNITY BANANA PROJECT

## Project description

An agent has to navigate and act in an simulated environment consisting of a room filled with yellow and blue bananas.
The objective is to write a program that navigates the agend through the environment collecting yellow banans.

The environment is written in UNITY which can be observed through a continuous 37 dimensional statespace. The statespace contains information about the position of the agent in the environment, and about object. in its field of view. Thus the world is at agiven time only partially observed by the agent.
Additionally the world keeps changing over time as new bananas drop inside the world from time to time and the agent removes / collects bananas from the environment when it moves near them.

 The agent can perform 4 discrete actions:

- 0 - walk forward
- 1 - walk backward
- 2 - turn left
- 3 - turn right

When the agent moves near a banana this banana is collected and will give a reward of `+1` if the banana is yell and `-1` if the banana is blue.

## Project goal
The environment is considered solved if the agent get an **average reward of >=13** in 100 consecutive runs.

## Installation

1. Create virtual python environment and source it:
    - `python3.6 -m venv p1_env`
    - `source p1_env/bin/activate`
    - `pip install -U pip`
2. Clone repository:
    - `git clone https://github.com/sk-stm/banana_project.git`
    - `cd banana_project`
    - `pip install -r requirements.txt`
3. Follow the instructions to perform a minimal install of the environment for you system in https://github.com/openai/gym#installation
4. Select the environment according to your system:
    - Set the `ENV_PATH` variable in `main.py` to the path to the file that holds the appropriate environment for your system.
    You can download your specific environment:
        - Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
        - Max OS: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
        - Windows 32Bit: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
        - Windows 64Bit: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Run training:
1. Set the `TRAIN_MODE` variable in `main.py` to `True`
2. Run main.py and the agent start training in the environment.

## Run inference:
1. Set the `TRAIN_MODE` variable in `main.py` to `False`
2. Run main.py and the agent start training in the environment.
