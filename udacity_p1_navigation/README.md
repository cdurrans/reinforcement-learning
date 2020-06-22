[//]: # (Image References)

# Project 1: Navigation

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

![Trained Agent][image1]

### Introduction

For this project, I trained an agent to navigate (and collect bananas!) in a large, square world.  

My agent was rewarded +1 for collecting a yellow banana, and punished -1 for collecting a blue banana. The environment is considered solved when the agent has a average reward of +13 over 100 consecutive episodes. I successfully trained mine to have an average reward of +15 in 700 episodes.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

For futher information about the environment and for instructions to download the simulator go here: https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md

The task is episodic and I utilized an implementation of OpenAI's Deep Q-Network from the famous paper: <em>Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013. Algorithm: DQN.</em>

For the full code implementation check out these files in<br>https://github.com/cdurrans/reinforcement-learning/udacity_p1_navigation/:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Navigation.ipynb is the main file to run and watch training.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* model.py contains the pytorch model.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* dqn_agent.py contains the DQN agent.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* model.pth is the saved pytorch model.<br>


In order to train the agent make sure you use the library versions specified in the requirements.txt or ones compatible to them and the code. Use the Navigation.ipynb for training.




