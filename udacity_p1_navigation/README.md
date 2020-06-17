[//]: # (Image References)

# Project 1: Navigation

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

![Trained Agent][image1]

### Introduction

For this project, I trained an agent to navigate (and collect bananas!) in a large, square world.  

For futher information about the environment and for instructions to download the simulator go here: https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md

A quick re-cap is that my agent was rewarded +1 for collecting a yellow banana, and punished -1 for collecting a blue banana. My goal was to have a score of +13 over a 100 consecutive span of episodes. I successfully trained mine to have a reward score on average of +15.

The task is episodic and I utilized an implementation of OpenAI's Deep Q-Network from the famous paper: <em>Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013. Algorithm: DQN.</em>

For the full code implementation check out these files in<br>https://github.com/cdurrans/reinforcement-learning/udacity_p1_navigation/:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Navigation.ipynb is the main file to run and watch training.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* model.py contains the pytorch model.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* dqn_agent.py contains the DQN agent.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* ddqn_agent.py contains the Double Q-Network agent.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* model.pth is the saved pytorch model.<br>









