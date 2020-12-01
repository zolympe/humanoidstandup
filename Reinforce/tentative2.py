#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:14:25 2020

@author: jupiter
"""
#!pip install --upgrade pip
#!pip install pandas
#!pip install tqdm
#!pip install gym
#!pip install Box2D

import sys
#from collections import deque
#import random
#import numpy as np

import gym

if "." not in sys.path:
    sys.path.append("../../") 

from mySimulation import ExperimentR
#from agentDQN import DQNAgent
from agentREINFORCEMENT import REINFORCEAgent

interactiveGraph = True
interactiveBot = True

# %matplotlib nbagg
#env = gym.make("LunarLander-v2")
env = gym.make('HumanoidStandup-v2')
print(f"num action {env.action_space.shape[0]}")
print(f"Num obs {env.observation_space.shape[0]}")

# Creation de l agent
agent = REINFORCEAgent(actions_space=env.action_space.shape[0], observation_space=env.observation_space.shape[0],
                       learning_rate=0.001,gamma=0.99,batch_size=1)

# Creation de l experiement
experiment = ExperimentR(env, agent,interactiveBot=True)

# Lancement de la simulation
experiment.run_reinforce() # 1, interactiveGraph,interactiveBot)

# agent.clear_agent()