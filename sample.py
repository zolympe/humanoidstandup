#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:29:29 2020

@author: jupiter
"""
import gym
import os
#from lib.simulation import Experiment
#import numpy as np

# os.system("export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so")
env = gym.make('HumanoidStandup-v2')
#env = gym.make('KungFuMaster-ramDeterministic-v4')

from gym import envs
print(envs.registry.all())    # print the available environments
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample()    # take a random action
        print(f"Action {action}")
        observation, reward, done, info = env.step(action)
        print(f"Shape {observation.shape}")
        print(f"Observation {observation}")
        print(f"Reward {reward},info {info}")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()