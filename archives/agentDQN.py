#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:26:05 2020

@author: jupiter
"""
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.optimizers import SGD, Adam
import random

def QNetwork(obs_size, num_actions, nhidden, lr):
    model = Sequential()
    model.add(Dense(nhidden, input_dim=obs_size, activation=relu))
    model.add(Dense(nhidden, activation=relu))
    model.add(Dense(num_actions, activation=linear))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=lr))
    
    return model

class Agent(object):  
        
    def __init__(self, actions):
        self.actions = actions
        # self.num_actions = len(actions)
        self.num_actions = actions

    def act(self, state):
        raise NotImplementedError
        
        
class DQNAgent(Agent):
    """Deep Q-Learning agent"""

    def __init__(self, actions, obs_size, **kwargs):
        
        self.session=tf.Session()   
        super(DQNAgent, self).__init__(actions)
        
        # Taille de S
        self.obs_size = obs_size
        
        # Epsilon
        self.epsilon = kwargs.get('epsilon', .01)       
        # Si epsilon = 1, décroissance progressive
        if self.epsilon == 1:
            self.epsilon_decay = True
        else:
            self.epsilon_decay = False
        
        # Facteur de dévaluation
        self.gamma = kwargs.get('gamma', .99)
        
        # Hyperparamètres des réseaux de neurones (modèle et cible)
        self.batch_size = kwargs.get('batch_size', 64)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.lr = kwargs.get('learning_rate', .0001)
        self.tau = kwargs.get('tau', .05)
        self.env_range = kwargs.get('env_range', np.zeros(17))
        
        # Instanciation des réseaux de neurones (modèle et cible)
        self.model_network = QNetwork(self.obs_size, self.num_actions, kwargs.get('nhidden', 150), self.lr)
        self.target_network = QNetwork(self.obs_size, self.num_actions, kwargs.get('nhidden', 150), self.lr)
        self.target_network.set_weights(self.model_network.get_weights()) 

        # Mémoire pour replay
        self.memory  = deque(maxlen=kwargs.get('mem_size', 1000000))
    
        self.step_counter = 0
    
    def random_vect(self):
        # semble contenir les range de chaque articulation
        Size=self.env_range.shape[0]
        va=np.zeros(Size)
        for i in range(Size):
            d=self.env_range[i]
            # va[i]=np.random.random()*(d[1]-d[0]) + d[0]
            va[i]=np.random.random()*(0.4+0.4) -0.4
        print(va)
        return(va)
    
    def act(self, state):        
        # For test 

    
        if np.random.random() < self.epsilon:
            # i = np.random.randint(0,len(self.actions))
            va=self.random_vect()
        else: 
            i = np.argmax(self.model_network.predict(state.reshape(1, state.shape[0]))[0])
        return(va[1:18])
        self.step_counter += 1 
        #self.epsilon = max(.01, self.epsilon * .996)
        
        # decay epsilon after each epoch
        if self.epsilon_decay:
            if self.step_counter % self.epoch_length == 0:
                #print(self.epsilon)
                self.epsilon = max(.01, self.epsilon * .975)
        
        action = self.actions[i]        
        return action
    
    def learn(self, state1, action1, reward, state2, done):
        """
        Apprentissage: Mémorisation -> Replay -> Mise-à-jour Target
        """
        
        # Démarrer l'entraînement après 1 epoch
        if self.step_counter <= self.epoch_length:
            return
        
        # Sauvegarde la transition dans la mémoire de replay
        self.remember(state1, action1, reward, state2, done)
        
        # Experience replay
        self.replay()       
        
        # Mise-à-jour du modèle cible
        self.target_train() 

        
    def remember(self, state1, action1, reward, state2, done):
        """
        Sauvegarde la transition dans la mémoire de replay
        """
        self.memory.append([state1, action1, reward, state2, done])
    
    def replay(self):
        # Taille de la mémoire de replay insuffisante
        if len(self.memory) < self.batch_size: 
            return

        minibatch = random.sample(self.memory, self.batch_size)
        
        #for state1, action1, reward, state2, done in minibatch:
        #    target = self.target_network.predict(state1.reshape(1, state1.shape[0]))
        #    if done:
        #        target[0][action1] = reward
        #    else:
        #        Q_future = max(self.target_network.predict(state2.reshape(1, state2.shape[0]))[0])
        #        target[0][action1] = reward + self.gamma * Q_future
        #    
        #    self.model_network.fit(state1.reshape(1, state1.shape[0]), target, epochs=1, verbose=0)
            
        # Implémentation parallèle
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.target_network.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.target_network.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model_network.fit(states, targets_full, epochs=1, verbose=0)
        
        self.model_network.save_weights("weights.h5")
            
            
    def target_train(self):
        """
        Mise-à-jour des poids du réseau de neurones "cible"
        """
        model_weights = self.model_network.get_weights()
        target_weights = self.target_network.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_network.set_weights(target_weights)
        
    def clear_agent(self):
        self.session.close()