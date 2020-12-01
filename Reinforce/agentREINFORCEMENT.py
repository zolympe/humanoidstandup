#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 11:13:40 2020

@author: jupiter
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as BK

import matplotlib.pyplot as plt


class Agent(object):  
        
    def __init__(self, observation_space, action_space):
        #self.observation_space = observation_space
        self.state_size = observation_space # .shape[0]
        self.action_space = action_space
        self.num_actions = action_space #.n

    def act(self, state):
        raise NotImplementedError
        
class REINFORCEAgent(Agent):
    def __init__(self, observation_space, actions_space, 
                 learning_rate = 0.001, gamma = 0.99, hidden1=64, hidden2=64,
                 batch_size=10):
        super(REINFORCEAgent, self).__init__(observation_space, actions_space)

        # Hyperparamètres du policy gradient
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.hidden1, self.hidden2 = hidden1, hidden2
        self.batch_size=batch_size
        self.rangeMin=-0.4
        self.rangeMax=0.4
        self.motorRange=self.rangeMax-self.rangeMin
        self.et=self.motorRange/2
        # Création du modèle de la politique
        self.policy, self.predict = self.policy_network()

        # Mémoire de la trajectoire
        self.states_memory, self.actions_memory, self.rewards_memory = [], [], []
        
        self.render = False


            
    def policy_network(self):
        """
        La politique est modélisée par une réseau de neurones
        Entrée: état
        Sortie: probabilité de chaque action
        """
        def mapping_to_target_range( x, target_min=-0.4, target_max=0.4 ) :
            x02 = BK.tanh(x) + 1 # x in range(0,2)
            # x02 = BK.linear(x) + 1 # x in range(0,2)
            scale = ( target_max-target_min )/2.0
            return  x02 * scale + target_min
        # last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        inpt = Input(shape=(self.state_size,))
        # inpt = Input(shape=(self.state_size+self.action_space,))
        
        advantages = Input(shape=[1])
        dense1 = Dense(self.state_size*10, activation='relu')(inpt)
        # dense2 = Dense(np.sqrt(self.state_size*10*self.action_space), activation='relu')(dense1)
        dense3 = Dense(self.action_space*10, activation='relu')(dense1)
        outputs = Dense(self.num_actions, activation=mapping_to_target_range)(dense3) #tanh
        
        def custom_loss(y_true, y_pred):
            
            out = tf.keras.backend.clip(y_pred, 1e-8, 1-1e-8)
            log_likelihood = y_true*tf.keras.backend.log(out)

            return tf.keras.backend.sum(-log_likelihood * advantages)
        
        policy = Model(inputs=[inpt, advantages], outputs=[outputs])
        policy.compile(optimizer=Adam(lr=self.learning_rate), loss=custom_loss)

        predict = Model(inputs=[inpt], outputs=[outputs])

        return policy, predict
    

    def act(self, state):
        """
        Sélection d'une action suivant la sortie du réseau de neurones
        """
        state=state[np.newaxis, :] # .reshape(1, state.shape[0])
        
        prob=self.predict.predict(state,batch_size=self.batch_size)
        # prob=prob/np.max(prob)*self.motorRange
        # print(prob)
        #print(f"Prob {np.round(prob,2)}")
        action=np.zeros(self.num_actions)
        for i in range(self.num_actions):
           action[i]=np.random.normal(prob[0][i],scale=(self.et))
           if action[i]>self.rangeMax:
               action[i]=self.rangeMax                
           if action[i]<self.rangeMin:
                action[i]=self.rangeMin
        #print(f"Action {np.round(action,2)}")
        # action=np.random.choice(self.num_actions, 1, p=prob)[0]
        # action=action.reshape(-1,1)
        return(action)
                
          
        # i = np.argmax(self.predict.predict(state.reshape(1, state.shape[0]))[0])
        
    
    def discount_rewards(self, rewards):
        """
        La politique est évaluée à partir des gains dévalués
        """
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    
    def remember(self, state, action, reward):
        """
        Sauvegarde <s, a ,r> pour chaque instant
        """
        # Compléter le code ci-dessous ~ 3 lignes
        # self.memory.append([state1, action1, reward, state2, done])
        self.states_memory.append(state)
        self.actions_memory.append(action)
        self.rewards_memory.append(reward)

              
    def learn(self):
        """
        Mise à jour du "policy network" à chaque épisode
        """
        states_memory = np.array(self.states_memory)
        actions_memory = np.array(self.actions_memory)
        rewards_memory = np.array(self.rewards_memory)

        print(f"action memory{actions_memory.shape}")
        print(f"states shape {states_memory.shape}")
        print(f"rewards_memory {rewards_memory.shape}")
        #actions = np.zeros([len(actions_memory), self.num_actions])
        #actions[np.arange(len(actions_memory)), actions_memory] = 1

        discounted_rewards = self.discount_rewards(self.rewards_memory)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        # myInput=np.concatenate((states_memory,actions_memory))
        myCost=(self.policy.train_on_batch([states_memory, discounted_rewards], actions_memory))
        # self.policy.train_on_batch([myInput, discounted_rewards], actions_memory)
        self.states_memory, self.actions_memory, self.rewards_memory = [], [], []
        return(myCost)