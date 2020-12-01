#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 08:21:10 2020

@author: jupiter
"""
import numpy as np
import sys
import lib.plotting as plotting
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

class Experiment(object):
    def __init__(self, env, agent):
        
        self.env = env
        self.agent = agent
        
        self.episode_length = np.array([0])
        self.episode_reward = np.array([0])
        
        self.fig = plt.figure(figsize=(10, 5))
        plt.ion()
        # Grille de présentation
        gs = gridspec.GridSpec(2, 2)

        self.ax = plt.subplot(gs[:, 0])

        # Graph length        
        self.ax.xaxis.set_visible(True)
        self.ax.yaxis.set_visible(True)
        #self.ax.set_xticks(np.arange(-.5, 10, 1), minor=True);
        #self.ax.set_yticks(np.arange(-.5, 7, 1), minor=True);
        self.ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        
        self.ax1 = plt.subplot(gs[0, 1])
        self.ax1.yaxis.set_label_position("right")
        self.ax1.set_ylabel('Length')
        
        self.ax1.set_xlim(0, max(10, len(self.episode_length)+1))
        self.ax1.set_ylim(0, 51)
        
        self.ax2 = plt.subplot(gs[1, 1])
        self.ax2.set_xlabel('Episode')
        self.ax2.yaxis.set_label_position("right")
        self.ax2.set_ylabel('Reward')
        self.ax2.set_xlim(0, max(10, len(self.episode_reward)+1))
        self.ax2.set_ylim(0, 2)
        
        self.line, = self.ax1.plot(range(len(self.episode_length)),self.episode_length)
        self.line2, = self.ax2.plot(range(len(self.episode_reward)),self.episode_reward)
        
    def update_display_step(self):
        if not hasattr(self, 'imgplot'):
            self.imgplot = self.ax.imshow(self.env.render(mode='rgb_array'), interpolation='none', cmap='viridis')
        else:
            self.imgplot.set_data(self.env.render(mode='rgb_array'))
    
        self.fig.canvas.draw()
        
    def update_display_episode(self):  
        self.line.set_data(range(len(self.episode_length)),self.episode_length)
        self.ax1.set_xlim(0, max(10, len(self.episode_length)+1))
        self.ax1.set_ylim(0, max(self.episode_length)+1)
        
        self.line2.set_data(range(len(self.episode_reward)),self.episode_reward)
        self.ax2.set_xlim(0, max(10, len(self.episode_reward)+1))
        self.ax2.set_ylim(min(self.episode_reward)-1, max(self.episode_reward)+1)
        
        self.fig.canvas.draw()  
        self.fig.canvas.flush_events()

    def run_qlearning(self, max_number_of_episodes=100, interactiveGraph = False,interactiveBot=False, display_frequency=1):

        # repeat for each episode
        plt.ion()
        for episode_number in range(max_number_of_episodes):
            
            # initialize state
            state = self.env.reset()
            
            done = False # used to indicate terminal state
            R = 0 # used to display accumulated rewards for an episode
            t = 0 # used to display accumulated steps for an episode i.e episode length
            
            # repeat for each step of episode, until state is terminal
            while not done:
                if interactiveBot:
                    self.env.render()
                t += 1 # increase step counter - for display
                
                # choose action from state using policy derived from Q
                # 
                action = self.agent.act(state)
                print("action ",action)
                # For test Purpose
                #action = self.env.action_space.sample()

                # take action, observe reward and next state
                next_state, reward, done, _ = self.env.step(action)

                # agent learn (Q-Learning update)
                self.agent.learn(state, action, reward, next_state, done)
                
                # state <- next state
                state = next_state
                
                R += reward # accumulate reward - for display
                
                # if interactive display, show update for each step
                # if interactiveBot:
                    # self.update_display_step()
            
            self.episode_length = np.append(self.episode_length,t) # keep episode length - for display
            self.episode_reward = np.append(self.episode_reward,R) # keep episode reward - for display 
            
            # if interactive display, show update for the episode
            if interactiveGraph:
              self.update_display_episode()
        
        # if not interactive display, show graph at the end
        if not interactiveGraph:
            self.fig.clf()
            stats = plotting.EpisodeStats(
                episode_lengths=self.episode_length,
                episode_rewards=self.episode_reward,
                episode_running_variance=np.zeros(max_number_of_episodes))
            plotting.plot_episode_stats(stats, display_frequency)
            
class ExperimentR(object):
    def __init__(self, env, agent, EPISODES=100000, training=True, episode_max_length=None, 
                 interactiveBot=False,mean_episodes=50):
        self.env = env
        self.agent = agent
        self.EPISODES = EPISODES
        self.training = training
        self.episode_max_length = episode_max_length
        self.interactiveBot=interactiveBot
        self.mean_episodes = mean_episodes
        # self.stop_criterion = stop_criterion

    def run_reinforce(self):
        
        # Tableaux utiles pour l'affichage
        scores, mean, episodes,myCost = [], [], [],[]
        
        plt.ion()
        # fig = plt.figure()
        fig,(ax1,ax2)=plt.subplots(2,1,sharex=True )
        
        for i in range(self.EPISODES):
            done = False
            score = 0
            state = self.env.reset()

            counter = 0
            while not done:
                counter +=1

                # Afficher l'environnement
                if self.interactiveBot:
                    self.env.render()

                # Obtient l'action pour l'état courant
                action = self.agent.act(state)

                # Effectue l'action
                next_state, reward, done, _ = self.env.step(action)
                # print(f"Counter = {counter}, Reward = {round(reward,0)}")
                # Sauvegarde la transition <s, a, r> dans la mémoire
                self.agent.remember(state, action, reward)

                # Mise à jour de l'état
                state = next_state

                # Accumulation des récompenses
                score += reward

                # Arrête l'épisode après 'episode_max_length' instants
                #if self.episode_max_length != None and counter > self.episode_max_length:
                    # done = True

            # Lance l'apprentissage de la politique
            if self.training == True:
                myCost.append(self.agent.learn())

            # Arrête l'entraînement lorsque la moyenne des récompense sur 'mean_episodes' épisodes est supérieure à 
            #if np.mean(scores[-self.mean_episodes:]) > self.stop_criterion:
            #    break

            # Sauvegarde du modèle (poids) tous les 50 épisodes
            if self.training and i % 50 == 0:
                self.agent.predict.save(f"../{self.env.spec.id}_reinforce.h5")    
            
            # Affichage des récompenses obtenues
            if self.training == True:
                scores.append(score)
                mean.append(np.mean(scores[-self.mean_episodes:]))
                episodes.append(i)
                # fig.clf()
                ax1.plot(episodes, scores, 'b', label='gains')
                ax1.plot(episodes, mean, 'r', label='Moyenne des gains')
                ax1.set_xlabel("Épisodes")
                ax1.set_ylabel("Gains")
                ax2.plot(episodes,myCost)
                ax2.set_ylabel("Cost")
                fig.canvas.draw()
                fig.canvas.flush_events()