import gym
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    env = gym.make('HumanoidStandup-v2')
    agent = Agent(alpha=0.0005, beta=0.005,
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=1000, fc2_dims=300,  # b was 64 fc1 was 400 fc 2 was 300
                    n_actions=env.action_space.shape[0],gamma=0.99)
    showBot_episode=True
    showBot_turn=False
    freshStart=True
    n_games = 5000

    filename = 'humanoid_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    print("Important Values")
    print(f"best Score {best_score}")
    print(f"Obs Space {env.observation_space.shape}")
    print(f"Action Space {env.action_space.shape[0]}")
    
    score_history = []
    mean = []
    plt.ion()
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)

    if freshStart==False:
        print("Load checkpoint..")
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            # action=action/np.abs(action).max()
            # action=action*0.4
            observation_, reward, done, info = env.step(action)
            #print(f"{i} - > reward={reward} done={done}")
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
            if showBot_episode:
                env.render()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        mean.append(avg_score)

        if showBot_turn:
            env.render()
        if avg_score > best_score and i>0:
            best_score = avg_score
            agent.save_models()
            x = [z+1 for z in range(i+1)]
            sub_figure_file = 'plots/' + str(i) + filename + '.png'
            plot_learning_curve(x, score_history, sub_figure_file)

        # show Status
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
        ax1.plot(score_history)
        ax2.plot(mean)
        fig.canvas.draw()
        fig.canvas.flush_events()

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)




