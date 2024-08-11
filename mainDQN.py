import gymnasium as gym
from DQN_acrobot import Agent
from utils import plotLearning
import numpy as np
import torch





if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
    agent = Agent(gamma=0.99, epsilon=0.2, batch_size=64, n_actions=3,n_obs=6,eps_end=0.01, lr=0.0001)
    scores, eps_history = [], []
    n_games =1000
    
    for i in range(n_games):
        score = 0
        done = False
        observation,_ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward,terminated,truncated,_= env.step(action)
            score += reward
          

            agent.store_experience(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
            done = terminated or truncated
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    torch.save(agent.Q_target.state_dict(), 'acrobot_model1.pth')
    x = [i+1 for i in range(n_games)]
    filename = 'acrobot.png'
    plotLearning(x, scores, eps_history, filename)