import numpy as np

import torch

from tqdm.auto import tqdm

from agents.ddpg import DDPGAgent, DDPGConfig
from agents import CriticConfig, ActorConfig

import os
import os.path as osp

from utils import Buffer
import pathlib

os.environ['PYTHONPATH'] = f'{osp.join(pathlib.Path(__file__).parent, "Predators-and-Preys")}'
from predators_and_preys_env.env import PredatorsAndPreysEnv


def evaluate_policy(env, predator_agent, prey_agent, episodes=5):
    predator_agent.eval()
    prey_agent.eval()
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        pred_reward = prey_reward = 0.

        while not done:
            pda, pya = predator_agent.act(state), prey_agent.act(state)
            print(pda, pya)
            state, reward, done = env.step(pda, pya)
            pred_reward += np.sum(reward['predators'])
            prey_reward += np.sum(reward['preys'])
        returns.append((pred_reward, prey_reward))

    returns = np.array(returns)
    return returns.mean(0), returns.std(0)


if __name__ == '__main__':
    # predator_config = DDPGConfig(critic=CriticConfig(input_size=(3, 2, 3), entity='predator'),
    #                              actor=ActorConfig(entity='predator'),
    #                              entity='predator')
    # prey_config = DDPGConfig(critic=CriticConfig(input_size=(2, 3, 3), entity='prey'),
    #                          actor=ActorConfig(entity='prey'),
    #                          entity='prey')

    env = PredatorsAndPreysEnv(render=True)
    predator_agent = DDPGAgent().from_ckpt('./ddpg_pred_100000.pt', map_location='cpu')
    predator_agent.eval()
    prey_agent = DDPGAgent().from_ckpt('./ddpg_prey_100000.pt', map_location='cpu')
    prey_agent.eval()

    print(evaluate_policy(env, predator_agent, prey_agent))


