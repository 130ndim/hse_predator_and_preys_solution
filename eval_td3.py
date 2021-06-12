import numpy as np

import torch

from tqdm.auto import tqdm

from agents.td3 import TD3Agent, TD3Config
from agents import CriticConfig, ActorConfig

import os
import os.path as osp

from utils import Buffer, Penalty
import pathlib

os.environ['PYTHONPATH'] = f'{osp.join(pathlib.Path(__file__).parent, "Predators-and-Preys")}'
from predators_and_preys_env.env import PredatorsAndPreysEnv


def evaluate_policy(env, predator_agent, prey_agent, episodes=5):
    predator_agent.eval()
    prey_agent.eval()
    returns = []
    env.reset()
    for i in range(episodes):
        env.seed(1000)
        done = False
        state = env.reset()
        # print(state)
        pred_reward = prey_reward = 0.

        while not done:
            pda, pya = predator_agent.act(state), prey_agent.act(state)
            print(pda, pya)
            state, reward, done = env.step(pda, pya)
            # print(reward)
            Penalty(5)(state, reward)
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
    config = {
        "game": {
            "num_obsts": 10,
            "num_preds": 2,
            "num_preys": 5,
            "x_limit": 9,
            "y_limit": 9,
            "obstacle_radius_bounds": [0.8, 1.5],
            "prey_radius": 0.8,
            "predator_radius": 1.0,
            "predator_speed": 6.0,
            "prey_speed": 9.0,
            "world_timestep": 1 / 40,
            "frameskip": 2
        },
        "environment": {
            "frameskip": 2,
            "time_limit": 1000
        }
    }
    env = PredatorsAndPreysEnv(config, render=True)
    predator_agent = TD3Agent().from_ckpt('./td3_pred_70000.pt', map_location='cpu')
    # predator_agent = TD3Agent(TD3Config(actor=ActorConfig(hidden_sizes=(256, 256, 256))))

    predator_agent.eval()
    # predator_agent.actor.load_state_dict(torch.load('pred_pretr.pt', map_location='cpu')['state_dict'])
    predator_agent.add_noise_on_inference = True
    # predator_agent.noise.scale = 0.1
    prey_agent = TD3Agent().from_ckpt('./td3_prey_70000.pt', map_location='cpu')
    # prey_agent = TD3Agent(TD3Config(actor=ActorConfig(hidden_sizes=(256, 256, 256))))
    prey_agent.eval()
    # prey_agent.actor.load_state_dict(torch.load('prey_pretr.pt', map_location='cpu')['state_dict'])
    prey_agent.add_noise_on_inference = True
    # prey_agent.noise.scale = 0.1
    print(prey_agent.noise)
    for n, p in predator_agent.actor.net.seq.named_parameters():
        print(n, p)

    # for n, p in prey_agent.actor.named_parameters():
    #     print(n, p.mean(), p.std())


    print(evaluate_policy(env, predator_agent, prey_agent))


