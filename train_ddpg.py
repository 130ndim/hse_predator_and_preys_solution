from collections import defaultdict

import os
import os.path as osp

import numpy as np

import torch

from tqdm.auto import tqdm

from agents.ddpg import DDPGAgent, DDPGConfig
from agents import CriticConfig, ActorConfig

from utils.pyg_buffer import FasterBuffer
from utils import Penalty
from predators_and_preys_env.env import PredatorsAndPreysEnv

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-device', '--device', dest='device', default=None)
parser.add_argument('-batch_size', '--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('-ckpt_save_path', '--ckpt_save_path', dest='ckpt_save_path', default='.', type=str)
parser.add_argument('-penalize_deadlocks', '--penalize_deadlocks', dest='penalize_deadlocks',
                    action='store_true', default=False)
parser.add_argument('-buffer_steps', '--buffer_steps', dest='buffer_steps', type=int, default=50000)
parser.add_argument('-n_preds', '--n_preds', dest='n_preds', type=int, default=2)
parser.add_argument('-n_preys', '--n_preys', dest='n_preys', type=int, default=5)
parser.add_argument('-n_obsts', '--n_obsts', dest='n_obsts', type=int, default=10)
parser.add_argument('-prey_ckpt', '--prey_ckpt', dest='prey_ckpt', type=str, default=None)
parser.add_argument('-pred_ckpt', '--pred_ckpt', dest='pred_ckpt', type=str, default=None)
args = parser.parse_args()


CONFIG = {
    "game": {
        "num_obsts": args.n_obsts,
        "num_preds": args.n_preds,
        "num_preys": args.n_preys,
        "x_limit": 9,
        "y_limit": 9,
        "obstacle_radius_bounds": [0.8, 1.5],
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/40,
        "frameskip": 2
    },
    "environment": {
        "frameskip": 2,
        "time_limit": 1000
    }
}


def evaluate_policy(env, predator_agent, prey_agent, episodes=5):
    predator_agent.eval()
    prey_agent.eval()
    returns = []
    for i in range(episodes):
        env.seed(777 + i)
        done = False
        state = env.reset()
        pred_reward = prey_reward = 0.

        while not done:
            state, reward, done = env.step(predator_agent.act(state), prey_agent.act(state))
            pred_reward += np.sum(reward['predators'])
            prey_reward += np.sum(reward['preys'])
        returns.append((pred_reward, prey_reward))

    returns = np.array(returns)
    return returns.mean(0), returns.std(0)


if __name__ == '__main__':
    steps = 2000000 + args.buffer_steps
    batch_size = 64

    loss_aggr_steps = 100

    if args.penalize_deadlocks:
        penalty = Penalty()
    else:
        penalty = lambda x, y: y

    predator_config = DDPGConfig(critic=CriticConfig(),
                                 actor=ActorConfig(),
                                 entity='predator',
                                 # actor_update_freq=1,
                                 # soft_update_freq=1,
                                 # tau=0.999,
                                 )
    prey_config = DDPGConfig(critic=CriticConfig(),
                             actor=ActorConfig(),
                             entity='prey',
                             # actor_update_freq=1,
                             # soft_update_freq=1,
                             # tau=0.999,
                             )

    env = PredatorsAndPreysEnv(CONFIG, render=False)
    predator_agent = DDPGAgent(predator_config).to(args.device)
    prey_agent = DDPGAgent(prey_config).to(args.device)

    if args.pred_ckpt is not None:
        predator_agent.actor.load_state_dict(args.pred_ckpt, map_location=args.device)
    if args.prey_ckpt is not None:
        prey_agent.actor.load_state_dict(args.prey_ckpt, map_location=args.device)
    print(next(predator_agent.actor.parameters()).device)

    buffer = FasterBuffer(buffer_size=400000)

    global_step_count = 0
    done = True
    step_count = 0
    state = env.reset()
    # buffer_bar = tqdm(total=buffer_steps)
    global_bar = tqdm(total=steps)

    pred_loss = []
    prey_loss = []
    rewards = []
    aggr_loss = defaultdict(list)
    while True:
        predator_agent.train()
        prey_agent.train()
        if global_step_count < args.buffer_steps:
            predator_actions = np.random.rand(env.predator_action_size) * 2 - 1
            prey_actions = np.random.rand(env.prey_action_size) * 2 - 1
            # buffer_bar.update(1)
        else:
            # buffer_bar.disable = True
            pred_loss_ = predator_agent.update(buffer.get_batch(batch_size, 'predator'))
            pred_loss.append(pred_loss_)
            for k, v in pred_loss_.items():
                aggr_loss['pred_'+k].append(v)

            prey_loss_ = prey_agent.update(buffer.get_batch(batch_size, 'prey'))
            prey_loss.append(prey_loss_)
            for k, v in prey_loss_.items():
                aggr_loss['prey_'+k].append(v)

            predator_actions = predator_agent.act(state)
            prey_actions = prey_agent.act(state)

        next_state, reward, done = env.step(predator_actions, prey_actions)
        reward = penalty(next_state, reward)

        # print(next_state)
        buffer.append(
            state,
            {'predators': predator_actions.tolist(), 'preys': prey_actions.tolist()},
            next_state,
            reward,
            done
        )
        step_count += 1

        if done:
            state = env.reset()
            step_count = 0
        else:
            state = next_state

        global_step_count += 1

        global_bar.update(1)
        os.makedirs(args.ckpt_save_path, exist_ok=True)
        if global_step_count > args.buffer_steps:

            if (global_step_count) % loss_aggr_steps == 0:
                losses = {k: np.nanmean(v) for k, v in aggr_loss.items()}
                global_bar.set_postfix(losses)
                aggr_loss = defaultdict(list)

            if (global_step_count) % 20000 == 0:
                mean, std = evaluate_policy(env, predator_agent, prey_agent)
                rewards.append((mean, std))
                global_bar.set_description(f'pred_r = {int(mean[0])}({int(std[0])}) | '
                                           f'prey_r = {int(mean[1])}({int(std[1])})')

            if (global_step_count) % 5000 == 0:
                predator_agent.save(osp.join(args.ckpt_save_path, f'ddpg_pred_{global_step_count}.pt'))
                prey_agent.save(osp.join(args.ckpt_save_path, f'ddpg_prey_{global_step_count}.pt'))

            if (global_step_count) % 200000 == 0:
                torch.save(pred_loss, osp.join(args.ckpt_save_path, 'pred_loss.pt'))
                torch.save(prey_loss, osp.join(args.ckpt_save_path, 'prey_loss.pt'))
                torch.save(rewards, osp.join(args.ckpt_save_path, 'rewards.pt'))

