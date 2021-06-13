import math

import os
import os.path as osp

from predators_and_preys_env.env import PredatorsAndPreysEnv
import numpy as np

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F

from torch_geometric.data import Batch

from tqdm.auto import tqdm

from examples.simple_chasing_agents.agents import ChasingPredatorAgent
from examples.simple_chasing_agents.agents import FleeingPreyAgent
import time

from agents.pyg import ActorConfig, PreyActor, PredatorActor
from utils.pyg_buffer import state2tensor

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-device', '--device', dest='device', default='cpu')
parser.add_argument('-batch_size', '--batch_size', dest='batch_size', type=int, default=128)
parser.add_argument('-n_steps', '--n_steps', dest='n_steps', type=int, default=300000)
parser.add_argument('-ckpt_save_path', '--ckpt_save_path', dest='ckpt_save_path', default='.', type=str)
parser.add_argument('-lr', '--lr', dest='lr', default=1e-3, type=float)
parser.add_argument('-wd', '--wd', dest='wd', default=1e-5, type=float)
args = parser.parse_args()


# def angdiff(a1, a2):
#     return (math.pi * ((a1 - a2 + 1) % 2 - 1)).abs().sum()

def angle_loss(x, y):
    return torch.atan2(torch.sin(math.pi * (x - y)), torch.cos(math.pi * (x - y))).mean()


def pretrain(prey_config: ActorConfig = ActorConfig(),
             predator_config: ActorConfig = ActorConfig(),
             prey_teacher=FleeingPreyAgent(),
             predator_teacher=ChasingPredatorAgent()):

    prey_actor = PreyActor(prey_config)
    print(prey_actor)
    predator_actor = PredatorActor(predator_config)
    print(predator_actor)
    device = torch.device(args.device)
    env = PredatorsAndPreysEnv(render=False)
    pred_optim = AdamW(predator_actor.parameters(), lr=args.lr, weight_decay=args.wd)
    prey_optim = AdamW(prey_actor.parameters(), lr=args.lr, weight_decay=args.wd)

    pred_scheduler = StepLR(pred_optim, step_size=100000)
    prey_scheduler = StepLR(pred_optim, step_size=100000)

    prey_actor.to(device)
    predator_actor.to(device)

    batch = []
    pred_actions = []
    prey_actions = []

    bar = tqdm(total=args.n_steps)

    done = True
    step_count = 0
    state_dict = None
    os.makedirs(args.ckpt_save_path, exist_ok=True)

    # evaluate_batch = []
    while True:
        if done:
            state_dict = env.reset()

        pred_action = predator_teacher.act(state_dict)
        prey_action = prey_teacher.act(state_dict)

        batch.append(state2tensor(state_dict))
        pred_actions.append(torch.tensor(pred_action, dtype=torch.float))
        prey_actions.append(torch.tensor(prey_action, dtype=torch.float))
        # evaluate_batch.append(list(state2tensor(state_dict)) +
        #                       [torch.tensor(pred_action, dtype=torch.float),
        #                        torch.tensor(prey_action, dtype=torch.float)])
        state_dict, _, done = env.step(pred_action, prey_action)

        # if len(evaluate_batch) == args.batch_size * 100:
        #     pred_error = prey_error = 0.
        #     for j in range(100):
        #         pred_state, prey_state, obst_state, prey_is_alive, pred_action, prey_action = \
        #             [torch.stack(x, dim=0).to(device) for x in zip(*evaluate_batch[args.batch_size * j:args.batch_size * (j + 1)])]
        #
        #         pred_angle = predator_actor(pred_state, prey_state, obst_state, prey_is_alive)
        #         pred_error += angdiff(pred_angle.squeeze(), pred_action)
        #         prey_angle = prey_actor(pred_state, prey_state, obst_state, prey_is_alive)
        #         prey_error += angdiff(prey_angle.squeeze(), prey_action)
        #     evaluate_batch = []
        #     bar.set_description({'pred_error': pred_error / 100, 'prey_error': prey_error / 100})

        if len(batch) == args.batch_size:
            state = Batch.from_data_list(batch).to(device)

            pred_action = torch.cat(pred_actions).view(-1, 1).to(device)
            prey_action = torch.cat(prey_actions).view(-1, 1).to(device)

            pred_comps = predator_actor(state)
            pred_optim.zero_grad()
            pred_loss = F.mse_loss(pred_comps, pred_action)
            pred_loss.backward()
            pred_optim.step()
            pred_scheduler.step()

            prey_comps = prey_actor(state)

            prey_optim.zero_grad()
            prey_loss = F.mse_loss(prey_comps, prey_action)
            prey_loss.backward()
            prey_optim.step()
            prey_scheduler.step()

            res = {'pred_mse': pred_loss.item(), 'prey_mse': prey_loss.item()}
            bar.set_postfix(res)

            batch = []
            pred_actions = []
            prey_actions = []

            step_count += 1
            bar.update(1)
            if step_count % 1000 == 0:
                torch.save({'config': predator_config,
                            'state_dict': predator_actor.state_dict(),
                            'optim': pred_optim.state_dict()},
                           osp.join(args.ckpt_save_path, f'predator_step_{step_count}.pt'))
                torch.save({'config': prey_config,
                            'state_dict': prey_actor.state_dict(),
                            'optim': prey_optim.state_dict()},
                           osp.join(args.ckpt_save_path, f'prey_step_{step_count}.pt'))
            if step_count == args.n_steps:
                break


if __name__ == '__main__':
    prey_config = ActorConfig(entity='prey')
    predator_config = ActorConfig(entity='predator')
    pretrain(prey_config, predator_config)
