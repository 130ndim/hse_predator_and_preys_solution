from dataclasses import dataclass

import math

from typing import Optional, Sequence, Tuple, Union
from typing_extensions import Literal

import torch
from torch import Tensor, nn
from torch.nn import init

from .net import FFN
from .net.layers import MHA, Embedding, PointConv

from utils.buffer import State


@dataclass
class CriticConfig:
    input_size: Union[int, Tuple[int, int, int]] = 3
    hidden_sizes: Sequence[int] = (64, 64, 64)

    max_grad_norm: float = 0.5

    lr: float = 1e-3

    entity: Optional[Literal['predator', 'prey']] = None


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = FFN(config.hidden_sizes + [1])

    def forward(self, state: Tensor, action: Tensor):
        out = self.net(torch.cat([state, action], dim=-1))
        return out


class GNNCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.embedding = Embedding(config.hidden_sizes[0])
        self.mha = MHA(hs=config.hidden_sizes[0])

        self.net = FFN(list(config.hidden_sizes) + [1])

        self.action_lin = nn.Linear(2, config.hidden_sizes[0])

        self._entity = config.entity
        self.reset_parameters()

    def reset_parameters(self):
        self.net.seq[-1].weight.data.uniform_(1e-2, 1e-2)

    def forward(self, state: State, action: Tensor):
        pred_state, prey_state, obst_state = state.pred_state, state.prey_state, state.obst_state

        h_pred, h_prey, h_obst = self.embedding(pred_state, prey_state, state.obst_state)
        action = action * math.pi
        action_emb = self.action_lin(torch.cat([action.sin(), action.cos()], dim=-1))

        pos = torch.cat([pred_state, prey_state, obst_state[..., :-1]], dim=1)

        if self._entity == 'prey':
            h_prey = h_prey + action_emb
        else:
            h_pred = h_pred + action_emb

        out = self.mha(h_pred, h_prey, h_obst, pos)

        out = out[self._entity == 'prey']
        out = self.net(out)

        return out


class PCCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.embedding = nn.Embedding(5, config.hidden_sizes[0])
        self.obst_embedding = nn.Linear(1, config.hidden_sizes[0])

        self.action_embedding = nn.Linear(2, config.hidden_sizes[0])
        self.conv1 = PointConv(config.hidden_sizes[0])
        # self.conv2 = PointConv(config.hidden_sizes[0], config.hidden_sizes[0])

        self._entity = config.entity

        self.net = FFN(list(config.hidden_sizes) + [1])
        # self.net.seq.add_module(str(len(self.net.seq)), nn.Tanh())
        self.entity_idx = int(config.entity == 'prey')

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_normal_(self.obst_embedding.weight)
        nn.init.zeros_(self.obst_embedding.bias)
        nn.init.xavier_normal_(self.action_embedding.weight)
        nn.init.zeros_(self.action_embedding.bias)
        self.conv1.reset_parameters()
        self.net.reset_parameters()
        self.net.seq[-1].weight.data.uniform_(-1e-2, 1e-2)

    def forward(self, state: State, action: Tensor):
        B = state.pred_state.size(0)

        pos_pred = state.pred_state / 9.

        pos_prey, is_alive_prey = state.prey_state.split((2, 1), dim=-1)
        pos_prey = pos_prey / 9.
        is_alive_prey = is_alive_prey.squeeze(-1).long()

        pos_obst, r_obst = state.obst_state.split((2, 1), dim=-1)
        r_obst = (r_obst - 0.8) / 0.7
        pos_obst /= 9.

        x_pred = self.embedding.weight[2].repeat((B, pos_pred.size(1), 1))
        x_prey = self.embedding.weight[3].repeat((B, pos_prey.size(1), 1)) + self.embedding(is_alive_prey)
        x_obst = self.embedding.weight[4].repeat((B, pos_obst.size(1), 1)) + self.obst_embedding(r_obst)

        action *= math.pi
        x_action = self.action_embedding(torch.cat([action.sin(), action.cos()], dim=-1))

        if self._entity == 'prey':
            x_prey += x_action
        else:
            x_pred += x_action

        out = self.conv1(x_pred, x_prey, x_obst, pos_pred, pos_prey, pos_obst)[self.entity_idx]
        # out = self.conv2(x_pred, x_prey, x_obst, x_pred, x_prey, x_obst)[self.entity_idx]

        out = self.net(out + x_action)
        return out
