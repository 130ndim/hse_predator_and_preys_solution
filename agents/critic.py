from dataclasses import dataclass

import math

from typing import Optional, Sequence, Tuple, Union
from typing_extensions import Literal

import torch
from torch import Tensor, nn
from torch.nn import init

from .net import FFN
from .net.layers import PointConv, MHA
from .utils import reset


@dataclass
class CriticConfig:
    input_size: Union[int, Tuple[int, int, int]] = 3
    hidden_sizes: Sequence[int] = (64, 64, 64)

    max_grad_norm: float = 0.5

    lr: float = 3e-4

    entity: Optional[Literal['predator', 'prey']] = None


class PredatorCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_sizes[0])
        self.obst_embedding = nn.Linear(1, config.hidden_sizes[0])
        self.action_embedding = nn.Linear(2, config.hidden_sizes[0])
        self.pos_embedding = nn.Linear(2, config.hidden_sizes[0])

        self.conv1 = PointConv(config.hidden_sizes[0])
        self.conv2 = PointConv(config.hidden_sizes[0])

        self.net = FFN(list(config.hidden_sizes) + [1])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_normal_(self.obst_embedding.weight)
        nn.init.xavier_normal_(self.pos_embedding.weight)
        nn.init.xavier_normal_(self.action_embedding.weight)
        nn.init.zeros_(self.obst_embedding.bias)
        nn.init.zeros_(self.pos_embedding.bias)
        nn.init.zeros_(self.action_embedding.bias)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.net.reset_parameters()
        self.net.seq[-1].weight.data.uniform_(-1e-2, 1e-2)

    def forward(self, pred_state, prey_state, obst_state, prey_is_alive, action):
        B = pred_state.size(0)

        pos_pred = pred_state
        pos_prey = prey_state

        pos_obst, r_obst = obst_state.split((2, 1), dim=-1)

        action *= math.pi
        x_action = self.action_embedding(torch.cat([action.sin(), action.cos()], dim=-1))

        x_pred = self.embedding.weight[0].repeat((B, pos_pred.size(1), 1)) + x_action
        x_prey = self.embedding.weight[1].repeat((B, pos_prey.size(1), 1))
        x_obst = self.embedding.weight[2].repeat((B, pos_obst.size(1), 1)) + self.obst_embedding(r_obst)

        x_prey[~prey_is_alive].fill_(0.)
        out_pred, out_prey, out_obst = self.conv1(x_pred, x_prey, x_obst, pos_pred, pos_prey, pos_obst, prey_is_alive)
        out_prey[~prey_is_alive].fill_(0.)
        out = self.conv2(out_pred + x_action, out_prey, out_obst, pos_pred, pos_prey, pos_obst, prey_is_alive)[0]
        out = out + x_action + self.pos_embedding(pos_pred)

        out = self.net(out)
        return out


class PreyCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_sizes[0])
        self.obst_embedding = nn.Linear(1, config.hidden_sizes[0])
        self.action_embedding = nn.Linear(2, config.hidden_sizes[0])
        self.pos_embedding = nn.Linear(2, config.hidden_sizes[0])

        self.conv1 = PointConv(config.hidden_sizes[0])
        self.conv2 = PointConv(config.hidden_sizes[0])

        self.net = FFN(list(config.hidden_sizes) + [1])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_normal_(self.obst_embedding.weight)
        nn.init.xavier_normal_(self.pos_embedding.weight)
        nn.init.xavier_normal_(self.action_embedding.weight)
        nn.init.zeros_(self.obst_embedding.bias)
        nn.init.zeros_(self.pos_embedding.bias)
        nn.init.zeros_(self.action_embedding.bias)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.net.reset_parameters()
        self.net.seq[-1].weight.data.uniform_(-1e-2, 1e-2)

    def forward(self, pred_state, prey_state, obst_state, prey_is_alive, action):
        B = pred_state.size(0)

        pos_pred = pred_state
        pos_prey = prey_state

        pos_obst, r_obst = obst_state.split((2, 1), dim=-1)

        action *= math.pi
        x_action = self.action_embedding(torch.cat([action.sin(), action.cos()], dim=-1))

        x_pred = self.embedding.weight[0].repeat((B, pos_pred.size(1), 1))
        x_prey = self.embedding.weight[1].repeat((B, pos_prey.size(1), 1)) + x_action
        x_obst = self.embedding.weight[2].repeat((B, pos_obst.size(1), 1)) + self.obst_embedding(r_obst)

        x_prey[~prey_is_alive].fill_(0.)
        out_pred, out_prey, out_obst = self.conv1(x_pred, x_prey, x_obst, pos_pred, pos_prey, pos_obst, prey_is_alive)
        out_prey[~prey_is_alive].fill_(0.)
        out = self.conv2(out_pred, out_prey + x_action, out_obst, pos_pred, pos_prey, pos_obst, prey_is_alive)[1]
        out = out + x_action
        out[~prey_is_alive].fill_(0.)

        out = self.net(out + self.pos_embedding(pos_prey))
        return out


class PredatorAttnCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_sizes[0])

        self.seqs = nn.ModuleList([
            FFN([3] + [config.hidden_sizes[0]] * 2),
            FFN([2] + [config.hidden_sizes[0]] * 2),
            FFN([3] + [config.hidden_sizes[0]] * 2)
        ])

        self.attn = MHA(config.hidden_sizes[0])

        self.net = FFN(list(config.hidden_sizes) + [1])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        self.attn.reset_parameters()
        for seq in self.seqs:
            reset(seq)
        reset(self.net)
        self.net.seq[-2].weight.data.uniform_(-1e-2, 1e-2)

    def forward(self, pred_state, prey_state, obst_state, prey_is_alive, action):
        pos_pred = pred_state
        pos_prey = prey_state
        prey_is_dead = ~prey_is_alive

        pos_obst, r_obst = obst_state.split((2, 1), dim=-1)

        x_pred = self.seqs[0](torch.cat([pos_pred, action], dim=-1)) + self.embedding.weight[0]
        x_prey = self.seqs[1](pos_prey) + self.embedding.weight[1]
        x_obst = self.seqs[2](obst_state) + self.embedding.weight[2]

        x = torch.cat([x_pred, x_prey, x_obst], dim=1)
        pos = torch.cat([pos_pred, pos_prey, pos_obst], dim=1)

        mask = torch.cat([pos_pred.new_zeros(pos_pred.size()[:2], dtype=torch.bool),
                          prey_is_dead,
                          pos_obst.new_zeros(pos_obst.size()[:2], dtype=torch.bool)], dim=1)
        mask = pos_pred.new_zeros(pos_pred.size()[:2], dtype=torch.bool).unsqueeze(-1) & mask.unsqueeze(-2)

        out = self.attn(x_pred, x, pos_pred, pos, mask)

        out = self.net(out)
        return out


class PreyAttnCritic(nn.Module):
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_sizes[0])
        self.obst_embedding = nn.Linear(1, config.hidden_sizes[0])

        self.seqs = nn.ModuleList([
            FFN([2] + [config.hidden_sizes[0]] * 2),
            FFN([3] + [config.hidden_sizes[0]] * 2),
            FFN([3] + [config.hidden_sizes[0]] * 2)
        ])

        self.attn = MHA(config.hidden_sizes[0])

        self.net = FFN(list(config.hidden_sizes) + [1])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        self.attn.reset_parameters()
        for seq in self.seqs:
            reset(seq)
        reset(self.net)
        self.net.seq[-2].weight.data.uniform_(-1e-2, 1e-2)

    def forward(self, pred_state, prey_state, obst_state, prey_is_alive, action):
        B = pred_state.size(0)

        pos_pred = pred_state
        pos_prey = prey_state
        prey_is_dead = ~prey_is_alive

        pos_obst, r_obst = obst_state.split((2, 1), dim=-1)

        x_pred = self.seqs[0](pos_pred) + self.embedding.weight[0]
        x_prey = self.seqs[1](torch.cat([pos_prey, action], dim=-1)) + self.embedding.weight[1]
        x_obst = self.seqs[2](obst_state) + self.embedding.weight[2]

        x = torch.cat([x_pred, x_prey, x_obst], dim=1)
        pos = torch.cat([pos_pred, pos_prey, pos_obst], dim=1)

        mask = torch.cat([pos_pred.new_zeros(pos_pred.size()[:2], dtype=torch.bool),
                          prey_is_dead,
                          pos_obst.new_zeros(pos_obst.size()[:2], dtype=torch.bool)], dim=1)
        mask = prey_is_dead.unsqueeze(-1) & mask.unsqueeze(-2)

        out = self.attn(x_prey, x, pos_prey, pos, mask)
        out[prey_is_dead].fill_(0.)

        out = self.net(out)
        return out
