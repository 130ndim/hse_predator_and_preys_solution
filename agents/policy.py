from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple, Union
from typing_extensions import Literal

import torch
from torch import Tensor, nn

from .net import FFN
from .net.layers import PointConv, MHA
from .utils import reset


@dataclass
class ActorConfig:
    input_size: Union[int, Tuple[int, int, int]] = (2, 2, 3)
    hidden_sizes: Sequence[int] = (64, 64, 64)

    max_grad_norm: float = 0.5

    lr: float = 3e-4

    atan_trick: bool = False

    entity: Optional[Literal['predator', 'prey']] = None


class PredatorActor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_sizes[0])
        self.obst_embedding = nn.Linear(1, config.hidden_sizes[0])

        self.pos_embedding = nn.Linear(2, config.hidden_sizes[0])

        self.conv1 = PointConv(config.hidden_sizes[0])
        self.conv2 = PointConv(config.hidden_sizes[0])

        self.net = FFN(list(config.hidden_sizes) + [2])
        self.net.seq.add_module(str(len(self.net.seq)), nn.Tanh())

        self.atan_trick = config.atan_trick

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_normal_(self.obst_embedding.weight)
        nn.init.xavier_normal_(self.pos_embedding.weight)
        nn.init.zeros_(self.obst_embedding.bias)
        nn.init.zeros_(self.pos_embedding.bias)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.net.reset_parameters()
        self.net.seq[-2].weight.data.uniform_(-1e-2, 1e-2)

    def get_components(self, pred_state, prey_state, obst_state, prey_is_alive):
        B = pred_state.size(0)

        pos_pred = pred_state
        pos_prey = prey_state

        pos_obst, r_obst = obst_state.split((2, 1), dim=-1)

        x_pred = self.embedding.weight[0].repeat((B, pos_pred.size(1), 1))
        x_prey = self.embedding.weight[1].repeat((B, pos_prey.size(1), 1))
        x_obst = self.embedding.weight[2].repeat((B, pos_obst.size(1), 1)) + self.obst_embedding(r_obst)

        x_prey[~prey_is_alive].fill_(0.)
        out_pred, out_prey, out_obst = self.conv1(x_pred, x_prey, x_obst, pos_pred, pos_prey, pos_obst, prey_is_alive)
        out_prey[~prey_is_alive].fill_(0.)
        out = self.conv2(out_pred, out_prey, out_obst, pos_pred, pos_prey, pos_obst, prey_is_alive)[0]
        out = self.net(out + self.pos_embedding(pos_pred))
        return out

    def forward(self, pred_state, prey_state, obst_state, prey_is_alive):
        out = self.get_components(pred_state, prey_state, obst_state, prey_is_alive)

        out = torch.atan2(*out.permute(2, 0, 1)).unsqueeze(-1)
        out /= math.pi
        return out


class PreyActor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_sizes[0])
        self.obst_embedding = nn.Linear(1, config.hidden_sizes[0])

        self.pos_embedding = nn.Linear(2, config.hidden_sizes[0])

        self.conv1 = PointConv(config.hidden_sizes[0])
        self.conv2 = PointConv(config.hidden_sizes[0])

        self.net = FFN(list(config.hidden_sizes) + [2])
        self.net.seq.add_module(str(len(self.net.seq)), nn.Tanh())

        self.atan_trick = config.atan_trick

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_normal_(self.obst_embedding.weight)
        nn.init.xavier_normal_(self.pos_embedding.weight)
        nn.init.zeros_(self.obst_embedding.bias)
        nn.init.zeros_(self.pos_embedding.bias)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.net.reset_parameters()
        self.net.seq[-2].weight.data.uniform_(-1e-2, 1e-2)

    def get_components(self, pred_state, prey_state, obst_state, prey_is_alive):
        B = pred_state.size(0)

        pos_pred = pred_state
        pos_prey = prey_state

        pos_obst, r_obst = obst_state.split((2, 1), dim=-1)

        x_pred = self.embedding.weight[0].repeat((B, pos_pred.size(1), 1))
        x_prey = self.embedding.weight[1].repeat((B, pos_prey.size(1), 1))
        x_obst = self.embedding.weight[2].repeat((B, pos_obst.size(1), 1)) + self.obst_embedding(r_obst)

        x_prey[~prey_is_alive].fill_(0.)
        out_pred, out_prey, out_obst = self.conv1(x_pred, x_prey, x_obst, pos_pred, pos_prey, pos_obst, prey_is_alive)
        out_prey[~prey_is_alive].fill_(0.)
        out = self.conv2(out_pred, out_prey, out_obst, pos_pred, pos_prey, pos_obst, prey_is_alive)[1]
        out[~prey_is_alive].fill_(0.)

        out = self.net(out + self.pos_embedding(pos_prey))
        return out

    def forward(self, pred_state, prey_state, obst_state, prey_is_alive):
        out = self.get_components(pred_state, prey_state, obst_state, prey_is_alive)
        out = torch.atan2(*out.permute(2, 0, 1)).unsqueeze(-1)
        out /= math.pi
        return out


class PredatorAttnActor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_sizes[0])

        self.seqs = nn.ModuleList([
            FFN([2] + [config.hidden_sizes[0]] * 2),
            FFN([2] + [config.hidden_sizes[0]] * 2),
            FFN([3] + [config.hidden_sizes[0]] * 2)
        ])

        self.attn = MHA(config.hidden_sizes[0])

        self.net = FFN(list(config.hidden_sizes) + [1])
        self.net.seq.add_module(str(len(self.net.seq)), nn.Tanh())

        self.atan_trick = config.atan_trick

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        self.attn.reset_parameters()
        for seq in self.seqs:
            reset(seq)
        reset(self.net)
        self.net.seq[-2].weight.data.uniform_(-1e-2, 1e-2)

    def forward(self, pred_state, prey_state, obst_state, prey_is_alive):
        pos_pred = pred_state
        pos_prey = prey_state
        prey_is_dead = ~prey_is_alive

        pos_obst, r_obst = obst_state.split((2, 1), dim=-1)

        x_pred = self.seqs[0](pos_pred) + self.embedding.weight[0]
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


class PreyAttnActor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_sizes[0])

        self.seqs = nn.ModuleList([
            FFN([2] + [config.hidden_sizes[0]] * 2),
            FFN([2] + [config.hidden_sizes[0]] * 2),
            FFN([3] + [config.hidden_sizes[0]] * 2)
        ])

        self.attn = MHA(config.hidden_sizes[0])

        self.net = FFN(list(config.hidden_sizes) + [1])
        self.net.seq.add_module(str(len(self.net.seq)), nn.Tanh())

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)
        self.attn.reset_parameters()
        for seq in self.seqs:
            reset(seq)
        reset(self.net)
        self.net.seq[-2].weight.data.uniform_(-1e-2, 1e-2)

    def forward(self, pred_state, prey_state, obst_state, prey_is_alive):
        B = pred_state.size(0)

        pos_pred = pred_state
        pos_prey = prey_state
        prey_is_dead = ~prey_is_alive

        pos_obst, r_obst = obst_state.split((2, 1), dim=-1)

        x_pred = self.seqs[0](pos_pred) + self.embedding.weight[0]
        x_prey = self.seqs[1](pos_prey) + self.embedding.weight[1]
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

