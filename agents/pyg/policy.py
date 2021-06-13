from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple, Union
from typing_extensions import Literal

import torch
from torch import Tensor, nn
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU as LReLU, init, ELU

from torch_geometric.nn import NNConv, GATConv

from ..net import FFN
from ..utils import reset


@dataclass
class ActorConfig:
    input_size: Union[int, Tuple[int, int, int]] = (2, 2, 3)
    hidden_sizes: Sequence[int] = (16, 64, 64)

    max_grad_norm: float = 0.5

    lr: float = 1e-3

    atan_trick: bool = False

    entity: Optional[Literal['predator', 'prey']] = None


class PredatorActor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.pos_embedding = FFN([3] + [config.hidden_sizes[0]] * 2, layer_norm=False, act=ELU())
        self.entity_embedding = nn.Embedding(3, config.hidden_sizes[0])

        self.edge_type_embedding = nn.Embedding(10, 10)

        self.conv1 = NNConv(
            config.hidden_sizes[0],
            config.hidden_sizes[0],
            nn=FFN([12, config.hidden_sizes[0], config.hidden_sizes[0] ** 2], layer_norm=False, act=ELU()),
            aggr='mean'
        )
        self.conv2 = GATConv(
            config.hidden_sizes[0],
            config.hidden_sizes[1],
            heads=4,
            concat=False,
            add_self_loops=True,
            negative_slope=0.1
        )

        self.ln = nn.LayerNorm(config.hidden_sizes[1], elementwise_affine=False)

        self.act = ELU()

        self.net = FFN(list(config.hidden_sizes[1:]) + [1], act=ELU())
        self.net.seq.add_module(str(len(self.net.seq)), nn.Softsign())

        self.atan_trick = config.atan_trick

        self.reset_parameters()

    def reset_parameters(self):
        self.pos_embedding.reset_parameters()
        nn.init.xavier_normal_(self.edge_type_embedding.weight)
        nn.init.xavier_normal_(self.entity_embedding.weight)
        self.conv1.reset_parameters()
        for m in self.conv1.nn.modules():
            if isinstance(m, Lin):
                init.xavier_uniform_(m.weight, 0.01)
                init.zeros_(m.bias)
        for p in self.conv2.parameters():
            if p.ndim == 1:
                init.zeros_(p)
            else:
                init.xavier_uniform_(p, 0.01)
        init.zeros_(self.conv2.bias)
        self.net.reset_parameters()
        self.net.seq[-2].weight.data.uniform_(-0.01, 0.01)

    def forward(self, state):
        x = state.x
        edge_index = state.edge_index
        row, col = edge_index
        edge_attr = state.edge_attr
        mask = state.mask
        # is_dead_mask = state.is_dead_mask

        pos = x[:, :2].clone()
        rel_coords = pos[row] - pos[col]
        e = torch.cat([self.edge_type_embedding(edge_attr), rel_coords], dim=1)

        x = self.pos_embedding(x) + self.entity_embedding(mask)

        x = self.act(self.conv1(x, edge_index, e))
        x = self.ln(self.conv2(x, edge_index))

        x = x[mask == 0]
        x = self.net(x)
        return x

    # def forward(self, state):
    #     out = self.get_components(state)
    #
    #     out = torch.atan2(*out.T).unsqueeze(-1)
    #     out /= math.pi
    #     return out


class PreyActor(nn.Module):
    def __init__(self, config: ActorConfig):
        super().__init__()
        self.pos_embedding = FFN([3] + [config.hidden_sizes[0]] * 2, layer_norm=False, act=ELU())
        self.entity_embedding = nn.Embedding(3, config.hidden_sizes[0])

        self.edge_type_embedding = nn.Embedding(10, 10)

        self.conv1 = NNConv(
            config.hidden_sizes[0],
            config.hidden_sizes[0],
            nn=FFN([12, config.hidden_sizes[0], config.hidden_sizes[0] ** 2], layer_norm=False, act=ELU()),
            aggr='mean'
        )
        self.conv2 = GATConv(
            config.hidden_sizes[0],
            config.hidden_sizes[1],
            heads=4,
            concat=False,
            add_self_loops=True,
            negative_slope=0.1
        )

        self.ln = nn.LayerNorm(config.hidden_sizes[1], elementwise_affine=False)

        self.act = ELU()

        self.net = FFN(list(config.hidden_sizes[1:]) + [1], act=ELU())
        self.net.seq.add_module(str(len(self.net.seq)), nn.Softsign())

        self.atan_trick = config.atan_trick

        self.reset_parameters()

    def reset_parameters(self):
        self.pos_embedding.reset_parameters()
        nn.init.xavier_normal_(self.edge_type_embedding.weight)
        nn.init.xavier_normal_(self.entity_embedding.weight)
        self.conv1.reset_parameters()
        for m in self.conv1.nn.modules():
            if isinstance(m, Lin):
                init.xavier_uniform_(m.weight, 0.01)
                init.zeros_(m.bias)
        for p in self.conv2.parameters():
            if p.ndim == 1:
                init.zeros_(p)
            else:
                init.xavier_uniform_(p, 0.01)
        init.zeros_(self.conv2.bias)
        self.net.reset_parameters()
        self.net.seq[-2].weight.data.uniform_(-0.01, 0.01)

    def forward(self, state):
        x = state.x
        edge_index = state.edge_index
        row, col = edge_index
        edge_attr = state.edge_attr
        mask = state.mask
        # is_dead_mask = state.is_dead_mask

        pos = x[:, :2].clone()
        rel_coords = pos[row] - pos[col]

        x = self.pos_embedding(x) + self.entity_embedding(mask)
        e = torch.cat([self.edge_type_embedding(edge_attr), rel_coords], dim=1)

        x = self.act(self.conv1(x, edge_index, e))
        x = self.ln(self.conv2(x, edge_index))

        x = x[mask == 1]
        x = self.net(x)
        return x

    # def forward(self, state):
    #     out = self.get_components(state)
    #
    #     out = torch.atan2(*out.T).unsqueeze(-1)
    #     out /= math.pi
    #     return out

