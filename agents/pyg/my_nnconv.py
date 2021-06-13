from typing import Union, Tuple, Callable
from torch_geometric.typing import OptTensor, OptPairTensor

from torch.nn import Parameter

from torch_geometric.typing import Adj, Size

import torch
from torch import Tensor, nn


class _NNConv(nn.Module):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, nn: Callable, aggr: str = 'add',
                 root_weight: bool = True, bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # self.reset_parameters()

    # def reset_parameters(self):
        # reset(self.nn)
        # if self.root is not None:
        #     uniform(self.root.size(0), self.root)
        # zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        row, col = edge_index

        e = self.nn(edge_attr).view(-1, self.in_channels, self.out_channels)
        x_j = x[row]
        x_i = x
        m = torch.matmul(x_j.unsqueeze(1), e).squeeze(1)
        x_m = torch.zeros_like(x)
        for i in range(col.max() + 1):
            x_m[i] = m[col == i].mean()

        if self.root is not None:
            x_i = x @ self.root

        if self.bias is not None:
            x_i = x_i + self.bias

        out = x_i + x_m

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        weight = self.nn(edge_attr)
        weight = weight.view(-1, self.in_channels_l, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def __repr__(self):
        return '{}({}, {}, aggr="{}", nn={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.aggr, self.nn)

if __name__ == "__main__":
    from torch_geometric.data import Data
    from torch_geometric.nn import NNConv
    from torch.nn import Linear
    with torch.no_grad():
        d = Data(x=torch.rand(4, 4),
                 edge_index=torch.tensor([[0, 1, 2, 0, 3, 2], [1, 0, 0, 2, 2, 3]], dtype=torch.long),
                 edge_attr=torch.rand(6, 1))

        net1 = NNConv(4, 4, Linear(1, 16, bias=False), aggr='mean')
        net2 = InferenceNNConv(4, 4, Linear(1, 16, bias=False),  aggr='mean')
        net1.root.data.fill_(1.)
        net2.root.data.fill_(1.)
        net1.nn.weight.data.fill_(1.)
        net2.nn.weight.data.fill_(1.)
        net1.bias.data.fill_(0.)
        net2.bias.data.fill_(0.)

        assert torch.all(net1(d.x, d.edge_index, d.edge_attr) == net2(d.x, d.edge_index, d.edge_attr))
