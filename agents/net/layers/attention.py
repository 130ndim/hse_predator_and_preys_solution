import torch
from torch import nn
from torch.nn import Embedding as Emb, Conv2d, MultiheadAttention


class MHA(nn.Module):
    def __init__(self, hs, num_heads=4, conv_input_size=2):
        super().__init__()
        self.entity_emb = Emb(3, hs)

        self.attn = MultiheadAttention(hs, num_heads)

        self.conv = Conv2d(conv_input_size or hs, num_heads, 1)

        self.act = nn.LeakyReLU(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.attn._reset_parameters()

    def forward(self, x_pred, x_prey, x_obst, pos):
        sizes = torch.tensor([x_pred.size(1), x_prey.size(1), x_obst.size(1)], device=x_pred.device)

        x = torch.cat([x_pred, x_prey, x_obst], dim=1)

        B, N, C = x.size()

        dm = (pos.unsqueeze(2) - pos.unsqueeze(1)).permute(0, 3, 1, 2)

        attn_mask = self.conv(dm).sigmoid().reshape(-1, N, N)

        x = x + self.entity_emb.weight.repeat_interleave(sizes, dim=0)

        x = x.transpose(0, 1)

        x = x + self.act(self.attn(x, x, x, attn_mask=attn_mask)[0])

        out_pred, out_prey, out_obst = x.transpose(0, 1).split(sizes.tolist(), dim=1)

        return out_pred, out_prey, out_obst
