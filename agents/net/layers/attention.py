import torch
from torch import nn
from torch.nn import Conv2d, MultiheadAttention, Sequential as Seq, LeakyReLU as LReLU, Linear as Lin, init

from ...utils import reset


class MHA(nn.Module):
    def __init__(self, hs, num_heads=4, conv_input_size=2):
        super().__init__()
        self.num_heads = num_heads
        self.attn = MultiheadAttention(hs, num_heads)

        self.conv = Seq(Conv2d(conv_input_size, hs, 1), LReLU(0.1), Conv2d(hs, num_heads, 1))

        self.pos_enc = Seq(Lin(2, hs), LReLU(0.1), Lin(hs, hs))

        self.out_seq = Seq(Lin(hs, 4 * hs), LReLU(0.1), Lin(4 * hs, hs))

        self.ln1 = nn.LayerNorm(hs)
        self.ln2 = nn.LayerNorm(hs)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.conv)
        reset(self.pos_enc)
        reset(self.out_seq)

        self.attn._reset_parameters()
        init.normal_(self.ln1.weight, mean=1., std=0.01)
        init.zeros_(self.ln1.bias)
        init.normal_(self.ln2.weight, mean=1., std=0.01)
        init.zeros_(self.ln2.bias)

    def forward(self, e_x, e_all, x_pos, all_pos, mask):
        B, L, C = e_x.size()
        _, S, _ = e_all.size()

        e_x += self.pos_enc(x_pos)
        e_all += self.pos_enc(all_pos)
        dm = (x_pos.unsqueeze(2) - all_pos.unsqueeze(1)).permute(0, 3, 1, 2)

        attn_mask = self.conv(dm).reshape(-1, self.num_heads, L, S)

        attn_mask = attn_mask.masked_fill_(mask.unsqueeze(1), -float('inf'))
        e_all = e_all.transpose(0, 1)
        out = self.attn(e_x.transpose(0, 1), e_all, e_all,
                        attn_mask=attn_mask.reshape(-1, L, S))[0].transpose(0, 1)

        out1 = self.ln1(e_x + out)
        out2 = self.out_seq(out1)
        out = self.ln2(out2 + out1)
        return out
