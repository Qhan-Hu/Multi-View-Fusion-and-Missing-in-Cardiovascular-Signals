import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_

class CosinePositionEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super(CosinePositionEncoding, self).__init__()

        pos_embed = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0,seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2)*-(math.log(10000.0) / embed_dim))
        pos_embed[:, 0::2] = torch.sin(position * div_term)/math.sqrt(embed_dim)
        pos_embed[:, 1::2] = torch.cos(position * div_term)/math.sqrt(embed_dim)
        self.pos_embed = pos_embed.unsqueeze(0)

    def forward(self, x):
        B, _, _ = x.shape
        x = x + self.pos_embed.expand(B, -1, -1).to(x.device)
        return x
    

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, _, _ = x.shape
        x = x + self.pos_embed.expand(B, -1, -1)
        return x