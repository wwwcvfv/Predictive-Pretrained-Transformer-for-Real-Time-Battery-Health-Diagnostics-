import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, x, x_c, x_mask=None, cross_mask=None):
        x_, _ = self.self_attn(x, x, x, attn_mask=x_mask)
        x = x + self.dropout1(x_)
        x = self.norm1(x)

        x_, _ = self.multihead_attn(x, x_c, x_c, attn_mask=cross_mask)
        x = x + self.dropout2(x_)
        x = self.norm2(x)

        x_ = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x_)
        x= self.norm3(x)

        return x
