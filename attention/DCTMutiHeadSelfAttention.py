import torch
from torch import nn
import numpy as np
# import utils.util as util

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


class DiscreteCosineMultiHeadSelfAttention(nn.Module):
    def __init__(self, in_features, kernel_size=10, d_model=512, num_heads=8, dct_n=10, dropout=0.1):
        super(DiscreteCosineMultiHeadSelfAttention, self).__init__()

        self.in_features = in_features
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dct_n = dct_n
        self.dropout = dropout

        # Multi-head attention components
        self.q_linear = nn.Linear(in_features, d_model).double()
        self.k_linear = nn.Linear(in_features, d_model).double()
        self.v_linear = nn.Linear(in_features, d_model).double()

        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout).double()

        # Discrete Cosine Transform matrices
        dct_m, idct_m = get_dct_matrix(kernel_size)
        # print(dct_m.shape)
        self.register_buffer('dct_m', torch.from_numpy(dct_m).double())  # Register as buffer to move to GPU
        self.register_buffer('idct_m', torch.from_numpy(idct_m).double())  # Register as buffer to move to GPU

    def forward(self, src):
        bs, seq_len, _ = src.size()
        # print(self.dct_m.shape)
        # print(src.shape)

        src_dct = torch.matmul(src, self.dct_m[:self.dct_n, :].unsqueeze(dim=0)).to(torch.double)

        # Discrete Cosine Transform
        # src_dct = torch.matmul(src, self.dct_m[:self.dct_n].unsqueeze(dim=0))

        # Multi-head attention
        query = self.q_linear(src_dct).transpose(0, 1)  # (seq_len, bs, d_model)
        key = self.k_linear(src_dct).transpose(0, 1)    # (seq_len, bs, d_model)
        value = self.v_linear(src_dct).transpose(0, 1)  # (seq_len, bs, d_model)

        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(query, key, value)
        # print(attn_output.transpose(0s

        # Discrete Cosine Transform for attention output
        # attn_output_dct = torch.matmul(attn_output.transpose(0, 1), self.dct_m[:self.dct_n].unsqueeze(dim=0))



        # Inverse Discrete Cosine Transform
        attn_output_idct = torch.matmul(attn_output, self.idct_m[:, :self.dct_n].unsqueeze(dim=0))

        return attn_output_idct.transpose(0, 1)  # (bs, seq_len, d_model)


