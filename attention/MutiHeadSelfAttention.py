import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.W_q = nn.Linear(25, in_dim).double()
        self.W_k = nn.Linear(25, in_dim).double()
        self.W_v = nn.Linear(25, in_dim).double()

        self.W_o = nn.Linear(in_dim, in_dim).double()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # print("Input size:", x.size()) 
        # print("W_q weight size:", self.W_q.weight.size())

        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # print("q size:", q.size())
        # print("k size:", k.size())
        # print("v size:", v.size())
        

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_dim)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_dim)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.head_dim)

        attn_scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attn_output = torch.matmul(attn_scores, v)

        attn_output = attn_output.view(self.num_heads, batch_size, seq_len, self.head_dim).permute(1, 2, 0, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        # print("attn_output size:", attn_output.size())

        attn_output = self.W_o(attn_output)

        return attn_output