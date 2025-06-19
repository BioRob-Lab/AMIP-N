import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DWT import *

class WaveletMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads,wavename='haar'):
        super(WaveletMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // (num_heads*2)
        
        # Wavelet transform layers
        self.wavelet = DWT_1D(wavename=wavename).double()
        self.iwavelet = IDWT_1D(wavename=wavename).double()
        
        self.W_q = nn.Linear(in_dim, in_dim).double()
        self.W_k = nn.Linear(in_dim, in_dim).double()
        self.W_v = nn.Linear(in_dim, in_dim).double()
        
        self.W_o = nn.Linear(in_dim, in_dim).double()  # Combine low and high frequency parts
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Perform linear transformation on queries, keys, and values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # Perform wavelet transform on queries, keys, and values
        q_L, q_H = self.wavelet(q)
        k_L, k_H = self.wavelet(k)
        v_L, v_H = self.wavelet(v)
        
        L = F.softmax(q_L + k_L, dim=-1)
        H = F.softmax(q_H + k_H, dim=-1)
        L_out = L = F.softmax(L + v_L, dim=-1)
        H_out = L = F.softmax(H + v_H, dim=-1)

        output = self.iwavelet(L_out,H_out)
        # Apply attention scores to values
        # attn_scores = F.softmax(torch.cat((L_out, H_out), dim=-1), dim=-1)
        
        # Reshape and linear transformation for output
        attn_output = output.view(batch_size, seq_len, -1)
        output = attn_output
        
        return output
    
    