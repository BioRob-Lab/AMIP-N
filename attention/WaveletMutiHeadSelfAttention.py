import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Utility Functions and Classes ###

def get_filter(base, k):
    # Placeholder function for getting filter coefficients
    # Replace this with actual implementation or filter coefficients
    # Here is a simplified example:
    if base == 'legendre':
        # Example coefficients (random initialization)
        H0 = np.random.randn(k, k)
        H1 = np.random.randn(k, k)
        G0 = np.random.randn(k, k)
        G1 = np.random.randn(k, k)
        PHI0 = np.random.randn(k, k)
        PHI1 = np.random.randn(k, k)
    else:
        raise ValueError(f"Unsupported base '{base}'")
    
    return H0, H1, G0, G1, PHI0, PHI1

### Multi-wavelet Transform Classes ###

class sparseKernelFT1d(nn.Module):
    def __init__(self, k, alpha, c=1):
        super(sparseKernelFT1d, self).__init__()
        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k

    def compl_mul1d(self, x, weights):
        return torch.einsum("bix,iox->box", x, weights)

    def forward(self, x):
        B, N, c, k = x.shape
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        l = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x

class MWT_CZ1d(nn.Module):
    def __init__(self, k=3, alpha=64, L=0, c=1, base='legendre'):
        super(MWT_CZ1d, self).__init__()
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1
        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
        self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):
        B, N, c, k = x.shape
        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_x = x[:, 0:nl - N, :, :]
        x = torch.cat([x, extra_x], 1)
        Ud = []
        Us = []
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud.append(self.A(d) + self.B(x))
            Us.append(self.C(d))
        x = self.T0(x)
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :N, :, :]
        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)
        x = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x

class MultiWaveletTransform(nn.Module):
    def __init__(self, ich=1, k=8, alpha=16, c=128, nCZ=1, L=0, base='legendre'):
        super(MultiWaveletTransform, self).__init__()
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.MWT_CZ = nn.ModuleList(MWT_CZ1d(k, alpha, L, c, base) for i in range(nCZ))

    def forward(self, x, attn_mask=None):
        B, L, H, E = x.shape
        x = x.view(B, L, -1)
        V = self.Lk0(x).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, -1, E)
        return V

class MultiWaveletMutiHeadSelfAttention(nn.Module):
    """
    Multi-head Self-Attention with Multi-wavelet Transform.
    """

    def __init__(self, num_heads=8, ich=1, k=8, alpha=16, c=128, nCZ=1, L=0, base='legendre', attention_dropout=0.1):
        super(MultiWaveletMutiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

        # Multi-wavelet transform blocks
        self.multiwavelet_transform = MultiWaveletTransform(ich=ich, k=k, alpha=alpha, c=c, nCZ=nCZ, L=L, base=base)

        # Multi-head self-attention
        self.self_attention = MultiHeadSelfAttention(num_heads=num_heads, input_dim=c * k)

        # Linear transformation
        self.linear = nn.Linear(c * k, ich)

    def forward(self, x, mask=None):
        # Apply multi-wavelet transform
        wavelet_output = self.multiwavelet_transform(x, mask)

        # Multi-head self-attention
        attn_output = self.self_attention(wavelet_output, wavelet_output, wavelet_output, mask)

        # Linear transformation
        output = self.linear(attn_output)

        return output, None

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head Self-Attention layer.
    """

    def __init__(self, num_heads=8, input_dim=512, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Define layers for linear transformation of queries, keys, and values
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output linear layer
        self.out = nn.Linear(input_dim, input_dim)

    def forward(self, q, k, v, mask=None):
        # Linear transformation for queries, keys, and values
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # Splitting into multiple heads
        q = self.split_heads(q, self.num_heads)
        k = self.split_heads(k, self.num_heads)
        v = self.split_heads(v, self.num_heads)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        attention_output = torch.matmul(attention_weights, v)

        # Concatenating heads
        attention_output = self.combine_heads(attention_output)

        # Linear transformation
        attention_output = self.out(attention_output)

        return attention_output

    def split_heads(self, x, num_heads):
        # Split the last dimension into (num_heads, head_dim)
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the heads back together
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)


