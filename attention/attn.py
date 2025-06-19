import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import List, Tuple
from models.DWT import *

import numpy as np

import math
from math import sqrt
from math import sqrt, ceil
from math import log
from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class LogSparceAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=16,mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 c=64,k=8, ich=512, L=0, base='legendre', mode_select_method='random',initializer=None, activation='tanh'):
        super(LogSparceAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def log_mask(self, win_len, sub_len):
        mask = torch.zeros((win_len, win_len), dtype=torch.float)
        for i in range(win_len):
            mask[i] = self.row_mask(i, sub_len, win_len)
        return mask.view(1, 1, mask.size(0), mask.size(1))

    def row_mask(self, index, sub_len, win_len):
        log_l = math.ceil(np.log2(sub_len))

        mask = torch.zeros((win_len), dtype=torch.float)
        if((win_len // sub_len) * 2 * (log_l) > index):
            mask[:(index + 1)] = 1
        else:
            while(index >= 0):
                if((index - log_l + 1) < 0):
                    mask[:index] = 1
                    break
                mask[index - log_l + 1:(index + 1)] = 1  # Local attention
                for i in range(0, log_l):
                    new_index = index - log_l + 1 - 2**i
                    if((index - new_index) <= sub_len and new_index >= 0):
                        mask[new_index] = 1
                index -= sub_len
        return mask

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # if self.mask_flag:
            # if attn_mask is None:
            #     attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # scores.masked_fill_(attn_mask.mask, -np.inf)
        mask = self.log_mask(L, S)
        mask_tri = mask[:, :, :scores.size(-2), :scores.size(-1)]
        scores = scores.to(queries.device)
        mask_tri = mask_tri.to(queries.device)
        scores = scores * mask_tri + -1e9 * (1 - mask_tri)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=16, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 c=64,k=8, ich=512, L=0, base='legendre', mode_select_method='random',initializer=None, activation='tanh' ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn


class FourierProbAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, modes=16, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 c=64,k=8, ich=512, L=0, base='legendre', mode_select_method='random',initializer=None, activation='tanh'):
        super(FourierProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # 傅里叶变换所需的参数
        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index = self.get_frequency_modes(seq_len, modes=modes)
        self.scale_factor = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            self.scale_factor * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    def get_frequency_modes(self, seq_len, modes=16, mode_select_method='random'):
        if mode_select_method == 'random':
            return np.random.choice(seq_len // 2, modes, replace=False)
        elif mode_select_method == 'top':
            return np.arange(modes)
        else:
            raise ValueError("mode_select_method should be 'random' or 'top'")

    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        Q_K_sample_real = Q_K_sample.real
        Q_K_sample_imag = Q_K_sample.imag

        M_real = Q_K_sample_real.max(-1)[0] - torch.div(Q_K_sample_real.sum(-1), L_K)
        M_imag = Q_K_sample_imag.max(-1)[0] - torch.div(Q_K_sample_imag.sum(-1), L_K)

        M = M_real + M_imag
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert(L_Q == L_V)
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn_real = torch.softmax(scores.real, dim=-1)  # Softmax on real part
        attn_imag = torch.softmax(scores.imag, dim=-1)  # Softmax on imaginary part

        context_in[torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                index, :] = torch.matmul(attn_real, V.real).type_as(context_in) + \
                                1j * torch.matmul(attn_imag, V.imag).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn_real).to(attn_real.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.complex(attn_real, attn_imag)
            return (context_in, attns)
        else:
            return (context_in, None)


    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # Apply Fourier Transform to queries, keys, and values
        queries_ft = torch.fft.rfft(queries, dim=-1)
        keys_ft = torch.fft.rfft(keys, dim=-1)
        values_ft = torch.fft.rfft(values, dim=-1)

        # Compute the probabilistic attention in the frequency domain
        U_part = self.factor * ceil(np.log(L_K)) # c*ln(L_k)
        u = self.factor * ceil(np.log(L_Q)) # c*ln(L_q)

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        scores_top, index = self._prob_QK(queries_ft, keys_ft, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values_ft, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values_ft, scores_top, index, L_Q, attn_mask)

        # Apply Inverse Fourier Transform to context
        context = torch.fft.irfft(context, n=queries.size(-1))

        return context.contiguous(), attn
    

class MultiWaveletProbAttention(nn.Module):
    """
    1D Multiwavelet Probabilistic Attention layer.
    """
    def __init__(self, in_channels, out_channels, seq_len, modes=16, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 c=64,k=8, ich=200, L=0, base='legendre', mode_select_method='random',initializer=None, activation='tanh' ):
        super(MultiWaveletProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # Define your wavelet transformation parameters
        self.c = c
        self.k = k
        self.L = L
        self.base = base
        self.mode_select_method = mode_select_method
        self.activation = activation

        # Probabilistic Attention components
        self.L_Q = seq_len
        self.L_K = seq_len
        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize wavelet transformation layers
        self.H0, self.H1, self.G0, self.G1, self.PHI0, self.PHI1 = get_filter(base, k)
        self.H0r = self.H0 @ self.PHI0
        self.G0r = self.G0 @ self.PHI0
        self.H1r = self.H1 @ self.PHI1
        self.G1r = self.G1 @ self.PHI1

        self.H0r[np.abs(self.H0r) < 1e-8] = 0
        self.H1r[np.abs(self.H1r) < 1e-8] = 0
        self.G0r[np.abs(self.G0r) < 1e-8] = 0
        self.G1r[np.abs(self.G1r) < 1e-8] = 0

        self.attn_dropout = attention_dropout
        self.Wq = nn.Linear(ich, c * k)
        self.Wk = nn.Linear(ich, c * k)
        self.Wv = nn.Linear(ich, c * k)
        self.Wo = nn.Linear(c * k, out_channels)


    def wavelet_transform(self, x):
            print(x.shape)
            B, L, H, E = x.shape  # Batch, Length, Hidden, Embedding
            x = x.view(B, L, -1)
            q = self.Wq(x)
            k = self.Wk(x)
            v = self.Wv(x)
            return q.view(B, L, self.c, self.k), k.view(B, L, self.c, self.k), v.view(B, L, self.c, self.k)
    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        q_wavelet, k_wavelet, v_wavelet = self.wavelet_transform(queries),self.wavelet_transform(keys),self.wavelet_transform(values)

        print(f"q_wavelet shape: {q_wavelet.shape}, k_wavelet shape: {k_wavelet.shape}, v_wavelet shape: {v_wavelet.shape}")

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(q_wavelet, k_wavelet, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn



class WaveletMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads, wavename='haar', attention_dropout=0.1, output_attention=False):
        super(WaveletMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // (num_heads * 2)
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # Wavelet transform layers
        self.wavelet = DWT_1D(wavename=wavename).double()
        self.iwavelet = IDWT_1D(wavename=wavename).double()

        # self.W_q = nn.Linear(in_dim, in_dim).double()
        # self.W_k = nn.Linear(in_dim, in_dim).double()
        # self.W_v = nn.Linear(in_dim, in_dim).double()

        self.W_o = nn.Linear(in_dim, in_dim).double()  # Combine low and high frequency parts

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size, seq_len, _ = queries.size()

        # Perform linear transformation on queries, keys, and values
        q = queries
        k = keys
        v = values

        # Perform wavelet transform on queries, keys, and values
        q_L, q_H = self.wavelet(q)
        k_L, k_H = self.wavelet(k)
        v_L, v_H = self.wavelet(v)

        # Compute attention scores
        attn_L = torch.matmul(q_L, k_L.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.double))
        attn_H = torch.matmul(q_H, k_H.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.double))

        if attn_mask is not None:
            attn_L = attn_L.masked_fill(attn_mask == 0, -1e9)
            attn_H = attn_H.masked_fill(attn_mask == 0, -1e9)

        attn_L = F.softmax(attn_L, dim=-1)
        attn_H = F.softmax(attn_H, dim=-1)
        attn_L = self.dropout(attn_L)
        attn_H = self.dropout(attn_H)

        # Apply attention scores to values
        L_out = torch.matmul(attn_L, v_L)
        H_out = torch.matmul(attn_H, v_H)

        # Inverse wavelet transform to combine low and high frequency parts
        output = self.iwavelet(L_out, H_out)

        # Reshape and linear transformation for output
        attn_output = self.W_o(output)

        if self.output_attention:
            attn = (attn_L, attn_H)
        else:
            attn = None

        return attn_output.contiguous(), attn
        


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
