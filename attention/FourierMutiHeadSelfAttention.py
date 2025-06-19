# import torch
# import torch.nn as nn
# import numpy as np

# def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
#     """
#     Get modes on frequency domain:
#     'random' means sampling randomly;
#     'else' means sampling the lowest modes;
#     """
#     modes = min(modes, seq_len // 2)
#     if mode_select_method == 'random':
#         index = list(range(0, seq_len // 2))
#         np.random.shuffle(index)
#         index = index[:modes]
#     else:
#         index = list(range(0, modes))
#     index.sort()
#     return index

# class FourierMultiHeadSelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, seq_len, modes=40, mode_select_method='random', activation='softmax'):
#         super(FourierMultiHeadSelfAttention, self).__init__()
#         """
#         Fourier enhanced multi-head self-attention layer.
#         """
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.seq_len = seq_len
#         self.modes = modes
#         self.mode_select_method = mode_select_method
#         self.activation = activation
        
#         # Fourier modes for self-attention
#         self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
#         # print('modes={}, index={}'.format(modes, self.index))
        
#         # Scale for initialization
#         self.scale = (1 / (embed_dim * num_heads))
        
#         # Learnable parameters for attention
#         self.query_weights = nn.Parameter(
#             self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
#         self.key_weights = nn.Parameter(
#             self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
#         self.value_weights = nn.Parameter(
#             self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
        
#         # Output transformation weights
#         self.out_weights = nn.Parameter(
#             self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
        
#     def compl_mul1d(self, input, weights):
#         # Complex multiplication: (batch, heads, x, freqs) * (heads, x, freqs, out_freqs) -> (batch, heads, x, out_freqs)
#         return torch.einsum("bhxf,hfyo->bhxyo", input, weights)
    
#     # def compl_mul1d(self, input, weights):
#     # # Expand input to match the expected dimensions for einsum
#     #     input = input.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
#     #     return torch.einsum("bhxf,hfo->bhxo", input, weights)
#     # def compl_mul1d(self, input, weights):
#     # # 扩展输入张量以匹配 einsum 函数的预期维度
#     #     input = input.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
#     #     weights = weights.unsqueeze(0).expand(self.num_heads, -1, -1, -1)
#     #     return torch.einsum("bhxf,bhyf->bhxy", input, weights)

#     # def compl_mul1d(self, input, weights):
#     # # Expand input to match the expected dimensions for einsum
#     #     input = input.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
#     #     weights = weights.unsqueeze(0).expand(self.num_heads, weights.size(1), weights.size(2), weights.size(3))
#     #     return torch.einsum("bhxf,bhyf->bhxyf", input, weights)


#     def forward(self, x):
#         B, L, E = x.shape
#         x = x.permute(0, 2, 1)  # (B, E, L)
        
#         # Compute Fourier coefficients
#         x_ft = torch.fft.rfft(x, dim=-1)
        
#         # Perform Fourier-based multi-head self-attention
#         out_ft = torch.zeros(B, self.num_heads, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
#         for wi, i in enumerate(self.index):
#             q_ft = self.compl_mul1d(x_ft, self.query_weights[:, :, wi])
#             k_ft = self.compl_mul1d(x_ft, self.key_weights[:, :, wi])
#             v_ft = self.compl_mul1d(x_ft, self.value_weights[:, :, wi])
            
#             # Perform attention mechanism in frequency domain
#             attn_scores = torch.einsum("bhxy,bhyo->bhxo", q_ft, k_ft)
#             if self.activation == 'softmax':
#                 attn_scores = torch.softmax(abs(attn_scores), dim=-1)
#                 attn_scores = torch.complex(attn_scores, torch.zeros_like(attn_scores))
#             elif self.activation == 'tanh':
#                 attn_scores = attn_scores.tanh()
#             else:
#                 raise Exception('{} activation function is not implemented'.format(self.activation))
            
#             xqkv_ft = torch.einsum("bhxo,bhyo->bhxyo", attn_scores, v_ft)
#             out_ft[:, :, :, i] = torch.einsum("bhxyo,hfyo->bhxf", xqkv_ft, self.out_weights[:, :, wi])
        
#         # Return to time domain
#         out = torch.fft.irfft(out_ft, n=L, dim=-2)
#         return out.permute(0, 2, 1)  # (B, L, E)

#     # def forward(self, x):
#     #     B, L, E = x.shape
#     #     x = x.permute(0, 2, 1)  # (B, E, L)
        
#     #     # Compute Fourier coefficients
#     #     x_ft = torch.fft.rfft(x, dim=-1)
        
#     #     # Perform Fourier-based multi-head self-attention
#     #     out_ft = torch.zeros(B, self.num_heads, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
#     #     for wi, i in enumerate(self.index):
#     #         q_ft = self.compl_mul1d(x_ft, self.query_weights[:, :, wi])
#     #         k_ft = self.compl_mul1d(x_ft, self.key_weights[:, :, wi])
#     #         v_ft = self.compl_mul1d(x_ft, self.value_weights[:, :, wi])
            
#     #         # Perform attention mechanism in frequency domain
#     #         attn_scores = torch.einsum("bhxo,bhyo->bhxo", q_ft, k_ft)
#     #         if self.activation == 'softmax':
#     #             attn_scores = torch.softmax(attn_scores.abs(), dim=-1)
#     #             attn_scores = torch.complex(attn_scores, torch.zeros_like(attn_scores))
#     #         elif self.activation == 'tanh':
#     #             attn_scores = attn_scores.tanh()
#     #         else:
#     #             raise Exception('{} activation function is not implemented'.format(self.activation))
            
#     #         xqkv_ft = torch.einsum("bhxo,bhyo->bhxo", attn_scores, v_ft)
#     #         out_ft[:, :, :, i] = torch.einsum("bhxo,hfo->bhxf", xqkv_ft, self.out_weights[:, :, wi])
        
#     #     # Return to time domain
#     #     out = torch.fft.irfft(out_ft, n=L, dim=-2)
#     #     return out.permute(0, 2, 1)  # (B, L, E)

import torch
import torch.nn as nn
import numpy as np

def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    Get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index

# class FourierMultiHeadSelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, seq_len, modes=40, mode_select_method='random', activation='softmax'):
#         super(FourierMultiHeadSelfAttention, self).__init__()
#         """
#         Fourier enhanced multi-head self-attention layer.
#         """
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.seq_len = seq_len
#         self.modes = modes
#         self.mode_select_method = mode_select_method
#         self.activation = activation
        
#         # Fourier modes for self-attention
#         self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        
#         # Scale for initialization
#         self.scale = (1 / (embed_dim * num_heads))
        
#         # Learnable parameters for attention
#         self.query_weights = nn.Parameter(
#             self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
#         self.key_weights = nn.Parameter(
#             self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
#         self.value_weights = nn.Parameter(
#             self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
        
#         # Output transformation weights
#         self.out_weights = nn.Parameter(
#             self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
        
#     # def compl_mul1d(self, input, weights):
#     #     # Complex multiplication: (batch, heads, x, freqs) * (heads, x, freqs, out_freqs) -> (batch, heads, x, out_freqs)
#     #     return torch.einsum("bhxf,hfyo->bhxyo", input, weights)
#     def compl_mul1d(self, input, weights):
#         # Reshape weights to match the einsum operation
#         weights = weights.unsqueeze(1).unsqueeze(2)  # shape: [8, 1, 1, 5]
        
#         # Complex multiplication: (batch, heads, x, freqs) * (heads, x, freqs, out_freqs) -> (batch, heads, x, out_freqs)
#         return torch.einsum("bhxf,hxfo->bhxo", input, weights)


#     def forward(self, x):
#         B, L, E = x.shape
#         x = x.permute(0, 2, 1)  # (B, E, L)
        
#         # Split input into num_heads parts
#         x = x.reshape(B, self.num_heads, E, L // self.num_heads)
        
#         # Compute Fourier coefficients for each head
#         x_ft = torch.fft.rfft(x, dim=-1)
        
#         # Perform Fourier-based multi-head self-attention
#         out_ft = torch.zeros(B, self.num_heads, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
#         for wi, i in enumerate(self.index):
#             q_ft = self.compl_mul1d(x_ft, self.query_weights[:, :, wi])
#             k_ft = self.compl_mul1d(x_ft, self.key_weights[:, :, wi])
#             v_ft = self.compl_mul1d(x_ft, self.value_weights[:, :, wi])
            
           
#             # Perform attention mechanism in frequency domain
#             attn_scores = torch.einsum("bhxy,bhyo->bhxo", q_ft, k_ft)
#             if self.activation == 'softmax':
#                 attn_scores = torch.softmax(abs(attn_scores), dim=-1)
#                 attn_scores = torch.complex(attn_scores, torch.zeros_like(attn_scores))
#             elif self.activation == 'tanh':
#                 attn_scores = attn_scores.tanh()
#             else:
#                 raise Exception('{} activation function is not implemented'.format(self.activation))
            
#             xqkv_ft = torch.einsum("bhxo,bhyo->bhxyo", attn_scores, v_ft)
#             out_ft[:, :, :, i] = torch.einsum("bhxyo,hfyo->bhxf", xqkv_ft, self.out_weights[:, :, wi])
        
#         # Return to time domain
#         out = torch.fft.irfft(out_ft, n=L, dim=-2)
#         return out.permute(0, 2, 1)  # (B, L, E)


import torch
import torch.nn as nn

class FourierMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, modes=40, mode_select_method='random', activation='softmax'):
        super(FourierMultiHeadSelfAttention, self).__init__()
        """
        Fourier enhanced multi-head self-attention layer.
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.modes = modes
        self.mode_select_method = mode_select_method
        self.activation = activation
        
        # Fourier modes for self-attention
        self.index = torch.tensor(get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method))  # Ensure index is a tensor
        
        # Scale for initialization
        self.scale = (1 / (embed_dim * num_heads))
        
        # Learnable parameters for attention
        self.query_weights = nn.Parameter(
            self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
        self.key_weights = nn.Parameter(
            self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
        self.value_weights = nn.Parameter(
            self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
        
        # Output transformation weights
        self.out_weights = nn.Parameter(
            self.scale * torch.rand(num_heads, embed_dim // num_heads, len(self.index), dtype=torch.cfloat))
        
    def compl_mul1d(self, input, weights):
        # Complex multiplication: (batch, heads, x, freqs) * (heads, x, freqs, out_freqs) -> (batch, heads, x, out_freqs)
        return torch.matmul(input.unsqueeze(-2), weights.unsqueeze(-1)).squeeze(-1)

    def forward(self, x):
        B, L, E = x.shape
        x = x.permute(0, 2, 1)  # (B, E, L)
        
        # Split input into num_heads parts
        x = x.reshape(B, self.num_heads, E, L // self.num_heads)
        
        # Compute Fourier coefficients for each head
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Perform Fourier-based multi-head self-attention
        out_ft = torch.zeros(B, self.num_heads, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        
        # Compute attention for each mode
        for wi, i in enumerate(self.index):
            q_ft = self.compl_mul1d(x_ft, self.query_weights[:, :, wi])
            k_ft = self.compl_mul1d(x_ft, self.key_weights[:, :, wi])
            v_ft = self.compl_mul1d(x_ft, self.value_weights[:, :, wi])
            
            # Perform attention mechanism in frequency domain
            attn_scores = torch.matmul(q_ft.conj(), k_ft.transpose(-2, -1))  # Shape: (B, num_heads, L // num_heads, L // num_heads)
            
            # Apply softmax activation along the frequency dimension
            attn_scores = torch.softmax(torch.abs(attn_scores), dim=-1)
            
            # Weighted sum using the attention scores
            xqkv_ft = torch.matmul(attn_scores, v_ft.transpose(-2, -1))
            
            # Apply output transformation
            out_ft[:, :, :, i] = torch.matmul(xqkv_ft, self.out_weights[:, :, wi].transpose(-2, -1))
        
        # Return to time domain
        out = torch.fft.irfft(out_ft, n=L, dim=-2)
        return out.permute(0, 2, 1)  # (B, L, E)
