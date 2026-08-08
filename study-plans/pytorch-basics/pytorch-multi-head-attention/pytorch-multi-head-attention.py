import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Returns: None
        """
        super().__init__()

        self.D = d_model
        self.H = num_heads

        self.W_q = nn.Parameter(torch.randn(self.D, self.D))
        self.W_k = nn.Parameter(torch.randn(self.D, self.D))
        self.W_v = nn.Parameter(torch.randn(self.D, self.D))
        self.W_o = nn.Parameter(torch.randn(self.D, self.D))

    def forward(self, Q, K, V):
        """
        Returns: output tensor
        """
        N, T_q, D = Q.size()
        _, T_k, _ = K.size()

        # (N, T, D)
        Q = Q @ self.W_q
        K = K @ self.W_k
        V = V @ self.W_v

        # (N, H, T, D_h)
        Q = Q.reshape(N, T_q, self.H, -1).transpose(1, 2)
        K = K.reshape(N, T_k, self.H, -1).transpose(1, 2)
        V = V.reshape(N, T_k, self.H, -1).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) # (N, H, T_q, T_k)
        scores /= math.sqrt(self.D / self.H)
        weights = torch.softmax(scores, dim=-1)
        attn = weights @ V # (N, H, T_q, D_h)
        concat = attn.transpose(1, 2).reshape(N, T_q, D) # (N, T, D)
        return concat @ self.W_o