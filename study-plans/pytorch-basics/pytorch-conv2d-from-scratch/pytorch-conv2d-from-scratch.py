import torch
import torch.nn as nn

class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Returns: None
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight)

        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        """
        Returns: convolved output tensor of shape (batch, out_channels, H-k+1, W-k+1)
        """
        N, _, H, W = x.size()
        D1 = self.in_channels
        D2 = self.out_channels
        K = self.kernel_size

        out_h = H - K + 1
        out_w = W - K + 1
        out = torch.empty(
            N, D2, out_h, out_w,
            dtype=x.dtype,
            device=x.device
        )

        # for d2 in range(D2):
        #     for i in range(out_h):
        #         for j in range(out_w):
        #             patch = x[:, :, i:i+K, j:j+K] # (N, D1, K, K)
        #             kernel = self.weight[d2] # (D1, K, K)
        #             res = (patch * kernel).sum(dim=(1, 2, 3)) # (N,)
        #             out[:, d2, i, j] = res + self.bias[d2]

        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, :, i:i+K, j:j+K].reshape(N, -1) # (N, -1)
                kernel = self.weight.reshape(D2, -1) # (D2, -1)
                out[:, :, i, j] = patch @ kernel.T + self.bias

        return out