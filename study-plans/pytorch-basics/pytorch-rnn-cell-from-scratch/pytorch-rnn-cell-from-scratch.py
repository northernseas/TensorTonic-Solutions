import torch
import torch.nn as nn

class RNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        """
        Returns: None
        """
        super().__init__()

        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_prev):
        """
        Returns: new hidden state tensor
        """
        return torch.tanh(
            x @ self.W_ih.T + self.b_ih +
            h_prev @ self.W_hh.T + self.b_hh
        )