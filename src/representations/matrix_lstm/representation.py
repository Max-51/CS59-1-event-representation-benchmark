
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class MatrixLSTMSurface(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.shared_lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bias=True,
        )
    def forward(self, sequences, lengths):
        packed = pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.shared_lstm(packed)
        return h_n[-1]   # (N_pix, hidden_size)
