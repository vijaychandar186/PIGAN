import torch
import torch.nn as nn

class RNNModel(nn.Module):
    """Basic RNN model with LSTM, GRU, or simple RNN cells."""
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, dropout_p: float, cell_type: str):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dropout_p)
        if cell_type == 'LSTM':
            self.encoder = nn.LSTM(input_dim, hidden_size)
        elif cell_type == 'GRU':
            self.encoder = nn.GRU(input_dim, hidden_size)
        elif cell_type == 'RNN':
            self.encoder = nn.RNN(input_dim, hidden_size)
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, input_seq: torch.Tensor, hidden_state: torch.Tensor) -> tuple:
        input_seq = self.dropout(input_seq)
        encoder_outputs, _ = self.encoder(input_seq, hidden_state)
        scores = self.out(encoder_outputs[-1, :, :])
        dummy_weights = torch.zeros(input_seq.shape[1], input_seq.shape[0])
        return scores, dummy_weights

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        if self.cell_type == 'LSTM':
            h_init = torch.zeros(1, batch_size, self.hidden_size)
            c_init = torch.zeros(1, batch_size, self.hidden_size)
            return (h_init, c_init)
        return torch.zeros(1, batch_size, self.hidden_size)