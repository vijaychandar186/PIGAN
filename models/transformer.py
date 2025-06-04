import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """Transformer model for sequence classification."""
    def __init__(self, input_dim: int, output_dim: int, dropout_p: float):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = 128
        self.dropout = nn.Dropout(dropout_p)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fnn = nn.Linear(input_dim, output_dim)

    def forward(self, input_seq: torch.Tensor, hidden_state: torch.Tensor) -> tuple:
        out = self.dropout(input_seq)
        out = self.transformer_encoder(out)
        out = out[-1, :, :]
        out = self.fnn(out)
        return out, out

    def init_hidden(self, batch_size: int) -> tuple:
        h_init = torch.zeros(1, batch_size, self.hidden_size)
        c_init = torch.zeros(1, batch_size, self.hidden_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h_init = h_init.to(device)
        c_init = c_init.to(device)
        return (h_init, c_init)