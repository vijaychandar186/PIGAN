import torch
import torch.nn as nn

class EncoderForClassification(nn.Module):
    """Encoder for MutaGAN model with bidirectional LSTM."""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, _) = self.rnn(x)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return hidden_cat

class MutaGANModel(nn.Module):
    """MutaGAN model for mutation prediction."""
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, num_layers: int, dropout_p: float, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.encoder = EncoderForClassification(input_dim, hidden_size, num_layers, device)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.device = device

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None) -> tuple:
        x = x.permute(1, 0, 2)
        encoded = self.encoder(x)
        encoded = self.dropout(encoded)
        logits = self.fc(encoded)
        return logits, None

    def init_hidden(self, batch_size: int) -> None:
        return None