import torch
import torch.nn as nn


class MutaGANEncoder(nn.Module):
    """
    Bidirectional LSTM encoder that concatenates forward and backward
    final hidden states. Matches MutaGAN reference architecture.
    """

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_dim, hidden_size, num_layers,
            bidirectional=True, batch_first=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, input_dim)
        _, (h, _) = self.rnn(x)
        # Concatenate final forward and backward hidden states → (batch, hidden*2)
        return torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)


class MutaGANModel(nn.Module):
    """
    MutaGAN discriminator adapted for continuous ProtVec embeddings.

    Architecture matches reference define_discriminator:
        BiLSTM encoder
        → concat fwd+bwd hidden (hidden*2)
        → Dropout(0.2) → BatchNorm
        → Dense(128) → LeakyReLU(0.1)
        → Dropout(0.2) → BatchNorm
        → Dense(64) → LeakyReLU(0.1)
        → Dropout(0.2) → BatchNorm
        → Dense(output_dim)

    Determines whether a parent-child sequence pair is a valid
    evolutionary relationship (mutation prediction).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout_p: float,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device

        self.encoder = MutaGANEncoder(input_dim, hidden_size, num_layers)

        enc_dim = hidden_size * 2  # bidirectional concatenation

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.BatchNorm1d(enc_dim),
            nn.Linear(enc_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor, hidden_state=None) -> tuple:
        # x: (seq_len, batch, input_dim)
        encoded = self.encoder(x)          # (batch, hidden*2)
        logits = self.classifier(encoded)  # (batch, output_dim)
        return logits, None

    def init_hidden(self, batch_size: int):
        return None
