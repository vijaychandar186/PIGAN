import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    """
    A temporal attention model using an Transformer encoder.
    """

    def __init__(self, input_dim, output_dim, dropout_p, hidden_size: int = 256):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_p)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=5, dim_feedforward=hidden_size,
            dropout=0.1, batch_first=False, norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, input_seq, hidden_state):
        out = self.transformer_encoder(self.dropout(input_seq))
        out = self.classifier(out[-1, :, :])
        return out, out

    def init_hidden(self, batch_size):
        return None
