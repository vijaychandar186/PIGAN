import torch
import torch.nn as nn


class PIGANGenerator(nn.Module):
    """Transformer encoder produces a phylogenetic embedding that conditions an LSTM decoder."""

    def __init__(self, input_dim, hidden_size, num_layers, dropout_p,
                 nhead=5, transformer_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_p)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=hidden_size,
            dropout=dropout_p, batch_first=False, norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.h_proj = nn.Linear(input_dim, hidden_size)
        self.c_proj = nn.Linear(input_dim, hidden_size)
        self.decoder = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=False)
        self.out_proj = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        encoded = self.transformer_encoder(self.dropout(x))
        phylo_embed = encoded[-1, :, :]
        h0 = self.h_proj(phylo_embed).unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c0 = self.c_proj(phylo_embed).unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        last_input = x[-1, :, :].unsqueeze(0)
        dec_out, _ = self.decoder(last_input, (h0, c0))
        return self.out_proj(dec_out.squeeze(0))


class PIGANDiscriminator(nn.Module):
    """Transformer encoder + MLP classifier over the full parent-child sequence."""

    def __init__(self, input_dim, hidden_size, num_layers, dropout_p,
                 output_dim=2, nhead=5, transformer_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=hidden_size,
            dropout=0.1, batch_first=False, norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers, enable_nested_tensor=False
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

    def forward(self, x, hidden_state=None):
        encoded = self.transformer_encoder(x)
        h = encoded[-1, :, :]
        return self.classifier(h), h
