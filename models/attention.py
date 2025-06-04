import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    """LSTM model with attention mechanism."""
    def __init__(self, seq_length: int, input_dim: int, output_dim: int, hidden_size: int, dropout_p: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.encoder = nn.LSTM(input_dim, hidden_size)
        self.attn = nn.Linear(hidden_size, seq_length)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, input_seq: torch.Tensor, hidden_state: torch.Tensor) -> tuple:
        input_seq = self.dropout(input_seq)
        encoder_outputs, (h, _) = self.encoder(input_seq, hidden_state)
        attn_applied, attn_weights = self.attention(encoder_outputs, h)
        scores = self.out(attn_applied.reshape(-1, self.hidden_size))
        return scores, attn_weights

    def attention(self, encoder_outputs: torch.Tensor, hidden: torch.Tensor) -> tuple:
        attn_weights = F.softmax(torch.squeeze(self.attn(hidden)), dim=1)
        attn_weights = torch.unsqueeze(attn_weights, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        return attn_applied, torch.squeeze(attn_weights)

    def init_hidden(self, batch_size: int) -> tuple:
        h_init = torch.zeros(1, batch_size, self.hidden_size)
        c_init = torch.zeros(1, batch_size, self.hidden_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h_init = h_init.to(device)
        c_init = c_init.to(device)
        return (h_init, c_init)

class DualAttentionRNNModel(nn.Module):
    """Dual attention RNN model with input and temporal attention."""
    def __init__(self, seq_length: int, input_dim: int, output_dim: int, hidden_size: int, dropout_p: float):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_p)
        self.encoder = nn.LSTM(input_dim, hidden_size)
        self.input_attn_weights = nn.Linear(2 * hidden_size, seq_length)
        self.input_attn_transform = nn.Linear(seq_length, seq_length)
        self.input_attn_vector = nn.Linear(seq_length, 1)
        self.temporal_attn_transform = nn.Linear(hidden_size, hidden_size)
        self.temporal_attn_vector = nn.Linear(hidden_size, 1)
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> tuple:
        x = self.dropout(x)
        h_seq = []
        for t in range(self.seq_length):
            x_tilde, _ = self.input_attention(x, hidden_state, t)
            ht, hidden_state = self.encoder(x_tilde, hidden_state)
            h_seq.append(ht)
        h = torch.cat(h_seq, dim=0)
        context, beta = self.temporal_attention(h)
        logits = self.out(context)
        return logits, torch.squeeze(beta)

    def input_attention(self, x: torch.Tensor, hidden_state: tuple, t: int) -> tuple:
        x = x.permute(1, 2, 0)
        h, c = hidden_state
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)
        hc = torch.cat([h, c], dim=2)
        e = self.input_attn_vector(torch.tanh(self.input_attn_weights(hc) + self.input_attn_transform(x)))
        e = torch.squeeze(e)
        alpha = F.softmax(e, dim=1)
        xt = x[:, :, t]
        x_tilde = alpha * xt
        x_tilde = torch.unsqueeze(x_tilde, 0)
        return x_tilde, alpha

    def temporal_attention(self, h: torch.Tensor) -> tuple:
        h = h.permute(1, 0, 2)
        l = self.temporal_attn_vector(torch.tanh(self.temporal_attn_transform(h)))
        l = torch.squeeze(l)
        beta = F.softmax(l, dim=1)
        beta = torch.unsqueeze(beta, 1)
        context = torch.bmm(beta, h)
        context = torch.squeeze(context)
        return context, beta

    def init_hidden(self, batch_size: int) -> tuple:
        h_init = torch.zeros(1, batch_size, self.hidden_size)
        c_init = torch.zeros(1, batch_size, self.hidden_size)
        return (h_init, c_init)