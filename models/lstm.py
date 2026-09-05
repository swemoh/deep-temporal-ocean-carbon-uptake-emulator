import torch
from torch import nn, utils


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        # h: (B, T, H)
        w = self.score(h).squeeze(-1)          # (B, T)
        a = torch.softmax(w, dim=1)            # (B, T)
        context = torch.sum(h * a.unsqueeze(-1), dim=1)  # (B, H)
        return context, a


class LSTMAttnHeteroRegressor(nn.Module):
    """
    Heteroscedastic LSTM + attention:
      - outputs mean mu(x) and log-variance logvar(x)
      - optionally returns attention weights for analysis
    """
    def __init__(
        self,
        n_features,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
        min_logvar=-10.0,
        max_logvar=5.0,
        return_attention=False,
    ):
        super().__init__()
        self.return_attention = return_attention

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.attn = AttentionPool(out_dim)

        # Two heads instead of one: mean and log-variance
        self.mu_head = nn.Linear(out_dim, 1)
        self.logvar_head = nn.Linear(out_dim, 1)

        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

    def forward(self, x, return_attn: bool = False):
        # x: (B, T, F)
        h, _ = self.lstm(x)                 # (B, T, out_dim)
        context, attn_w = self.attn(h)      # (B, out_dim), (B, T)

        mu = self.mu_head(context)          # (B, 1)
        logvar = self.logvar_head(context)  # (B, 1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)

        if return_attn:
            return mu, logvar, attn_w

        if self.return_attention:
            return mu, logvar, attn_w
        return mu, logvar