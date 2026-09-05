import torch
from torch import nn, utils


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: (B, C, T)
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class AttentionPool1D(nn.Module):
    """
    Attention pooling over time for sequences (B, C, T).
    Returns:
      context: (B, C)
      attn_w : (B, T)  (weights sum to 1 over T)
    """
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)  # (B,1,T)

    def forward(self, z):
        # z: (B, C, T)
        w = self.score(z).squeeze(1)        # (B, T)
        a = torch.softmax(w, dim=-1)        # (B, T)
        context = torch.sum(z * a.unsqueeze(1), dim=-1)  # (B, C)
        return context, a


class TCNAttenHeteroRegressor(nn.Module):
    def __init__(self, n_features, channels=(64, 64, 64), kernel_size=3, dropout=0.1,
                 min_logvar=-10.0, max_logvar=5.0, return_attention=False):
        super().__init__()
        self.return_attention = return_attention

        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation, padding=padding, dropout=dropout)
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        self.attn_pool = AttentionPool1D(in_ch)


        self.mu_head = nn.Linear(in_ch, 1)
        self.logvar_head = nn.Linear(in_ch, 1)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

    def forward(self, x, return_attn: bool = False):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        z = self.tcn(x)  # (B, C, T)

        context, attn_w = self.attn_pool(z)  # (B, C), (B, T)

        mu = self.mu_head(context)              # (B, 1)
        logvar = self.logvar_head(context)      # (B, 1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)

        if return_attn:
            return mu, logvar, attn_w
        return mu, logvar