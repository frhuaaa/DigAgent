import torch
import torch.nn as nn

from backbone.modules import PositionalEncoding

class LSTMModel(nn.Module):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        use_pe=False,
        use_bn=False,
        use_ln=False,
    ):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.use_batch_norm = bool(use_bn)

        self.bn_in = (
            nn.BatchNorm1d(num_features=d_feat)
            if use_bn
            else nn.Identity()
        )

        self.pos_encoder = (
            PositionalEncoding(d_model=d_feat, max_len=5000)
            if use_pe
            else nn.Identity()
        )

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.ln_out = (
            nn.LayerNorm(hidden_size)
            if use_ln
            else nn.Identity()
        )

        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_batch_norm:
            x = x.permute(0, 2, 1)
            x = self.bn_in(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.bn_in(x)
        x = self.pos_encoder(x)
        out, _ = self.rnn(x)
        h_last = self.ln_out(out[:, -1, :])
        return self.fc_out(h_last).squeeze(-1)
