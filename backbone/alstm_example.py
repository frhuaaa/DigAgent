
import torch
import torch.nn as nn
from backbone.modules import PositionalEncoding

class ALSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU", attention_hidden_size=32, use_pe=False, use_bn=False):
        super().__init__()
        if attention_hidden_size <= 0:
            raise ValueError("attention_hidden_size must be positive")
        self.hid_size = hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self.use_pe = use_pe
        self.use_bn = use_bn
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.bn_in = (
            nn.BatchNorm1d(num_features=self.input_size)
            if self.use_bn
            else nn.Identity()
        )

        self.pos_encoder = (
            PositionalEncoding(d_model=self.input_size, max_len=5000)
            if self.use_pe
            else nn.Identity()
        )
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(
                in_features=self.hid_size,
                out_features=self.attention_hidden_size,
            ),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(
                in_features=self.attention_hidden_size,
                out_features=1,
                bias=False,
            ),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_bn:
            inputs = inputs.permute(0, 2, 1)
            inputs = self.bn_in(inputs)
            inputs = inputs.permute(0, 2, 1)
        else:
            inputs = self.bn_in(inputs)

        inputs = self.pos_encoder(inputs)
        rnn_out, _ = self.rnn(self.net(inputs))
        attention_score = self.att_net(rnn_out)
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )
        return out[..., 0]
