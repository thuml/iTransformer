import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.enc_in)
        ])
        
        self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.enc_in)
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x: [B, L, D]
        x = self.rev(x_enc, 'norm') if self.rev else x_enc
        x = self.dropout(x)
        pred = torch.zeros(x.size(0), self.pred_len, x.size(2)).to('cuda')
        for idx, proj in enumerate(self.Linear):
            pred[:, :, idx] = proj(x[:, :, idx])
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]