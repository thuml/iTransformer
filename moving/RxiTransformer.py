import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.Invertible import RevIN


class Model(nn.Module):
    # Parrarel model of iTransformer and RLinear-CI

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # iTransoformer
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # RLinear-CI
        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.enc_in)
        ])
        self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.enc_in) if configs.rev else None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Saving x_enc
        x_enc_save = x_enc
        
        #iTransformer
        if self.use_norm:
            # Normalization from RevIN
            x_normed = self.rev(x_enc_save, 'norm') if self.rev else x_enc_save
        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model;    # L: seq_len;    S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        # B L N -> B S N
        enc_out = self.enc_embedding(x_normed, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            enc_out = self.rev(enc_out, 'denorm') if self.rev else enc_out
        
        # RLinear-CI
        x = self.dropout(x_normed)
        pred = torch.zeros(x.size(0), self.pred_len, x.size(2)).to('cuda')
        for idx, proj in enumerate(self.Linear):
           pred[:, :, idx] = proj(x[:, :, idx])
        rl_out = self.rev(pred, 'denorm') if self.rev else pred 
        
        # Add
        out = enc_out*0.5 + rl_out*0.5


        return out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]