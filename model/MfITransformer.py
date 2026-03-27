import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from typing import Dict, Optional, Tuple


class MfITransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    Extended for multi-frequency datasets.
    """

    def __init__(self, configs):
        super(MfITransformer, self).__init__()
        self.seq_len_list = getattr(configs, 'seq_len_list', [configs.seq_len])
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        self.mf_enable = getattr(configs, 'mf_enable', False)
        self.mf_freqs = getattr(configs, 'mf_freqs_list', [])
        self.mf_seq_lens = getattr(configs, 'mf_seq_lens_map', {})
        self.mf_pred_lens = getattr(configs, 'mf_pred_lens_map', {})

        self.enc_embeddings = nn.ModuleList([
            DataEmbedding_inverted(s_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
            for s_len in self.seq_len_list
        ])

        if self.mf_enable and self.mf_freqs:
            self.enc_embeddings_mf = nn.ModuleDict({
                f_key: DataEmbedding_inverted(
                    self.mf_seq_lens[f_key],
                    configs.d_model,
                    configs.embed,
                    configs.freq,
                    configs.dropout,
                )
                for f_key in self.mf_freqs
            })
            self.dec_embeddings_mf = nn.ModuleDict({
                f_key: DataEmbedding_inverted(
                    configs.label_len + self.mf_pred_lens.get(f_key, configs.pred_len),
                    configs.d_model,
                    configs.embed,
                    configs.freq,
                    configs.dropout,
                )
                for f_key in self.mf_freqs
            })
            self.projection_heads = nn.ModuleDict({
                f_key: nn.Linear(configs.d_model, self.mf_pred_lens.get(f_key, configs.pred_len), bias=True)
                for f_key in self.mf_freqs
            })
        else:
            self.enc_embeddings_mf = None
            self.dec_embeddings_mf = None
            self.projection_heads = None
        
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=None,
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast_mf(
        self,
        x_enc: Dict[str, torch.Tensor],
        x_mark_enc: Optional[Dict[str, torch.Tensor]],
        x_dec: Dict[str, torch.Tensor],
        x_mark_dec: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, list]]:
        """Forecast per frequency with shared encoder/decoder and per-frequency heads.

        `x_enc`, `x_dec`, and marks are dicts keyed by frequency.
        Each frequency runs through the same encoder/decoder modules sequentially,
        producing separate encoder states and decoder outputs per key.
        """
        dec_out = {}
        attn_dict = {}
        means_map = {}
        stdev_map = {}

        for f_key in self.mf_freqs:
            if f_key not in x_enc:
                raise ValueError('Missing frequency key in x_enc: {}'.format(f_key))
            if f_key not in x_dec:
                raise ValueError('Missing frequency key in x_dec: {}'.format(f_key))

            x_enc_f = x_enc[f_key]
            x_dec_f = x_dec[f_key]

            if self.use_norm:
                means = x_enc_f.mean(1, keepdim=True).detach()
                x_enc_f = x_enc_f - means
                stdev = torch.sqrt(torch.var(x_enc_f, dim=1, keepdim=True, unbiased=False) + 1e-5)
                x_enc_f = x_enc_f / stdev
                means_map[f_key] = means
                stdev_map[f_key] = stdev

            x_mark_enc_f = None if x_mark_enc is None else x_mark_enc.get(f_key)
            x_mark_dec_f = None if x_mark_dec is None else x_mark_dec.get(f_key)

            # Shared encoder, run per frequency to keep separate encoder states.
            enc_out_f = self.enc_embeddings_mf[f_key](x_enc_f, x_mark_enc_f)
            enc_out_f, attns_f = self.encoder(enc_out_f, attn_mask=None)

            # Shared decoder fed with per-frequency decoder inputs.
            n_targets = x_dec_f.shape[-1]
            dec_out_f = self.dec_embeddings_mf[f_key](x_dec_f, x_mark_dec_f)
            dec_out_f = self.decoder(dec_out_f, enc_out_f, x_mask=None, cross_mask=None)

            # Keep only original target-variable tokens if mark tokens were appended.
            dec_tokens_f = dec_out_f[:, :n_targets, :]
            pred_f = self.projection_heads[f_key](dec_tokens_f).permute(0, 2, 1)

            if self.use_norm:
                pred_len_f = pred_f.shape[1]
                pred_f = pred_f * stdev_map[f_key][:, 0, :].unsqueeze(1).repeat(1, pred_len_f, 1)
                pred_f = pred_f + means_map[f_key][:, 0, :].unsqueeze(1).repeat(1, pred_len_f, 1)

            dec_out[f_key] = pred_f
            attn_dict[f_key] = attns_f

        return dec_out, attn_dict

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, dataset_idx=0):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embeddings[dataset_idx](x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, dataset_idx=0):
        if isinstance(x_enc, dict):
            if self.enc_embeddings_mf is None or self.dec_embeddings_mf is None or self.projection_heads is None:
                raise ValueError('Received multi-frequency dict input but MF path is not initialized.')
            if not isinstance(x_dec, dict):
                raise ValueError('MF path expects dict decoder input x_dec keyed by frequency.')
            dec_out, attns = self.forecast_mf(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.output_attention:
                return dec_out, attns
            return dec_out

        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, dataset_idx=dataset_idx)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Model(MfITransformer):
    pass