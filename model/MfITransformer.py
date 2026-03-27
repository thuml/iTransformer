import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
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
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.mf_enable = getattr(configs, 'mf_enable', False)
        self.mf_freqs = getattr(configs, 'mf_freqs_list', [])
        self.mf_seq_lens = getattr(configs, 'mf_seq_lens_map', {})
        self.mf_pred_lens = getattr(configs, 'mf_pred_lens_map', {})

        if not self.mf_enable or not self.mf_freqs:
            raise ValueError('MfITransformer requires mf_enable=True and non-empty mf_freqs_list.')

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
        
        self.class_strategy = configs.class_strategy
        # Shared encoder across all frequency keys.
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

        # Optional shared feed-forward trunk after the shared encoder.
        mf_trunk_hidden = getattr(configs, 'mf_trunk_hidden', 0)
        if isinstance(mf_trunk_hidden, int) and mf_trunk_hidden > 0:
            self.shared_trunk = nn.Sequential(
                nn.Linear(configs.d_model, mf_trunk_hidden),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(mf_trunk_hidden, configs.d_model),
            )
        else:
            self.shared_trunk = nn.Identity()

        self.projection_heads = nn.ModuleDict({
            f_key: nn.Linear(configs.d_model, self.mf_pred_lens.get(f_key, configs.pred_len), bias=True)
            for f_key in self.mf_freqs
        })

    def forecast_mf(
        self,
        x_enc: Dict[str, torch.Tensor],
        x_mark_enc: Optional[Dict[str, torch.Tensor]],
        x_dec: Dict[str, torch.Tensor],
        x_mark_dec: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, list]]:
        """Forecast per frequency with shared encoder/trunk and per-frequency heads."""
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

            # Shared encoder and optional shared FF trunk.
            enc_out_f = self.enc_embeddings_mf[f_key](x_enc_f, x_mark_enc_f)
            enc_out_f, attns_f = self.encoder(enc_out_f, attn_mask=None)
            shared_tokens_f = self.shared_trunk(enc_out_f)

            # Use target-token count from decoder-input dict to support
            # frequency-specific target groups.
            n_targets = x_dec_f.shape[-1]
            if n_targets > shared_tokens_f.shape[1]:
                raise ValueError(
                    'Target token count ({}) exceeds encoder token count ({}) for frequency {}.'.format(
                        n_targets, shared_tokens_f.shape[1], f_key
                    )
                )

            target_tokens_f = shared_tokens_f[:, :n_targets, :]
            pred_f = self.projection_heads[f_key](target_tokens_f).permute(0, 2, 1)

            if self.use_norm:
                pred_len_f = pred_f.shape[1]
                pred_f = pred_f * stdev_map[f_key][:, 0, :].unsqueeze(1).repeat(1, pred_len_f, 1)
                pred_f = pred_f + means_map[f_key][:, 0, :].unsqueeze(1).repeat(1, pred_len_f, 1)

            dec_out[f_key] = pred_f
            attn_dict[f_key] = attns_f

        return dec_out, attn_dict

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, dataset_idx=0):
        if not isinstance(x_enc, dict):
            raise ValueError('MfITransformer only supports dict input x_enc keyed by frequency.')
        if not isinstance(x_dec, dict):
            raise ValueError('MfITransformer only supports dict input x_dec keyed by frequency.')
        if x_mark_enc is not None and not isinstance(x_mark_enc, dict):
            raise ValueError('MfITransformer expects dict x_mark_enc when provided.')
        if x_mark_dec is not None and not isinstance(x_mark_dec, dict):
            raise ValueError('MfITransformer expects dict x_mark_dec when provided.')

        dec_out, attns = self.forecast_mf(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out, attns
        return dec_out


class Model(MfITransformer):
    """Compatibility alias used by experiment loaders."""

    pass

