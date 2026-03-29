import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from typing import Dict, Optional, Tuple


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    Extended for multi-frequency datasets.

        Comparison notes against iTransformer:
        - Sections are marked with === SAME === or === MODIFIED ===.
        - SAME means the modeling step is conceptually the same,
            even if MF uses dicts/loops/ModuleDict naming.
    - "Step" states what this block does.
    - "Why" states why it differs (or matches) relative to iTransformer.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # === SAME (vs iTransformer) ===
        # Step: initialize shared forecasting flags.
        # Why: these controls are identical core behavior knobs.
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # === MODIFIED (vs iTransformer) ===
        # Step: load multi-frequency metadata from parsed CLI/config maps.
        # Why: MfITransformer needs per-frequency sequence/prediction settings.
        self.mf_freqs = getattr(configs, 'mf_freqs_list', [])
        self.mf_seq_lens = getattr(configs, 'mf_seq_lens_map', {})
        self.mf_pred_lens = getattr(configs, 'mf_pred_lens_map', {})

        if not self.mf_freqs:
            raise ValueError('MfITransformer requires non-empty mf_freqs_list.')

        # === SAME (concept vs iTransformer) ===
        # Step: map time-series input into token embeddings.
        # Why: same embedding stage, implemented per frequency for MF inputs.
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

        # === MODIFIED (vs iTransformer) ===
        # Step: create learnable frequency identity vectors.
        # Why: shared encoder benefits from explicit frequency context.
        self.freq_embeddings = nn.ParameterDict({
            f_key: nn.Parameter(torch.zeros(configs.d_model))
            for f_key in self.mf_freqs
        })
        
        # === SAME (vs iTransformer) ===
        # Step: keep class_strategy assignment for config compatibility.
        # Why: retained from base family even if unused in current forecast path.
        self.class_strategy = configs.class_strategy

        # === SAME (vs iTransformer) ===
        # Step: keep the same encoder architecture.
        # Why: attention/FFN stack is reused; only data flow differs.
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

        # === SAME (concept vs iTransformer) ===
        # Step: project encoded tokens to prediction horizon.
        # Why: same projection idea, instantiated per frequency in MF mode.
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
        # === MODIFIED (vs iTransformer) ===
        # Step: run multi-frequency forecasting in one fused latent pass.
        # Why: enables cross-frequency information sharing before projection.
        """Forecast via latent fusion over all frequencies in a single encoder pass."""
        dec_out = {}
        attn_dict = {}
        means_map = {}
        stdev_map = {}
        token_counts = {}
        token_slices = {}
        fused_tokens = []

        # === SAME (concept vs iTransformer) ===
        # Step: normalize input then embed before encoder.
        # Why: same normalization+embedding pipeline, executed per frequency.
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

            # Project each frequency to token space and inject frequency identity.
            tokens_f = self.enc_embeddings_mf[f_key](x_enc_f, x_mark_enc_f)
            n_tokens_f = tokens_f.shape[1]
            token_counts[f_key] = n_tokens_f
            fused_tokens.append(tokens_f + self.freq_embeddings[f_key].view(1, 1, -1))

        if not fused_tokens:
            return dec_out, attn_dict

        # === MODIFIED (vs iTransformer) ===
        # Step: concatenate tokens from all frequencies before encoding.
        # Why: this latent fusion step is MF-specific and not in iTransformer.
        combined_tokens = torch.cat(fused_tokens, dim=1)

        # === SAME (concept vs iTransformer) ===
        # Step: run token sequence through the shared encoder.
        # Why: same encoder operation as iTransformer once tokens are formed.
        combined_tokens, attns = self.encoder(combined_tokens, attn_mask=None)

        # === MODIFIED (vs iTransformer) ===
        # Step: split encoded tokens back into frequency-specific slices.
        # Why: each frequency uses its own projection head afterward.
        offset = 0
        for f_key in self.mf_freqs:
            n_tokens_f = token_counts[f_key]
            token_slices[f_key] = combined_tokens[:, offset:offset + n_tokens_f, :]
            offset += n_tokens_f

        # === SAME (concept vs iTransformer) ===
        # Step: project model outputs and denormalize back to original scale.
        # Why: same projection+denormalization step, repeated per frequency.
        for f_key in self.mf_freqs:
            x_dec_f = x_dec[f_key]
            shared_tokens_f = token_slices[f_key]

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
                stdev_f = stdev_map[f_key][:, 0, :n_targets].unsqueeze(1).repeat(1, pred_len_f, 1)
                means_f = means_map[f_key][:, 0, :n_targets].unsqueeze(1).repeat(1, pred_len_f, 1)
                pred_f = pred_f * stdev_f
                pred_f = pred_f + means_f

            dec_out[f_key] = pred_f
            attn_dict[f_key] = attns

        return dec_out, attn_dict

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, dataset_idx=0):
        # === MODIFIED (vs iTransformer) ===
        # Step: enforce dict inputs keyed by frequency before forward pass.
        # Why: MfITransformer contract is multi-frequency dictionaries, not a single tensor.
        if not isinstance(x_enc, dict):
            raise ValueError('MfITransformer only supports dict input x_enc keyed by frequency.')
        if not isinstance(x_dec, dict):
            raise ValueError('MfITransformer only supports dict input x_dec keyed by frequency.')
        if x_mark_enc is not None and not isinstance(x_mark_enc, dict):
            raise ValueError('MfITransformer expects dict x_mark_enc when provided.')
        if x_mark_dec is not None and not isinstance(x_mark_dec, dict):
            raise ValueError('MfITransformer expects dict x_mark_dec when provided.')

        # === SAME (concept vs iTransformer) ===
        # Step: return predictions and optional attention outputs.
        # Why: same output_attention branch pattern with MF-shaped return values.
        dec_out, attns = self.forecast_mf(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out, attns
        return dec_out