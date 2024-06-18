import torch
import numpy as np
from model.iTransformer import Model
from gpytorch.likelihoods import GaussianLikelihood

class Config:
    def __init__(self):
        self.seq_len = 96
        self.pred_len = 24
        self.output_attention = False
        self.use_norm = True
        self.d_model = 512
        self.embed = 'fixed'
        self.freq = 'h'
        self.dropout = 0.1
        self.class_strategy = 'sum'
        self.n_heads = 8
        self.factor = 5
        self.activation = 'relu'
        self.d_ff = 2048
        self.e_layers = 2

def main():
    configs = Config()

    model = Model(configs)

    # Example data
    batch_size = 32
    seq_len = configs.seq_len
    pred_len = configs.pred_len
    num_features = configs.d_model

    x_enc = torch.randn(batch_size, seq_len, num_features)
    x_mark_enc = torch.randn(batch_size, seq_len, num_features)
    x_dec = torch.randn(batch_size, pred_len, num_features)
    x_mark_dec = torch.randn(batch_size, pred_len, num_features)

    # Forward pass
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()
