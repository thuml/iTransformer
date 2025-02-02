import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted,DataEmbedding_oinverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len#将输入序列的长度（seq_len）从配置对象 configs 中提取并赋值给实例变量 self.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding处理将数据
        self.enc_embedding2 = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.enc_embedding1 = DataEmbedding_oinverted(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,#d_model 是模型的嵌入维度（或隐藏层维度）。它指的是输入和输出特征向量的维度
                    configs.d_ff,#d_ff 是前馈神经网络（Feed-Forward Network, FFN）的维度。每个注意力层后通常会有一个前馈层
                    dropout=configs.dropout,#dropout 是一种正则化技术，用于防止过拟合
                    activation=configs.activation#activation 指定了使用的激活函数类型
                ) for l in range(configs.e_layers)#循环代码，有几个encoder层执行几次
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)#创建一个层归一化（Layer Normalization）层的实例
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)#定义了一个线性层，用于将输入的特征向量转换为预测的输出

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):#这是一个方法的定义可以接受编码器和解码器的四个特征
        if self.use_norm:#判断是否使用归一化
            # Normalization from Non-stationary Transformer归一化处理
            means = x_enc.mean(1, keepdim=True).detach()#计算均值
            x_enc = x_enc - means#输入值减均值，使得归一化后的数据均值为零
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)#计算x_enc 的标准差
            x_enc /= stdev# x_enc 除以计算得到的标准差，从而将数据缩放，使得归一化后的数据标准差为1

        _, _, N = x_enc.shape # B L N这里使用 _ 变量来忽略前两个维度，提取出 x_enc 的第三个维度 N（1，96，4） （1，96，5）
        # B: batch_size;    E: d_model; （）
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates变量的数量
        #自己加的
        x_in = self.enc_embedding1(x_enc, None)
        x_enc, attns = self.encoder(x_in, attn_mask=None)
        # Embedding
        # B L N -> B N E 用了维度，数量和批量来作为输入      (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding2(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer反归一化处理
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]