import math
import torch
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from layers.AMS import AMS
from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN
from functools import reduce
from operator import mul
from models.Channel_conv import BiRNN2, BiRNN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer_nums = configs.layer_nums
        self.num_nodes = configs.enc_in
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.use_norm
        self.batch_norm = configs.batch_norm
        if self.revin:
            self.revin_layer = RevIN(
                num_features=configs.enc_in, affine=False, subtract_last=False
            )

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)

        # self.start_fc = nn.Linear(in_features=self.seq_len, out_features=self.d_model*self.seq_len)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device("cuda:{}".format(configs.gpu))

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(
                    self.seq_len,
                    self.seq_len,
                    self.num_experts_list[num],
                    self.device,
                    k=self.k,
                    num_nodes=self.num_nodes,
                    patch_size=self.patch_size_list[num],
                    noisy_gating=True,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    layer_number=num + 1,
                    residual_connection=self.residual_connection,
                    batch_norm=self.batch_norm,
                )
            )
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pred_len)
        )

        self.enc_in = configs.enc_in
        self.rnn_ablation = configs.rnn_ablation
        if self.rnn_ablation:
            self.ablation = BiRNN2(enc_in=configs.enc_in, d_model=self.seq_len, n_patch=configs.n_patch, dropout=configs.dropout)


    def forecast(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        bs, l, m = x.shape
        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, "norm")
    
        if self.rnn_ablation:
            x = self.ablation(x)

        # out = self.start_fc(x.permute(0,2,1)).reshape(bs, m, l, self.d_model).permute(0,2,1,3)
        out = self.start_fc(x.unsqueeze(-1))

        # batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        out = out.permute(0, 2, 1, 3).reshape(bs, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        # denorm
        if self.revin:
            out = self.revin_layer(out, "denorm")

        return out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out  = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]   # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')


