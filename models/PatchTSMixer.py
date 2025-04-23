import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi

from layers.SelfAttention_Family import AttentionLayer, ProbAttention, FullAttention
import torch.nn.init as init
from models.Channel_conv import BiRNN, BiRNN2
# from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction




# from models.Revin import RevIN

# from mambapy.mamba import Mamba, MambaConfig

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(*normalized_shape))  # 可学习的缩放参数

    def forward(self, x):
        # 计算均方根，指定维度为 -2 和 -1
        rms = torch.sqrt(torch.mean(x**2, dim=(-2, -1), keepdim=True) + self.eps)
        # 归一化并缩放
        return self.scale * x / rms



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in


        # config = PatchTSMixerConfig(context_length = self.seq_len, prediction_length = self.pred_len, \
        #         d_model=configs.d_model, num_input_channels=self.enc_in, mode='mix_channel')
        # self.patchtsmixer = PatchTSMixerForPrediction(config)

        # self.rnn_ablation = configs.rnn_ablation
        # if self.rnn_ablation:
        #     self.ablation = BiRNN2(enc_in=configs.enc_in, d_model=configs.seq_len, n_patch=configs.n_patch, dropout=configs.dropout)


        self.use_norm = configs.use_norm

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch, seq, channel = x_enc.shape

        if self.use_norm:
            means2 = x_enc.mean(1, keepdim=True).detach()       
            x_enc = x_enc - means2
            stdev2 = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev2     
    
        # if self.rnn_ablation:
        #     x_enc = self.ablation(x_enc)

        # x_enc = self.patchtsmixer(x_enc)
        # enc_out = x_enc[0]

        if self.use_norm:
            enc_out = enc_out * stdev2 + means2 
        return enc_out 
    
    
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        if  self.norm_method == 'revin':
            means2 = x_enc.mean(1, keepdim=True).detach()       
            x_enc = x_enc - means2
            stdev2 = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev2    
        if self.norm_method == 'last_value':
            min1 = x_enc[:,-1:,:] 
            # min1 = x_enc.mean(1, keepdim=True).detach() 
            x_enc = x_enc - min1

        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        dec_out = self.projection((x_enc ).transpose(1, 2)).transpose(1, 2)

        if  self.norm_method == 'revin':
            dec_out = dec_out *stdev2 + means2
        if self.norm_method == 'last_value':
            dec_out = dec_out + min1 
        # enc_out = self.revin(enc_out , 'denorm')
        return dec_out
    
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out  = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]   # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
