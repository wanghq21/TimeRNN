
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import DataEmbedding, PositionalEmbedding
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer

from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.SelfAttention_Family import AttentionLayer, ProbAttention, FullAttention
import torch.nn.init as init


class MoE(nn.Module):
    def __init__(self, hidden=96, nexpert=1, rate=0.1):
        super(MoE, self).__init__()
        self.nexpert = nexpert
        self.linear = torch.nn.Linear(hidden, nexpert*4)
        self.linear2 = torch.nn.Linear(nexpert*4, nexpert)
        self.dropout = torch.nn.Dropout(rate)
        self.relu = torch.nn.LeakyReLU(rate)


    def forward(self, x):
        x = self.dropout(self.linear2(self.dropout(self.relu(self.linear(x)))))
        x = torch.nn.functional.softmax(x/(math.sqrt(self.nexpert)), dim=-1)

        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        self.d_model = configs.d_model

        # self.dropout = torch.nn.Dropout(0.2)
        self.rate = configs.dropout
        self.dropout = torch.nn.Dropout(self.rate)
        # self.relu = torch.nn.LeakyReLU(self.rate)
        self.gelu = torch.nn.LeakyReLU(self.rate)
        # self.gelu = torch.nn.GELU()
        # self.gelu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        self.channel_patch = int(math.sqrt(self.enc_in))+1
        self.channel_patch2 = self.channel_patch

        self.patch = configs.patch
        self.n_patch = [self.seq_len//self.patch[i] if self.seq_len%self.patch[i]==0 else self.seq_len//self.patch[i]+1 for i in range(len(self.patch))]

        self.layers = len(self.patch)

        self.decomp = torch.nn.ModuleList([series_decomp(self.patch[i] + 1) for i in range(self.layers)])
        
        self.pred_linear = torch.nn.ModuleList([torch.nn.Linear(self.n_patch[i]*self.patch[i],  self.pred_len)for i in range(self.layers)])
        # self.pred_linear = torch.nn.Linear(self.seq_len, self.pred_len )

        self.moe = MoE(self.seq_len, len(self.patch), self.rate)
        self.w = torch.ones(len(self.patch), requires_grad=True)


        self.d_model = 64
        self.d_ff = 256

        # trend and season
        self.emb_t = torch.nn.ModuleList([torch.nn.Linear(self.patch[i], self.d_model)
                            for i in range(self.layers)])

        self.emb_s = torch.nn.ModuleList([torch.nn.Linear(self.patch[i], self.d_model)
                            for i in range(self.layers)])
        self.rnn_t_local = torch.nn.ModuleList([torch.nn.LSTM(self.n_patch[i], self.n_patch[i],batch_first=True) for i in range(self.layers)])
        self.rnn_t_global = torch.nn.ModuleList([torch.nn.LSTM(self.patch[i], self.patch[i],batch_first=True) for i in range(self.layers)])
        self.rnn_s_local = torch.nn.ModuleList([torch.nn.LSTM(self.n_patch[i], self.n_patch[i],batch_first=True) for i in range(self.layers)])
        self.rnn_s_global = torch.nn.ModuleList([torch.nn.LSTM(self.patch[i], self.patch[i],batch_first=True) for i in range(self.layers)])
 
        self.pos_t = torch.nn.ModuleList([PositionalEmbedding(self.d_model) for i in range(self.layers)])
        self.pos_s = torch.nn.ModuleList([PositionalEmbedding(self.d_model) for i in range(self.layers)])
        self.prediction_t_local = torch.nn.ModuleList([torch.nn.Linear(self.n_patch[i], self.n_patch[i])
                            for i in range(self.layers)])
        self.prediction_t_global = torch.nn.ModuleList([torch.nn.Linear(self.patch[i], self.patch[i])
                            for i in range(self.layers)])
        self.prediction_s_local = torch.nn.ModuleList([torch.nn.Linear(self.n_patch[i], self.n_patch[i])
                            for i in range(self.layers)])
        self.prediction_s_global = torch.nn.ModuleList([torch.nn.Linear(self.patch[i], self.patch[i])
                            for i in range(self.layers)])
        self.t_bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])
        self.s_bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])
        self.t_bn2 = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])
        self.s_bn2 = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])
  
        self.channel1 = torch.nn.ModuleList([torch.nn.Linear(self.channel_patch2, self.channel_patch2*4)
                            for i in range(self.layers)])
        self.channel2 = torch.nn.ModuleList([torch.nn.Linear(self.channel_patch2*4, self.channel_patch2)
                            for i in range(self.layers)])
        self.c_bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.n_patch[i]*self.patch[i])
                            for i in range(self.layers)])
        self.c_bn2 = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.n_patch[i]*self.patch[i])
                            for i in range(self.layers)])
        self.channel3 = torch.nn.ModuleList([torch.nn.Linear(self.channel_patch, self.channel_patch*4)
                            for i in range(self.layers)])
        self.channel4 = torch.nn.ModuleList([torch.nn.Linear(self.channel_patch*4, self.channel_patch)
                            for i in range(self.layers)])




    def encoder(self, x_enc, x_mark_enc):

        batch, seq, channel = x_enc.shape

        means2 = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means2
        stdev2 = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev2

        weight = self.moe(x_enc.permute(0,2,1)) 
        result = []

        trend = x_enc
        seaoson_pred = torch.zeros([batch, self.pred_len, channel], device=x_enc.device)
        trend_pred = torch.zeros([batch, self.pred_len, channel], device=x_enc.device)
        for i in range(self.layers):
            trend1 = x_enc
            new_seqlen = self.n_patch[i]*self.patch[i]
            trend1 = torch.cat((trend1, trend1[:,:new_seqlen-self.seq_len,:]), dim=1)
     
            if self.patch[i] != 1:
                season, trend1 = self.decomp[i](trend1)
            else:
                season = torch.zeros([batch, new_seqlen, channel], device=x_enc.device)
                trend1 = trend1

            trend1 = trend1.reshape(batch, self.n_patch[i], self.patch[i], channel)
            trend1 = trend1.permute(0,3,2,1)
            t1 = trend1
            trend1 = self.t_bn[i](trend1)
            trend1 = trend1.reshape(batch*channel, self.patch[i], self.n_patch[i])
            # trend1 = self.emb_t[i](trend1) +self.pos_t[i](trend1)
            trend1, _ = self.rnn_t_local[i](trend1)
            trend1 = t1 +  self.prediction_t_local[i](trend1).reshape(batch, channel, self.patch[i], self.n_patch[i])
            trend1 = trend1.permute(0,1,3,2)
            t1 = trend1
            trend1 = self.t_bn2[i](trend1)
            trend1 = trend1.reshape(batch*channel, self.n_patch[i], self.patch[i])
            # trend1 = self.emb_t[i](trend1) +self.pos_t[i](trend1)
            trend1, _ = self.rnn_t_global[i](trend1)
            trend1 = t1 +  self.prediction_t_global[i](trend1).reshape(batch, channel, self.n_patch[i], self.patch[i])    
            trend1 = trend1.reshape(batch, channel, new_seqlen).permute(0,2,1)

            season = season.reshape(batch, self.n_patch[i], self.patch[i], channel)
            season = season.permute(0,3,2,1)
            s1 = season
            season = self.s_bn[i](season)
            season = season.reshape(batch*channel, self.patch[i], self.n_patch[i])
            # trend1 = self.emb_t[i](trend1) +self.pos_t[i](trend1)
            season, _ = self.rnn_s_local[i](season)
            season = s1 + self.prediction_s_local[i](season).reshape(batch, channel, self.patch[i], self.n_patch[i])
            season = season.permute(0,1,3,2)
            s1 = season
            season = self.s_bn2[i](season)
            season = season.reshape(batch*channel, self.n_patch[i], self.patch[i])
            # trend1 = self.emb_t[i](trend1) +self.pos_t[i](trend1)
            season, _ = self.rnn_s_global[i](season)
            season = s1 + self.prediction_s_global[i](season).reshape(batch, channel, self.n_patch[i], self.patch[i])    
            season = season.reshape(batch, channel, new_seqlen).permute(0,2,1)

            trend1 = trend1 + season


            trend1 = torch.cat((trend1, trend1[:,:,:self.channel_patch*self.channel_patch2-self.enc_in]),dim=-1)
            trend1 = trend1.reshape(batch, new_seqlen, self.channel_patch, self.channel_patch2)
            x1 = trend1
            trend1 = self.c_bn[i](trend1)
            trend1 = x1 + self.dropout(self.channel2[i](self.dropout(self.gelu(self.channel1[i](trend1)))))
            x1 = trend1
            trend1 = self.c_bn2[i](trend1)
            trend1 = x1 + self.dropout(self.channel4[i](self.dropout(self.gelu(self.channel3[i](trend1.permute(0,1,3,2))))).permute(0,1,3,2))
            trend1 = trend1.reshape(batch, new_seqlen, -1)[:,:,:self.enc_in]


            # trend_pred = trend_pred  + self.pred_linear[i]((trend1 ).permute(0,2,1)).permute(0,2,1)
            result.append(self.pred_linear[i]((trend1).permute(0,2,1)))


        # 对于输入长度比较短，可以使用MoE，但是如果输入长度比较长，使用MoE还会性能下降
        # 并且当patch数量较多时可以使用MoE，但是patch数量较少时就直接使用sum或者mean
        # 但是通过实验证明，使用sum比使用mean好像要好一点点儿，同时注意使用sum和mean好像和直接相加效果一样
        # 但是又通过实验进一步证明，对几乎所有数据集都可以使用大量的patch
        # 相应的，可以调大dropout每一次只更新较少的元素
        # 同时，随着使用的patch的数量的逐渐增加，dropout的大小也逐渐增加
        # 注意dropout的影响还挺大的，有时候很小的一个改变甚至是只改变了0.1效果差别就挺大的
        # dec_out = trend_pred 
        # dec_out = dec_out[:,:,:self.enc_in]
        dec_out = (torch.matmul(torch.stack(result, dim=-1),weight.unsqueeze(-1)) ).squeeze(-1).permute(0,2,1)
        # dec_out = torch.sum(torch.mul(torch.stack(result, dim=-1), self.w.to(x_enc.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)),dim=-1).permute(0,2,1)


        dec_out = dec_out[:,:,:self.enc_in] * stdev2 + means2

        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.encoder(x_enc, x_mark_enc)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.encoder(x_enc, x_mark_enc)
    def anomaly_detection(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.encoder(x_enc, x_mark_enc)   
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        return None

