# 加上在频域进行操作
# 在时域可以使用avgpol，在频域呢
# 同时可以参考MoE，实现自动选择


import torch
import torch.nn as nn
import math
from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi

import torch
import torch.nn.functional as F

def smooth_with_local_fft_pytorch(vector, window_size):
    batch_size, num_features, seq_len = vector.shape
    smoothed_vector = torch.zeros_like(vector)
    half_window = window_size // 2
    for t in range(seq_len):
        start = max(0, t - half_window)
        end = min(seq_len, t + half_window)
        local_data = vector[..., start:end]
        # 对最后一维（时间维度）进行傅里叶变换
        fft_local_data = torch.fft.fft(local_data, dim=-1)
        # 低通滤波（示例：保留前一半频率成分）
        cutoff = fft_local_data.shape[-1] // 2
        # cutoff = 0
        fft_local_data[..., cutoff:] = 0
        # 逆傅里叶变换得到平滑后的数据
        smoothed_local_data = torch.real(torch.fft.ifft(fft_local_data, dim=-1))
        smoothed_vector[..., t] = smoothed_local_data[..., t - start]
    return smoothed_vector

import torch
import torch.fft as fft

def smooth_time_series_local_fft_pytorch(vector, window_size):
    batch_size, num_features, seq_len = vector.shape
    smoothed_vector = torch.zeros_like(vector)
    num_segments = (seq_len - window_size) // window_size + 1
    for i in range(num_segments):
        start = i * window_size
        end = start + window_size
        local_data = vector[..., start:end]
        fft_local_data = fft.fft(local_data, dim=-1)
        cutoff = fft_local_data.shape[-1] // 2
        fft_local_data[..., cutoff:] = 0
        smoothed_local_data = torch.real(fft.ifft(fft_local_data, dim=-1))
        smoothed_vector[..., start:end] = smoothed_local_data
    return smoothed_vector


class MoE(nn.Module):
    def __init__(self, hidden=96, nexpert=1):
        super(MoE, self).__init__()
        self.linear = torch.nn.Linear(hidden, nexpert)

    def forward(self, x):
        x = torch.nn.functional.softmax(self.linear(x), dim=-1)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        # self.dropout = torch.nn.Dropout(0.1)
        self.d_model = configs.d_model

        self.rate = configs.dropout
        self.dropout = torch.nn.Dropout(self.rate)
        self.gelu = torch.nn.LeakyReLU(self.rate)
        # self.gelu = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        # self.d = torch.nn.Dropout(0.2)
        # self.g = torch.nn.LeakyReLU(0.2)
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        # self.projection = torch.nn.Linear(configs.d_model, configs.enc_in, bias=True)

        self.patch = configs.patch
        self.layers = len(self.patch)

        self.decomp = torch.nn.ModuleList([series_decomp(self.patch[i] + 1) for i in range(self.layers)])
        
        self.s_pred_local = torch.nn.ModuleList([torch.nn.Linear(self.patch[i], self.patch[i] * 4) for i in range(self.layers)])
        self.s_pred_local2 = torch.nn.ModuleList([torch.nn.Linear(self.patch[i] * 4, self.patch[i]) for i in range(self.layers)])
        # self.s_pred_local_sigmoid = torch.nn.ModuleList([torch.nn.Linear(self.patch[i], self.patch[i] ) for i in range(self.layers)])
        # self.s_pred_local_sigmoid2 = torch.nn.ModuleList([torch.nn.Linear(self.patch[i] * 4, self.patch[i]) for i in range(self.layers)])
        
        self.s_pred_global = torch.nn.ModuleList([torch.nn.Linear(self.seq_len // self.patch[i], self.seq_len // self.patch[i] * 4)
                                                    for i in range(self.layers)])
        self.s_pred_global2 = torch.nn.ModuleList([torch.nn.Linear(self.seq_len // self.patch[i] * 4, self.seq_len // self.patch[i])
                                                    for i in range(self.layers)])
        # self.s_pred_global_sigmoid = torch.nn.ModuleList([torch.nn.Linear(self.seq_len // self.patch[i], self.seq_len // self.patch[i] )
        #                                             for i in range(self.layers)])
        # self.s_pred_global_sigmoid2 = torch.nn.ModuleList([torch.nn.Linear(self.seq_len // self.patch[i] * 4, self.seq_len // self.patch[i])
        #                                             for i in range(self.layers)])
        self.s_bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])
        self.s_bn2 = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])


        self.t_pred_local = torch.nn.ModuleList([torch.nn.Linear(self.patch[i], self.patch[i] * 4)
                                            for i in range(self.layers)])
        self.t_pred_local2 = torch.nn.ModuleList([torch.nn.Linear(self.patch[i] * 4, self.patch[i])
                                            for i in range(self.layers)])
        # self.t_pred_local_sigmoid = torch.nn.ModuleList([torch.nn.Linear(self.patch[i], self.patch[i] )
        #                                     for i in range(self.layers)])
        # self.t_pred_local_sigmoid2 = torch.nn.ModuleList([torch.nn.Linear(self.patch[i] * 4, self.patch[i])
        #                                     for i in range(self.layers)])

        self.t_pred_global = torch.nn.ModuleList([torch.nn.Linear(self.seq_len // self.patch[i], self.seq_len // self.patch[i] * 4)
                                            for i in range(self.layers)])
        self.t_pred_global2 = torch.nn.ModuleList([torch.nn.Linear(self.seq_len // self.patch[i] * 4, self.seq_len // self.patch[i])
                                            for i in range(self.layers)])
        # self.t_pred_global_sigmoid = torch.nn.ModuleList([torch.nn.Linear(self.seq_len // self.patch[i], self.seq_len // self.patch[i] )
        #                                     for i in range(self.layers)])
        # self.t_pred_global_sigmoid2 = torch.nn.ModuleList([torch.nn.Linear(self.seq_len // self.patch[i] * 4, self.seq_len // self.patch[i])
        #                                     for i in range(self.layers)])

        self.t_bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])
        self.t_bn2 = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])

        # self.c_patch = int(math.sqrt(self.enc_in))
        # self.shang = self.enc_in//self.c_patch if self.enc_in%self.c_patch == 0 else (self.enc_in//self.c_patch+1)
        # self.c1 = torch.nn.Linear(self.c_patch,self.c_patch, bias=True)
        # self.c2 = torch.nn.Linear(self.shang,self.shang, bias=True)
        
        # self.c_patch2 = int(math.sqrt(self.enc_in))
        # self.shang2 = self.enc_in//self.c_patch2 if self.enc_in%self.c_patch2 == 0 else (self.enc_in//self.c_patch2+1)
        # self.c3 = torch.nn.Linear(self.c_patch2,self.c_patch2, bias=True)
        # self.c4 = torch.nn.Linear(self.shang2,self.shang2, bias=True)
        # self.c_bn = torch.nn.BatchNorm1d(self.pred_len)

        # self.s_bn = torch.nn.BatchNorm2d(self.shang*self.c_patch)
        # self.t_bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.shang*self.c_patch) for i in range(self.layers)])

        # self.channel_l1 = torch.nn.Linear(self.enc_in, self.enc_in, bias=False)
        # self.channel_l2 = torch.nn.Linear(self.enc_in, self.enc_in, bias=False)
        # self.l1 = torch.nn.Linear(self.seq_len, self.seq_len*4)
        # self.l2 = torch.nn.Linear(self.seq_len*4, self.seq_len)

        # self.x_enc_local = torch.nn.Linear(1, 4)
        # self.x_enc_local2 = torch.nn.Linear( 4, 1)
        # # self.s_pred_local3 = torch.nn.Linear(self.patch[-1], self.patch[-1])
        # self.x_enc_global = torch.nn.Linear(self.seq_len, self.pred_len * 4)
        # self.x_enc_global2 = torch.nn.Linear(self.pred_len * 4, self.pred_len)

        # self.x_enc_bn = torch.nn.BatchNorm2d(self.enc_in)
        self.pred_linear = torch.nn.Linear(self.seq_len, self.pred_len)

        self.moe = MoE(self.seq_len, len(self.patch))



    def encoder(self, x_enc, x_mark_enc):

        batch, seq, channel = x_enc.shape

        means2 = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means2
        stdev2 = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev2

        # channel
        # x_enc = torch.cat((x_enc, x_enc[:,:,:(self.shang*self.c_patch-self.enc_in)]), dim=-1).reshape(batch, self.seq_len, self.shang, self.c_patch)
        # # x_enc = self.c_bn(x_enc)
        # x_enc = x_enc + self.c2(self.dropout(self.gelu(self.c1(x_enc).permute(0,1,3,2)))).permute(0,1,3,2)
        # x_enc = x_enc.reshape(batch, self.seq_len, -1)[:,:,:self.enc_in]


        # means2 = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means2
        # stdev2 = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev2

        # x_enc = self.l2(self.dropout(self.gelu(self.l1(x_enc.permute(0,2,1))))).permute(0,2,1)
        
        # x_enc1 = x_enc.reshape(batch, seq, 1, channel)
        # x_enc1 = x_enc1 + self.x_enc_local2(self.dropout(self.gelu(self.x_enc_local(x_enc1.permute(0,1,3,2))))).permute(0,1,3,2)
        # x_enc1 = self.x_enc_bn(x_enc1.permute(0,3,1,2)).permute(0,2,3,1)
        # x_enc1 = self.x_enc_global2(self.dropout(self.gelu(self.x_enc_global(x_enc1.permute(0,3,2,1))))).permute(0,3,2,1).reshape(batch, self.pred_len, channel)
        
        weight = self.moe(x_enc.permute(0,2,1)) 
        result = []
        
        trend = x_enc
        # seaoson_pred = torch.zeros([batch, self.seq_len, channel], device=x_enc.device)
        trend_pred = torch.zeros([batch, self.seq_len, channel], device=x_enc.device)
        # trend_pred = trend_pred + x_enc1
        for i in range(self.layers):
            if self.patch[i] != 1:
                season, trend = self.decomp[i](x_enc)
            else:
                season = torch.zeros([batch, self.seq_len, channel], device=x_enc.device)
                trend = x_enc
            # 使用时间点进行计算，但是复杂度太高，对于长时间序列不太友好
            # tmp = smooth_with_local_fft_pytorch(trend.permute(0,2,1), self.patch[i]).permute(0,2,1)
            # season = trend - tmp
            # trend = tmp

            # 使用时间片段进行计算，时间片段和进行FFT的滑动窗口大小相同
            # tmp = smooth_time_series_local_fft_pytorch(trend.permute(0,2,1), self.patch[i]).permute(0,2,1)
            # season = trend - tmp
            # trend = tmp

            # tmp = trend
            # trend = trend.permute(0,2,1)
            # trend = trend.reshape(batch, self.enc_in, seq//self.patch[i], self.patch[i])
            # trend = torch.fft.rfft(trend, dim=-1)
            # cutoff = trend.shape[-1] // 3
            # trend[..., cutoff:] = 0
            # trend = (torch.fft.irfft(trend, dim=-1))
            # trend = trend.reshape(batch, self.enc_in, seq).permute(0,2,1)
            # season = (tmp-trend)

            trend1 = trend.reshape(batch, seq//self.patch[i], self.patch[i], channel)
            trend1 = trend1 + self.t_pred_local2[i](self.dropout(self.gelu(self.t_pred_local[i](trend1.permute(0,1,3,2))))).permute(0,1,3,2)
            # trend1 = trend1 + t1
            # t_sigmoid = self.t_pred_local_sigmoid2[i](self.dropout(self.gelu(self.t_pred_local_sigmoid[i](trend1.permute(0,1,3,2))))).permute(0,1,3,2)
            # t_sigmoid = self.t_pred_local_sigmoid[i](trend1.permute(0,1,3,2)).permute(0,1,3,2)
            # trend1 = trend1 + torch.mul(self.sigmoid(t_sigmoid), self.gelu(t1))
            trend1 = self.t_bn[i](trend1.permute(0,3,1,2)).permute(0,2,3,1)
            
            trend1 = trend1 +  self.t_pred_global2[i](self.dropout(self.gelu(self.t_pred_global[i](trend1.permute(0,3,2,1))))).permute(0,3,2,1)
            # trend1 = trend1 + t1
            # # t_sigmoid = self.t_pred_global_sigmoid2[i](self.dropout(self.gelu(self.t_pred_global_sigmoid[i](trend1.permute(0,3,2,1))))).permute(0,3,2,1)
            # # t_sigmoid = self.t_pred_global_sigmoid[i](trend1.permute(0,3,2,1)).permute(0,3,2,1)
            # # trend1 = trend1 + torch.mul(self.sigmoid(t_sigmoid), self.gelu(t1))
            trend1 = self.t_bn2[i](trend1.permute(0,3,1,2)).permute(0,2,3,1)
            trend1 = trend1.reshape(batch, self.seq_len, channel)

            

            season = season.reshape(batch, seq//self.patch[i], self.patch[i], channel)
            season = season + self.s_pred_local2[i](self.dropout(self.gelu(self.s_pred_local[i](season.permute(0,1,3,2))))).permute(0,1,3,2)
            # season = season + s1
            # s_sigmoid = self.s_pred_local_sigmoid2[i](self.dropout(self.gelu(self.s_pred_local_sigmoid[i](season.permute(0,1,3,2))))).permute(0,1,3,2)
            # s_sigmoid = self.s_pred_local_sigmoid[i](season.permute(0,1,3,2)).permute(0,1,3,2)
            # season = season + torch.mul(self.sigmoid(s_sigmoid), self.gelu(s1))
            season = self.s_bn[i](season.permute(0,3,1,2)).permute(0,2,3,1)
            
            season =  season + self.s_pred_global2[i](self.dropout(self.gelu(self.s_pred_global[i](season.permute(0,3,2,1))))).permute(0,3,2,1)
            # season = season + s1
            # # s_sigmoid = self.s_pred_global_sigmoid2[i](self.dropout(self.gelu(self.s_pred_global_sigmoid[i](season.permute(0,3,2,1)))))
            # # s_sigmoid = self.s_pred_global_sigmoid[i](season.permute(0,3,2,1))
            season = self.s_bn2[i](season.permute(0,3,1,2)).permute(0,2,3,1)
            season = season.reshape(batch, self.seq_len, channel)

            # x_enc =   (trend1 + season)
            # trend_pred = trend_pred + trend1 + season
            result.append((trend1+season).permute(0,2,1))


        dec_out = (torch.matmul(torch.stack(result, dim=-1),weight.unsqueeze(-1)) ).squeeze(-1).permute(0,2,1)

        # dec_out = trend_pred
        dec_out = self.pred_linear(dec_out.permute(0,2,1)).permute(0,2,1)

        # dec_out = dec_out * stdev2 + means2
        # dec_out = x_enc

        # channel
        # 经过实验，发现放在最后面比放在最前面好，同时前面和后面都加上更好
        # 好像加不加batch normalization都一样，而且加在cat之前还是加在cat之后好像没有差别
        # 当然，还是加在cat之前要更好一点点点点，可能是加在cat之后会引入一些噪声
        # dec_out = self.c_bn(dec_out)
        # dec_out = torch.cat((dec_out, dec_out[:,:,:self.shang2*self.c_patch2-self.enc_in]), dim=-1).reshape(batch, self.pred_len, self.shang2, self.c_patch2)
        # # dec_out = self.c_bn(dec_out)
        # dec_out = dec_out + self.c4((self.dropout(self.gelu(self.c3(dec_out).permute(0,1,3,2))))).permute(0,1,3,2)
        # dec_out = dec_out.reshape(batch, self.pred_len, -1)[:,:,:self.enc_in]

        dec_out = dec_out * stdev2 + means2

        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.encoder(x_enc, x_mark_enc)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        return self.encoder(x_enc)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]

        return None

