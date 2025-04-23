
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
        cutoff = fft_local_data.shape[-1] // 4
        fft_local_data[..., cutoff:] = 0
        smoothed_local_data = torch.real(fft.ifft(fft_local_data, dim=-1))
        smoothed_vector[..., start:end] = smoothed_local_data
    return smoothed_vector



class ComplexMultiplication(nn.Module):
    def __init__(self, window_size1, window_size2, rate=0.1, enc_in=7):
        super(ComplexMultiplication, self).__init__()
        self.enc_in = enc_in
        self.window_size2 = window_size2
        self.window_size1 = window_size1
        self.a1 = window_size1 * 1
        self.a2 = window_size2 * 1
        # 分别为实部和虚部创建线性层
        self.real_linear = nn.Linear(self.a1, self.a2*4)  # c
        self.real_linear2 = nn.Linear(self.a2*4, self.a2)  # c
        self.imag_linear = nn.Linear(self.a1, self.a2*4)  # d
        self.imag_linear2 = nn.Linear(self.a2*4, self.a2)  # d

        # self.real_linear_sigmoid = nn.Linear(window_size1, window_size2  )  # c
        # # self.real_linear_sigmoid2 = nn.Linear(window_size2*4, window_size2)  # c
        # self.imag_linear_sigmoid = nn.Linear(window_size1, window_size2 )  # d
        # # self.imag_linear_sigmoid2 = nn.Linear(window_size2*4, window_size2)  # d

        self.sigmoid = torch.nn.Sigmoid()
        self.rate = rate
        # self.relu = torch.nn.ReLU()
        # relu和gelu都可以，总的来说，gelu和relu性能一样
        self.relu = torch.nn.LeakyReLU(self.rate)
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(self.rate)


        self.bn = torch.nn.BatchNorm2d(self.enc_in)
        self.bn2 = torch.nn.BatchNorm2d(self.enc_in)


    def forward(self, input_complex):
        
        # 提取输入复数的实部和虚部
        real_part = input_complex.real  # a
        r1 = real_part
        imag_part = input_complex.imag  # b
        p1 = imag_part
        real_part[:,:,:,self.a1//2:] = 0
        imag_part[:,:,:,self.a1//2:] = 0
        real_part = self.bn(real_part)
        imag_part = self.bn2(imag_part)

        real_part2 = r1+  self.dropout(self.real_linear2(self.dropout(self.relu(self.real_linear(real_part)))))-self.dropout(self.imag_linear2(self.dropout(self.relu(self.imag_linear(imag_part)))))
        imag_part2 = p1+  self.dropout(self.real_linear2(self.dropout(self.relu(self.real_linear(imag_part)))))+self.dropout(self.imag_linear2(self.dropout(self.relu(self.imag_linear(real_part)))))
        # real_part = self.bn(real_part2)
        # imag_part = self.bn2(imag_part2)
        # real_part2 = self.real_linear2(self.dropout(self.relu(self.real_linear(real_part))))
        # imag_part2 = self.imag_linear2(self.dropout(self.relu(self.imag_linear(real_part))))

        # real_part_sigmoid = (((self.real_linear_sigmoid(real_part))))-(((self.imag_linear_sigmoid(imag_part))))
        # imag_part_sigmoid = (((self.real_linear_sigmoid(imag_part))))+(((self.imag_linear_sigmoid(real_part))))
        
        # real_part_sigmoid = self.real_linear_sigmoid2(self.dropout(self.relu(self.real_linear_sigmoid(real_part))))-self.imag_linear_sigmoid2(self.dropout(self.relu(self.imag_linear_sigmoid(imag_part))))
        # imag_part_sigmoid = self.real_linear_sigmoid2(self.dropout(self.relu(self.real_linear_sigmoid(imag_part))))+self.imag_linear_sigmoid2(self.dropout(self.relu(self.imag_linear_sigmoid(real_part))))

        # real_part2 = r1 +  torch.mul(self.sigmoid(real_part_sigmoid), self.tanh(real_part2))
        # imag_part2 = p1 +  torch.mul(self.sigmoid(imag_part_sigmoid), self.tanh(imag_part2))

        # real_part2 = torch.cat((real_part2, torch.zeros([real_part.shape[0], real_part.shape[1],real_part.shape[2], self.window_size2-self.a2], device=real_part.device)),dim=-1)
        # imag_part2 = torch.cat((imag_part2, torch.zeros([imag_part.shape[0], imag_part.shape[1],imag_part.shape[2], self.window_size2-self.a2], device=imag_part.device)),dim=-1)

        # 组合实部和虚部得到输出复数
        return torch.complex(real_part2, imag_part2)

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

        self.rate = configs.dropout
        self.dropout = torch.nn.Dropout(self.rate)
        # self.relu = torch.nn.LeakyReLU(0.1)
        self.gelu = torch.nn.LeakyReLU(self.rate)

        self.patch = configs.patch
        self.n_patch = [self.seq_len//self.patch[i] if self.seq_len%self.patch[i]==0 else self.seq_len//self.patch[i]+1 for i in range(len(self.patch))]
        # print(self.n_patch)
        self.layers = len(self.patch)



        self.freq_linear_local = torch.nn.ModuleList([ComplexMultiplication(p//2+1, p//2+1, self.rate, self.enc_in) for p in self.patch])
        self.freq_linear_global = torch.nn.ModuleList([ComplexMultiplication(p//2+1, p//2+1, self.rate, self.enc_in) for p in self.n_patch])
        # self.linear2 = torch.nn.ModuleList([torch.nn.Linear(self.seq_len, self.seq_len) for i in range(len(self.patch))])
        # self.pred = ComplexMultiplication(self.seq_len//2+1, self.pred_len//2+1)
        # self.freq_linear_local_s = torch.nn.ModuleList([ComplexMultiplication(p //2+1,p //2+1, self.rate) for p in self.patch])
        # self.freq_linear_global_s = torch.nn.ModuleList([ComplexMultiplication((self.seq_len//p)//2+1 ,(self.pred_len//p)//2+1, self.rate) for p in self.patch])
        
        # self.freq_bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])
        # self.freq_bn2 = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])

        # self.freq_bn3 = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])
        # self.freq_bn4 = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.enc_in) for i in range(self.layers)])

        self.pred = torch.nn.ModuleList([torch.nn.Linear(self.n_patch[i]*self.patch[i], self.pred_len)for i in range(self.layers)])
        # self.pred = torch.nn.Linear(self.seq_len, self.pred_len) 
        self.moe = MoE(self.seq_len, len(self.patch), self.rate)
        self.w = torch.ones(len(self.patch), requires_grad=True)

        self.channel_patch = int(math.sqrt(self.enc_in))+1
        self.new_channel = self.channel_patch * self.channel_patch

        self.channel1 = torch.nn.ModuleList([torch.nn.Linear(self.channel_patch, self.channel_patch*4)
                            for i in range(self.layers)])
        self.channel2 = torch.nn.ModuleList([torch.nn.Linear(self.channel_patch*4, self.channel_patch)
                            for i in range(self.layers)])
        self.c_bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.n_patch[i]*self.patch[i])
                            for i in range(self.layers)])
        self.c_bn2 = torch.nn.ModuleList([torch.nn.BatchNorm2d(self.n_patch[i]*self.patch[i])
                            for i in range(self.layers)])
        self.channel3 = torch.nn.ModuleList([torch.nn.Linear(self.channel_patch, self.channel_patch*4)
                            for i in range(self.layers)])
        self.channel4 = torch.nn.ModuleList([torch.nn.Linear(self.channel_patch*4, self.channel_patch)
                            for i in range(self.layers)])


        
    #     self.initialize_weight(self.pred)

    # def initialize_weight(self, x):
    #     nn.init.xavier_uniform_(x.weight)
    #     if x.bias is not None:
    #         nn.init.constant_(x.bias, 0)

    def encoder(self, x_enc, x_mark_enc):

        batch, seq, channel = x_enc.shape

        means2 = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means2
        stdev2 = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev2

        # channel_w = torch.mean(x_enc, dim=-1, keepdim=True)
        # d = 1/torch.pow(x_enc-channel_w, 2)
        # d = torch.nn.Sigmoid()(d )
        # x_enc = x_enc + torch.mul(x_enc, d)

        # x_enc = x_enc.permute(0,2,1)

        weight = self.moe(x_enc.permute(0,2,1)) 
        result = []

        x_pred = torch.zeros([batch, self.pred_len, channel], device=x_enc.device)

        for i in range(self.layers):
            new_seqlen = self.n_patch[i]*self.patch[i]
            trend1 = torch.cat((x_enc, x_enc[:,:new_seqlen-self.seq_len,:]), dim=1)
            
            trend1 = trend1.reshape(batch, self.n_patch[i], self.patch[i], channel).permute(0,3,1,2)
            if self.patch[i] != 1:
                # x1 = trend1
                # trend1 = self.freq_bn[i](trend1)
                trend1 = torch.fft.rfft(trend1, dim=-1) 
                trend1 = self.freq_linear_local[i](trend1)
                trend1 = torch.real(torch.fft.irfft(trend1, dim=-1))
                # trend1 = trend1 + x1

            trend1 = trend1.permute(0,1,3,2)
            # x1 = trend1
            # trend1 = self.freq_bn2[i](trend1)
            trend1 = torch.fft.rfft(trend1, dim=-1) 
            trend1 = self.freq_linear_global[i](trend1)
            trend1 = torch.real(torch.fft.irfft(trend1, dim=-1))
            # trend1 = trend1 + x1
            trend1 = trend1.permute(0,1,3,2)

            # print(trend1.shape)
            trend1 = trend1.reshape(batch, self.enc_in, new_seqlen)

            trend1 = trend1.permute(0,2,1)
            trend1 = torch.cat((trend1, trend1[:,:,:self.channel_patch*self.channel_patch-self.enc_in]),dim=-1)
            trend1 = trend1.reshape(batch, new_seqlen, self.channel_patch, self.channel_patch)
            x1 = trend1
            trend1 = self.c_bn[i](trend1)
            trend1 = x1 + self.dropout(self.channel2[i](self.dropout(self.gelu(self.channel1[i](trend1)))))
            x1 = trend1
            trend1 = self.c_bn2[i](trend1)
            trend1 = x1 + self.dropout(self.channel4[i](self.dropout(self.gelu(self.channel3[i](trend1.permute(0,1,3,2))))).permute(0,1,3,2))
            trend1 = trend1.reshape(batch, new_seqlen, -1)[:,:,:self.enc_in]
            trend1 = trend1.permute(0,2,1)

            # x_pred = (x_pred + self.pred[i](trend1).permute(0,2,1)) 
            result.append(self.pred[i](trend1))
            # result.append((trend1))

        x_pred = torch.mean(torch.mul(torch.stack(result, dim=-1), self.w.to(x_enc.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)),dim=-1).permute(0,2,1)
        # x_pred = (torch.matmul(torch.stack(result, dim=-1),weight.unsqueeze(-1)) ).squeeze(-1).permute(0,2,1)

        dec_out = x_pred * stdev2 + means2

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
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]

        return None

