import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from mamba_ssm import Mamba


class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        # if self.training:
        ratio = F.softmax(combined_mean, dim=1)
        ratio = ratio.permute(0, 2, 1)
        ratio = ratio.reshape(-1, channels)
        indices = torch.multinomial(ratio, 1)
        indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
        combined_mean = torch.gather(combined_mean, 1, indices)
        combined_mean = combined_mean.repeat(1, channels, 1)
        # else:
            # weight = F.softmax(combined_mean, dim=1)
            # combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat

        return output 


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(*normalized_shape))  

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=(-2, -1), keepdim=True) + self.eps)
        return self.scale * x / rms


class ResBlock(nn.Module):
    def __init__(self, configs, seq_len=96):
        super(ResBlock, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = seq_len
        self.d_model = configs.d_model


        self.channel_function = configs.channel_function
        self.temporal_function = configs.temporal_function
        self.d_core = 32

        if self.temporal_function == 'patch':
            self.temporal_patch = nn.Sequential(
                RMSNorm([self.enc_in,self.seq_len]),
                nn.Linear(self.seq_len, configs.d_model),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, self.seq_len),
                nn.Dropout(configs.dropout)
            )
            self.patch = configs.patch
            self.patch_num = [self.seq_len // i for i in self.patch]
            self.decomp = torch.nn.ModuleList([series_decomp(i+1) for i in self.patch])
            self.temporal1 = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch_num[i],self.patch[i]]),
                nn.Linear(self.patch[i], self.patch[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch[i]),
                nn.Dropout(configs.dropout)
            ) for i in range(len(self.patch))])
            self.temporal2 = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch[i],self.patch_num[i]]),
                nn.Linear(self.patch_num[i], self.patch[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch_num[i]),
                nn.Dropout(configs.dropout)
            )  for i in range(len(self.patch))])
            self.temporal1_season = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch_num[i],self.patch[i]]),
                nn.Linear(self.patch[i], self.patch[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch[i]),
                nn.Dropout(configs.dropout)
            ) for i in range(len(self.patch))])
            self.temporal2_season = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch[i],self.patch_num[i]]),
                nn.Linear(self.patch_num[i], self.patch[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch_num[i]),
                nn.Dropout(configs.dropout)
            )  for i in range(len(self.patch))])
            self.linear_patch = torch.nn.ModuleList([nn.Linear(self.seq_len, self.seq_len) 
                    for i in range(len(self.patch))])

        if self.temporal_function == 'normal':
            self.temporal = nn.Sequential(
                RMSNorm([self.enc_in,self.seq_len]),
                nn.Linear(self.seq_len, configs.d_model),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, self.seq_len),
                nn.Dropout(configs.dropout)
            )

        if self.temporal_function == 'down':
            self.kernel = configs.patch
            self.layers = len(self.kernel)
            self.temporal_down = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.enc_in, self.seq_len//self.kernel[i]]),
                nn.Linear(self.seq_len//self.kernel[i], configs.d_model ),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model , self.seq_len//self.kernel[i] ),
                nn.Dropout(configs.dropout),
            ) for i in range(self.layers)])
            self.linear_down = torch.nn.ModuleList([nn.Linear(self.seq_len//self.kernel[i], self.seq_len) 
                    for i in range(self.layers)])


        if self.channel_function == 'RNN':
            self.norm = RMSNorm([self.enc_in,self.seq_len])
            self.linear1 = nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Dropout(configs.d2),
            )
            self.lstm = torch.nn.LSTM(input_size=self.seq_len,hidden_size=self.seq_len,
                                    num_layers=1,batch_first=True, bidirectional=True)
            self.pro = nn.Sequential( 
                torch.nn.Linear(self.seq_len*2, configs.seq_len),
                nn.SiLU(),
                nn.Dropout(configs.d2), 
            )

        if self.channel_function == 'RNN2':
            if configs.freq == 't':
                if configs.n_patch == -1:
                    self.n_patch = int(math.sqrt(configs.enc_in+5)) 
                else:
                    self.n_patch = configs.n_patch
                self.c_patch = (configs.enc_in+5) // self.n_patch + 1
            elif configs.freq == 'h':
                if configs.n_patch == -1:
                    self.n_patch = int(math.sqrt(configs.enc_in+4)) 
                else:
                    self.n_patch = configs.n_patch
                self.c_patch = (configs.enc_in+4) // self.n_patch + 1
            elif configs.freq == 'd':
                if configs.n_patch == -1:
                    self.n_patch = int(math.sqrt(configs.enc_in+3)) 
                else:
                    self.n_patch = configs.n_patch
                self.c_patch = (configs.enc_in+3) // self.n_patch + 1
            else:
                if configs.n_patch == -1:
                    self.n_patch = int(math.sqrt(configs.enc_in)) 
                else:
                    self.n_patch = configs.n_patch
                self.c_patch = configs.enc_in // self.n_patch + 1

            self.linear = nn.Sequential(
                torch.nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model,
                        kernel_size=1,stride=1,padding=0),
                torch.nn.SiLU(),
                torch.nn.Dropout(configs.dropout),
            )
            self.l1 = torch.nn.Conv1d(in_channels=configs.seq_len, out_channels=configs.d_model,
                    kernel_size=1, stride=1, groups=1)
            self.l1_trans = torch.nn.Conv1d(in_channels=configs.d_model, out_channels=configs.seq_len,
                    kernel_size=1, stride=1, groups=1)
            self.norm_lstm = torch.nn.LayerNorm(self.d_model)
            # self.norm_lstm = torch.nn.BatchNorm2d(self.seq_len)
            # self.norm_lstm = RMSNorm([self.c_patch, self.seq_len])
            self.lstm = torch.nn.LSTM(input_size=self.d_model,hidden_size=self.d_model,
                                    num_layers=1,batch_first=True, bidirectional=True)
            self.lstm_linear = nn.Sequential( 
                torch.nn.Conv1d(in_channels=self.d_model*2, out_channels=self.d_model,
                        kernel_size=1,stride=1,padding=0),
                nn.SiLU(),
                nn.Dropout(configs.dropout), 
            )
            self.lstm2 = torch.nn.LSTM(input_size=self.d_model, hidden_size=self.d_model,
                                    num_layers=1,batch_first=True, bidirectional=True)
            self.lstm_linear2 = nn.Sequential( 
                torch.nn.Conv1d(in_channels=self.d_model*2, out_channels=self.d_model,
                        kernel_size=1,stride=1,padding=0),
                nn.SiLU(),
                nn.Dropout(configs.dropout), 
            )



        if self.channel_function == 'MLP':
            self.final_linear = nn.Sequential(
                RMSNorm([self.seq_len,self.enc_in]),
                nn.Linear(self.enc_in, self.enc_in//4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.enc_in//4, self.enc_in),
                nn.Dropout(configs.dropout),
            )  
        if self.channel_function == 'STAR':
            self.final_linear = nn.Sequential(
                RMSNorm([self.enc_in,self.seq_len]),
                STAR(self.seq_len, configs.d_model)
            )  
        if self.channel_function == 'Mamba':
            self.mamba = nn.Sequential(
                RMSNorm([self.enc_in,self.seq_len]),
                Mamba(
                    d_model = configs.seq_len,
                    d_state = 16,
                    d_conv = 2,
                    expand = 1,
                )
            )
            # self.mamba2 = nn.Sequential(
            #     RMSNorm([self.enc_in,self.seq_len]),
            #     Mamba(
            #         d_model = configs.seq_len,
            #         d_state = 16,
            #         d_conv = 2,
            #         expand = 1,
            #     )
            # )

    def forward(self, x):
        B, L, D = x.shape

        if self.temporal_function == 'patch':
            add = torch.zeros([B, L, D], device=x.device)
            for i in range(len(self.patch)):
                if self.patch[i] == 1:
                    add = x + self.temporal_patch((x).transpose(1, 2)).transpose(1, 2)
                else:
                    season, x_group = self.decomp[i](x)
                    x_group = x_group.permute(0,2,1)
                    x_group = x_group.reshape(B, D, self.patch_num[i], self.patch[i])
                    x_group = x_group + self.temporal1[i](x_group)
                    x_group = x_group.permute(0,1,3,2)
                    x_group = x_group + self.temporal2[i](x_group)
                    x_group = x_group.permute(0,1,3,2).reshape(B, D, -1).permute(0,2,1)
                    season = season.permute(0,2,1)
                    season = season.reshape(B, D, self.patch_num[i], self.patch[i])
                    season = season + self.temporal1_season[i](season)
                    season = season.permute(0,1,3,2)
                    season = season + self.temporal2_season[i](season)
                    season = season.permute(0,1,3,2).reshape(B, D, -1).permute(0,2,1)
                    add = add + self.linear_patch[i]((x_group + season).permute(0,2,1)).permute(0,2,1) 
            x = add/(len(self.patch))

        if self.temporal_function == 'normal':
            x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)

        if self.temporal_function == 'down':
            add = torch.zeros([B, L, D], device=x.device)
            for i in range(self.layers):
                tmp =  torch.nn.AvgPool1d(kernel_size=self.kernel[i])(x.transpose(1, 2)) + torch.nn.MaxPool1d(kernel_size=self.kernel[i])(x.transpose(1, 2))  
                tmp = tmp + self.temporal_down[i](tmp)
                tmp = self.linear_down[i](tmp)
                add = add + tmp.permute(0,2,1)
            x = add/(self.layers)

        if self.channel_function == 'MLP':
            x = x + self.final_linear(x)
        if self.channel_function == 'STAR':
            x = x.permute(0,2,1)
            x = x + self.final_linear(x)
            x = x.permute(0,2,1)
        if self.channel_function == 'RNN':
            x = x.permute(0,2,1)
            h0 = torch.randn(2, B, self.seq_len, device=x.device)
            c0 = torch.randn(2, B, self.seq_len, device=x.device)
            x = x + torch.mul(self.linear1(x), self.pro(self.lstm(self.norm(x), (h0,c0))[0]))
            x = x.permute(0,2,1)
        if self.channel_function == 'RNN2':
            # x = self.l1(x)
            x = torch.cat((x, x[:,:,:(self.c_patch*self.n_patch-D)]), dim=-1)
            x = x.reshape(B, self.d_model, self.c_patch, self.n_patch)
            kv = x
            # x = self.norm_lstm(x)
            x = x.reshape(B*self.n_patch, self.d_model, self.c_patch).permute(0,2,1)

            x1, x2 = self.lstm(self.norm_lstm(x))
            x1 = self.lstm_linear(x1.permute(0,2,1))
            x1 = x1.reshape(B, self.d_model, self.c_patch, self.n_patch)
            x2 = torch.sum(x2[1].permute(1,0,2), dim=1, keepdim=True).reshape(B, self.n_patch, self.d_model)
            x21, x22 = self.lstm2((x2)) 
            x21 = self.lstm_linear2(x21.permute(0,2,1)).unsqueeze(-2)
            x = kv + torch.mul(self.linear(kv), (x1+ torch.mul(x21, x1) )) 
            x = x.reshape(B, self.d_model, self.c_patch*self.n_patch).contiguous()[:,:,:D]
            # x = self.l1_trans(x)


        if self.channel_function == 'Mamba':
            x = x.permute(0,2,1)
            x = x + self.mamba(x) 
            x = x.permute(0,2,1)

        # add = torch.zeros([B, L, D], device=x.device)
        # for i in range(self.layers):
        #     tmp =  torch.nn.AvgPool1d(kernel_size=self.kernel[i])(x.transpose(1, 2)) + torch.nn.MaxPool1d(kernel_size=self.kernel[i])(x.transpose(1, 2))  
        #     tmp = tmp + self.temporal_down[i](tmp)
        #     tmp = self.linear_down[i](tmp)
        #     add = add + tmp.permute(0,2,1)
        # x = add/(self.layers) 

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.freq = configs.freq

        self.model = nn.ModuleList([ResBlock(configs, seq_len=self.seq_len)
                                    for _ in range(configs.e_layers)])

        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.use_norm = configs.use_norm

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch, seq, channel = x_enc.shape

        # if self.freq == 't' or self.freq == 'h' or self.freq == 'd':
        #     x_enc = torch.cat((x_enc, x_mark_enc), dim=-1)

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()       
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection((x_enc ).transpose(1, 2)).transpose(1, 2)
        
        if self.use_norm:
            enc_out = enc_out *stdev + means

        return enc_out[:,:,:self.enc_in], x_enc


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, x_enc  = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]   # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
