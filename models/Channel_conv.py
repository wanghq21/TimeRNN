import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
from layers.RevIN import RevIN
from layers.SelfAttention_Family import AttentionLayer, ProbAttention, FullAttention
import torch.nn.init as init
from layers.Embed import PositionalEmbedding
from mamba_ssm import Mamba
# from layers.Conv_Blocks import Inception_Block_V1



class BiRNN2(nn.Module):
    def __init__(self, enc_in=21, d_model=96, n_patch=5, dropout=0.1):
        super(BiRNN2, self).__init__()
        self.d_model = d_model

        if n_patch == -1:
            self.n_patch = int(math.sqrt(enc_in)) 
        else:
            self.n_patch = n_patch
        self.c_patch = enc_in // self.n_patch + 1


        self.linear = nn.Sequential(
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
        )
        self.norm_lstm = torch.nn.LayerNorm(self.d_model)
        # self.norm_lstm = RMSNorm([self.c_patch, self.d_model])
        self.lstm = torch.nn.GRU(input_size=self.d_model,hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.lstm_linear = nn.Sequential( 
            torch.nn.Conv1d(in_channels=self.d_model*2, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0),            
            nn.SiLU(),
            nn.Dropout(dropout), 
        )

        self.lstm2 = torch.nn.GRU(input_size=self.d_model, hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.lstm_linear2 = nn.Sequential( 
            torch.nn.Conv1d(in_channels=self.d_model*2, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0),  
            nn.SiLU(),
            nn.Dropout(dropout), 
        )


    def forward(self, x):
        batch, seq, channel = x.shape

        x = torch.cat((x, x[:,:,:(self.c_patch*self.n_patch-channel)]), dim=-1)
        x = x.reshape(batch, self.d_model, self.c_patch, self.n_patch)
        kv = x
        x = x.reshape(batch*self.n_patch, self.d_model, self.c_patch).permute(0,2,1)

        x1, x2 = self.lstm(self.norm_lstm(x))
        # x1 = x1[:,:,:self.d_model] + x1[:,:,-self.d_model:]
        x1 = self.lstm_linear(x1.permute(0,2,1))
        x1 = x1.reshape(batch, self.d_model, self.c_patch, self.n_patch)
        x2 = torch.sum(x2.permute(1,0,2), dim=1, keepdim=True).reshape(batch, self.n_patch, self.d_model)
        x21, x22 = self.lstm2((x2)) 
        # x21 = x21[:,:,:self.d_model] + x21[:,:,-self.d_model:]
        x21 = self.lstm_linear2(x21.permute(0,2,1)).unsqueeze(-2)
        x = kv + torch.mul(self.linear(kv), (x1+ torch.mul(x21, x1) )) 
        x = x.reshape(batch, self.d_model, self.c_patch*self.n_patch).contiguous()[:,:,:channel]

        return x


class BiRNN(nn.Module):
    def __init__(self, enc_in=21, d_model=96, n_patch=5, dropout=0.1):
        super(BiRNN, self).__init__()
        self.d_model = d_model

        self.norm = torch.nn.LayerNorm(self.d_model)
        self.linear1 = nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
        )
        self.lstm = torch.nn.GRU(input_size=self.d_model,hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.pro = nn.Sequential( 
            # torch.nn.Conv1d(in_channels=self.d_model*2, out_channels=self.d_model,
            #         kernel_size=1,stride=1,padding=0),  
            nn.SiLU(),
            nn.Dropout(dropout), 
        )


    def forward(self, x):
        batch, seq, channel = x.shape

        x = x.permute(0,2,1)
        x1 = self.lstm(self.norm(x))[0]
        x1 = x1[:,:,:self.d_model] + x1[:,:,-self.d_model:]
        x = x + torch.mul(self.linear1(x), (self.pro(x1.permute(0,2,1)).permute(0,2,1)))
        x = x.permute(0,2,1)


        return x



def FFT_for_Period(x, k=3):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, groups=1, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i,groups=groups))
        self.kernels = nn.ModuleList(kernels)
        # kernels_gate = []
        # for i in range(self.num_kernels):
        #     kernels_gate.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i,groups=out_channels))
        # self.kernels_gate = nn.ModuleList(kernels_gate)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            # res_list.append(torch.mul(self.relu(self.kernels[i](x)), self.sigmoid(self.kernels_gate[i](x))))
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res



class Inception_Block_V1_1d(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, groups=1, init_weight=True):
        super(Inception_Block_V1_1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i,groups=groups))
        self.kernels = nn.ModuleList(kernels)
        # kernels_gate = []
        # for i in range(self.num_kernels):
        #     kernels_gate.append(nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i,groups=out_channels))
        # self.kernels_gate = nn.ModuleList(kernels_gate)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            # res_list.append(torch.mul(self.relu(self.kernels[i](x)), self.sigmoid(self.kernels_gate[i](x))))
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(*normalized_shape))  # 可学习的缩放参数

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=(-2, -1), keepdim=True) + self.eps)
        return self.scale * x / rms

class FFN_MOE(nn.Module):
    def __init__(self, configs):
        super(FFN_MOE, self).__init__()
        self.patch = [1,8,16,32]
        self.kernel = torch.nn.ModuleList([(nn.Sequential(
                # torch.nn.BatchNorm2d(self.seq_len),
                torch.nn.Conv2d(in_channels=configs.d_model, out_channels=configs.d_ff,
                        kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=i),
                torch.nn.ReLU(),
                torch.nn.Dropout(configs.dropout),
                torch.nn.Conv2d(in_channels=configs.d_ff, out_channels=configs.d_model,
                        kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=i),
                # torch.nn.Dropout(configs.dropout),
                )
            ) for i in self.patch])
        self.final = nn.Sequential(
            torch.nn.Conv2d(in_channels=configs.d_model, out_channels=configs.d_ff,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv2d(in_channels=configs.d_ff, out_channels=configs.d_model,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
            # torch.nn.Dropout(configs.dropout),
        )
        # self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        res = []
        for i in range(len(self.patch)):
            res.append(self.kernel[i](x))
        x = torch.stack(res, dim=-1).mean(-1)
        x = self.final(x)
        
        return x


class ResBlock_freq(nn.Module):
    def __init__(self, configs, seq_len=96, groups=1):
        super(ResBlock_freq, self).__init__()
        self.d_model = configs.d_model  
        # if configs.freq == 't':
        #     self.enc_in = configs.enc_in+4
        # elif configs.freq == 'h':
        #     self.enc_in = configs.enc_in+4
        # elif configs.freq == 'd':
        #     self.enc_in = configs.enc_in+3
        # else:
        #     self.enc_in = configs.enc_in

        self.channel_patch = int(math.sqrt(configs.enc_in))+1
        self.channel_patch2 = self.channel_patch

        self.d_core = 32
        self.num_kernel = 5

        # MLP
        self.linear1 = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Linear(self.channel_patch2, self.d_core),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch2),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.channel_patch2, self.d_core),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch2),
        )
        self.linear2 = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Linear(self.channel_patch, self.d_core),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.channel_patch, self.d_core),      
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch),    
        )
        self.linear_ffn = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv2d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
        )
        # self.linear_ffn = nn.Sequential(
        #     torch.nn.BatchNorm2d(self.d_model),
        #     FFN_MOE(configs),
        # )


        self.conv = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            Inception_Block_V1(self.d_model, self.d_model, num_kernels=self.num_kernel, groups=self.d_model),
        )
        self.conv_ffn = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv2d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
        )
        # self.conv_ffn = nn.Sequential(
        #     torch.nn.BatchNorm2d(self.d_model),
        #     FFN_MOE(configs),
        # )


        # MLP
        self.linear11 = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Linear(self.channel_patch2, self.d_core),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch2),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.channel_patch2, self.d_core),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch2),
        )
        self.linear21 = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Linear(self.channel_patch, self.d_core),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.channel_patch, self.d_core),      
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch),    
        )
        self.linear_ffn1 = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv2d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
        )
        # self.linear_ffn = nn.Sequential(
        #     torch.nn.BatchNorm2d(self.d_model),
        #     FFN_MOE(configs),
        # )


        self.conv1 = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            Inception_Block_V1(self.d_model, self.d_model, num_kernels=self.num_kernel, groups=self.d_model),
        )
        self.conv_ffn1 = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv2d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
        )
        # self.conv_ffn = nn.Sequential(
        #     torch.nn.BatchNorm2d(self.d_model),
        #     FFN_MOE(configs),
        # )


    def forward(self, x):
        batch, seq, channel = x.shape

        # temporal 和 channel谁在前和谁在后没有区别
        freq = torch.fft.fft(x, dim=-2)
        real, imag = freq.real, freq.imag
        

        # real = torch.cat((real, real[:,:,:self.channel_patch*self.channel_patch2-channel]),dim=-1)
        # real = real.reshape(batch, self.d_model, self.channel_patch, self.channel_patch2)
        # imag = torch.cat((imag, imag[:,:,:self.channel_patch*self.channel_patch2-channel]),dim=-1)
        # imag = imag.reshape(batch, self.d_model, self.channel_patch, self.channel_patch2)

        # real_1 = real + self.conv(real) - self.conv1(imag)
        # imag_1 = imag + self.conv1(real) + self.conv(imag)

        # real_2 = real_1 + self.conv_ffn(real_1) - self.conv_ffn1(imag_1)
        # imag_2 = imag_1 + self.conv_ffn1(real_1) + self.conv_ffn(imag_1)

        # real_3 = real_2 + self.linear1(real_2) - self.linear11(imag_2)
        # imag_3 = imag_2 + self.linear11(real_2) + self.linear1(imag_2)

        # real_4 = real_3 + self.linear2(real_3) - self.linear21(imag_3)
        # imag_4 = imag_3 + self.linear21(real_3) + self.linear2(imag_3)

        # real_5 = real_4 + self.linear_ffn(real_4) - self.linear_ffn1(imag_4)
        # imag_5 = imag_4 + self.linear_ffn1(real_4) + self.linear_ffn(imag_4)

        # real = real_5.reshape(batch, self.d_model, -1)[:, :, :channel]
        # imag = imag_5.reshape(batch, self.d_model, -1)[:, :, :channel]
            
        # 把linear放在freq conv之外

        real = torch.cat((real, real[:,:,:self.channel_patch*self.channel_patch2-channel]),dim=-1)
        real = real.reshape(batch, self.d_model, self.channel_patch, self.channel_patch2)
        real = real + self.conv(real)
        real = real + self.conv_ffn(real)  
        real = real + self.linear1(real) 
        real = real.permute(0,1,3,2)
        real = real + self.linear2(real) 
        real = real.permute(0,1,3,2)
        real = real + self.linear_ffn(real)
        real = real.reshape(batch, self.d_model, -1)[:, :, :channel]

        imag = torch.cat((imag, imag[:,:,:self.channel_patch*self.channel_patch2-channel]),dim=-1)
        imag = imag.reshape(batch, self.d_model, self.channel_patch, self.channel_patch2)
        imag = imag + self.conv1(imag)
        imag = imag + self.conv_ffn1(imag)  
        imag = imag + self.linear11(imag) 
        imag = imag.permute(0,1,3,2)
        imag = imag + self.linear21(imag) 
        imag = imag.permute(0,1,3,2)
        imag = imag + self.linear_ffn1(imag)
        imag = imag.reshape(batch, self.d_model, -1)[:, :, :channel]

        x = torch.complex(real, imag)
        x = torch.fft.ifft(x, dim=-2).to(torch.float32)

        return x



class ResBlock(nn.Module):
    def __init__(self, configs, seq_len=96, groups=1):
        super(ResBlock, self).__init__()
        self.d_model = configs.d_model
        # if configs.freq == 't':
        #     self.channel_patch = int(math.sqrt(configs.enc_in+5))+1
        #     self.channel_patch2 = self.channel_patch
        # elif configs.freq == 'h':
        #     self.channel_patch = int(math.sqrt(configs.enc_in+4))+1
        #     self.channel_patch2 = self.channel_patch
        # else:
        #     self.channel_patch = int(math.sqrt(configs.enc_in+3))+1
        #     self.channel_patch2 = self.channel_patch

        self.channel_patch = int(math.sqrt(configs.enc_in))+1
        self.channel_patch2 = self.channel_patch

        self.d_core = 32
        self.num_kernel = 5

        # MLP
        self.linear1 = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Linear(self.channel_patch2, self.d_core),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch2),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.channel_patch2, self.d_core),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch2),
        )
        self.linear2 = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Linear(self.channel_patch, self.d_core),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.channel_patch, self.d_core),      
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Linear(self.d_core, self.channel_patch),    
        )
        self.linear_ffn = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv2d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
        )
        # self.linear_ffn = nn.Sequential(
        #     torch.nn.BatchNorm2d(self.d_model),
        #     FFN_MOE(configs),
        # )


        self.conv = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            Inception_Block_V1(configs.d_model, configs.d_model, num_kernels=self.num_kernel, groups=configs.d_model),
        )
        self.conv_ffn = nn.Sequential(
            torch.nn.BatchNorm2d(self.d_model),
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv2d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=(1,1),stride=(1,1),padding=(0,0),groups=1),
        )
        # self.conv_ffn = nn.Sequential(
        #     torch.nn.BatchNorm2d(self.d_model),
        #     FFN_MOE(configs),
        # )

        # # if self.channel_function == 'RNN':
        # self.norm = torch.nn.LayerNorm(self.d_model)
        # self.linear1 = nn.Sequential(
        #     torch.nn.SiLU(),
        #     torch.nn.Dropout(0.1),
        # )
        # self.lstm = torch.nn.LSTM(input_size=self.d_model,hidden_size=self.d_model,
        #                         num_layers=1,batch_first=True, bidirectional=True)
        # self.pro = nn.Sequential( 
        #     torch.nn.Linear(self.d_model*2, configs.d_model),
        #     nn.SiLU(),
        #     nn.Dropout(0.1), 
        # )
        # self.norm = torch.nn.LayerNorm(self.d_model)
        # self.rnn_ffn = nn.Sequential(
        #     # torch.nn.LayerNorm(self.d_model),
        #     torch.nn.Conv1d(in_channels=self.d_model, out_channels=configs.d_ff,
        #             kernel_size=1,stride=1,padding=0,groups=1),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(configs.dropout),
        #     torch.nn.Conv1d(in_channels=configs.d_ff, out_channels=self.d_model,
        #             kernel_size=1,stride=1,padding=0,groups=1),
        # )


    def forward(self, x):
        batch, seq, channel = x.shape

        # x = x.permute(0,2,1)
        # h0 = torch.randn(2, batch, self.d_model, device=x.device)
        # c0 = torch.randn(2, batch, self.d_model, device=x.device)
        # x = x + torch.mul(self.linear1(x), self.pro(self.lstm(self.norm(x), (h0,c0))[0]))
        # x = x + self.rnn_ffn(self.norm(x).permute(0,2,1)).permute(0,2,1)
        # x = x.permute(0,2,1)
        # print('wang')

        # temporal 和 channel谁在前和谁在后没有区别

        x = torch.cat((x, x[:,:,:self.channel_patch*self.channel_patch2-channel]),dim=-1)
        x = x.reshape(batch, self.d_model, self.channel_patch, self.channel_patch2)
 
        x = x + self.conv(x)
        x = x + self.conv_ffn(x)  

        x = x + self.linear1(x) 
        x = x.permute(0,1,3,2)
        x = x + self.linear2(x) 
        x = x.permute(0,1,3,2)
        x = x + self.linear_ffn(x)
        
        x = x.reshape(batch, self.d_model, -1)[:, :, :channel]


        return x



class ResBlock_RNN(nn.Module):
    def __init__(self, configs, seq_len=96, groups=1):
        super(ResBlock_RNN, self).__init__()
        self.d_model = configs.d_model
        # if configs.freq == 't':
        #     self.channel_patch = int(math.sqrt(configs.enc_in+5))+1
        #     self.channel_patch2 = configs.enc_in // self.channel_patch + 1
        # elif configs.freq == 'h':
        #     self.channel_patch = int(math.sqrt(configs.enc_in+4))+1
        #     self.channel_patch2 = configs.enc_in // self.channel_patch + 1
        # else:
        #     self.channel_patch = int(math.sqrt(configs.enc_in+3))+1
        #     self.channel_patch2 = configs.enc_in // self.channel_patch + 1

        self.norm = torch.nn.LayerNorm(self.d_model)
        self.linear1 = nn.Sequential(
            torch.nn.Linear(self.d_model, configs.d_model),
            torch.nn.SiLU(),
            torch.nn.Dropout(configs.dropout),
        )
        self.lstm = torch.nn.GRU(input_size=self.d_model,hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.pro = nn.Sequential( 
            torch.nn.Linear(self.d_model*2, configs.d_model),
            nn.SiLU(),
            nn.Dropout(configs.dropout), 
        )
        self.ffn_norm1 = torch.nn.LayerNorm(self.d_model)
        self.rnn_ffn1 = nn.Sequential(
            # torch.nn.LayerNorm(self.d_model),
            torch.nn.Conv1d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=1,stride=1,padding=0,groups=1),
            torch.nn.SiLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv1d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0,groups=1),
        )        
        # self.ffn_norm2 = torch.nn.LayerNorm(self.d_model)
        # self.rnn_ffn2 = nn.Sequential(
        #     torch.nn.Conv1d(in_channels=self.d_model, out_channels=configs.d_ff,
        #             kernel_size=1,stride=1,padding=0,groups=8),
        #     torch.nn.SiLU(),
        #     torch.nn.Dropout(configs.dropout),
        #     torch.nn.Conv1d(in_channels=configs.d_ff, out_channels=self.d_model,
        #             kernel_size=1,stride=1,padding=0,groups=8),
        # )
        self.l = torch.nn.Conv1d(in_channels=self.d_model*2, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0,groups=1)

    def forward(self, x):
        batch, seq, channel = x.shape

        x = x.permute(0,2,1)
        # h0 = torch.randn(2, batch, self.d_model, device=x.device)
        # c0 = torch.randn(2, batch, self.d_model, device=x.device)
        x1, x2 = self.lstm(self.norm(x))

        x2 = torch.sum(x2.permute(1,0,2), dim=1, keepdim=True)
        x2 = x2.repeat(1, channel, 1).permute(0,2,1)
        x = torch.cat((x.permute(0,2,1), x2), dim=1)
        x = self.l(x)


        # x1 = x1[:,:,:self.d_model] + x1[:,:,-self.d_model:]
        # x = x + torch.mul(self.linear1(x), (self.pro(x1)))
        # x = x + self.rnn_ffn1(self.ffn_norm1(x).permute(0,2,1)).permute(0,2,1)
        # x = self.norm1(x)
        # x = x.permute(0,2,1)


        return x


# patch 内部再套patch好像不行啊
class ResBlock_RNN3(nn.Module):
    def __init__(self, configs, seq_len=96, groups=1):
        super(ResBlock_RNN3, self).__init__()
        self.d_model = configs.d_model
        if configs.freq == 't':
            self.channel_patch = int(math.sqrt(configs.enc_in+5)) + 1
            self.channel_patch2 = (configs.enc_in+5) // self.channel_patch + 1
        elif configs.freq == 'h':
            self.channel_patch = int(math.sqrt(configs.enc_in+4)) + 1
            # self.channel_patch = 50
            self.channel_patch2 = (configs.enc_in+4) // self.channel_patch + 1
        else:
            self.channel_patch = int(math.sqrt(configs.enc_in)) + 1
            self.channel_patch2 = configs.enc_in // self.channel_patch + 1


        self.linear = nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model),
            torch.nn.SiLU(),
            torch.nn.Dropout(configs.dropout),
        )

        self.norm_lstm = torch.nn.LayerNorm(self.d_model)
        self.lstm = torch.nn.GRU(input_size=self.d_model,hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.lstm_linear = nn.Sequential( 
            torch.nn.Linear(self.d_model*2, self.d_model),
            nn.SiLU(),
            nn.Dropout(configs.dropout), 
        )
        self.lstm2 = torch.nn.GRU(input_size=self.d_model,hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.lstm_linear2 = nn.Sequential( 
            torch.nn.Linear(self.d_model*2, self.d_model),
            nn.SiLU(),
            nn.Dropout(configs.dropout), 
        )
        self.lstm3 = torch.nn.GRU(input_size=self.d_model,hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.lstm_linear3 = nn.Sequential( 
            torch.nn.Linear(self.d_model*2, self.d_model),
            nn.SiLU(),
            nn.Dropout(configs.dropout), 
        )
        self.ffn_norm = torch.nn.LayerNorm(self.d_model)
        self.rnn_ffn = nn.Sequential(
            torch.nn.Conv1d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=1,stride=1,padding=0,groups=1),
            torch.nn.SiLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv1d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0,groups=1),
        )

        self.c1 = int(math.sqrt(self.channel_patch)) + 1
        self.c2 = self.channel_patch // self.c1 + 1

    def forward(self, x):
        batch, seq, channel = x.shape

        x = torch.cat((x, x[:,:,:(self.channel_patch*self.channel_patch2-channel)]), dim=-1)
        kv = x = x.reshape(batch, self.d_model, self.channel_patch, self.channel_patch2)
        x = x.reshape(batch*self.channel_patch2, self.d_model, self.channel_patch) 

        x = torch.cat((x, x[:,:,:(self.c1*self.c1-self.channel_patch)]), dim=-1)
        x = x.reshape(batch*self.channel_patch2, self.d_model, self.c1, self.c2)
        x = x.reshape(batch*self.channel_patch2*self.c2, self.d_model, self.c1).permute(0,2,1)

        x1, x2 = self.lstm(self.norm_lstm(x))
        x1 = self.lstm_linear(x1).permute(0,2,1).reshape(batch*self.channel_patch2, self.d_model, self.c1, self.c2)
        x1 = x1.reshape(batch, self.d_model, self.c1, self.c2, self.channel_patch2) 
        x2 = torch.sum(x2.permute(1,0,2), dim=1, keepdim=True).reshape(batch*self.channel_patch2, self.c2, self.d_model)
        x21, x22 = self.lstm2((x2))
        x21 = (self.lstm_linear2(x21)).permute(0,2,1).reshape(batch, self.d_model, self.c2, self.channel_patch2)
        x22 = torch.sum(x22.permute(1,0,2), dim=1, keepdim=True).reshape(batch, self.channel_patch2, self.d_model)
        x221, x222 = self.lstm3((x22))
        x221 = (self.lstm_linear3(x221)).permute(0,2,1).unsqueeze(-2)
        x21 = (x21 + torch.mul(x21, x221)).unsqueeze(-3)
        x1 = x1 + torch.mul(x1, x21)
        x1 = x1.reshape(batch, self.d_model, self.c1*self.c2, self.channel_patch2)[:,:,:self.channel_patch, :]
        # x2 = x2.reshape(batch*self.channel_patch2, self.d_model, self.c1*self.c2)[:,:,self.channel_patch]
        x = (kv) + torch.mul(self.linear(kv.permute(0,2,3,1)).permute(0,3,1,2), x1) 
        x = x.reshape(batch, self.d_model, -1)[:,:,:channel].permute(0,2,1)

        x = x + self.rnn_ffn(self.ffn_norm(x).permute(0,2,1)).permute(0,2,1)  
        x = x.permute(0,2,1)

        return x





class ResBlock_RNN2(nn.Module):
    def __init__(self, configs, seq_len=96, groups=1):
        super(ResBlock_RNN2, self).__init__()
        self.d_model = configs.d_model

        self.n_head = 1
        self.head = self.d_model // self.n_head
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


        # self.n_patch = int(math.sqrt(configs.enc_in)) 
        # self.n_patch = 5
        # self.c_patch = configs.enc_in // self.n_patch + 1


        self.linear = nn.Sequential(
            torch.nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0),
            torch.nn.SiLU(),
            torch.nn.Dropout(configs.dropout),
        )

        self.norm_lstm = torch.nn.LayerNorm(self.d_model)
        # self.norm_lstm = torch.nn.BatchNorm2d(self.d_model)
        self.lstm = torch.nn.GRU(input_size=self.d_model,hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.lstm_linear = nn.Sequential( 
            # torch.nn.Conv1d(in_channels=self.d_model*2, out_channels=self.d_model,
            #         kernel_size=1,stride=1,padding=0),
            nn.SiLU(),
            nn.Dropout(configs.dropout), 
        )

        self.lstm2 = torch.nn.GRU(input_size=self.d_model, hidden_size=self.d_model,
                                num_layers=1,batch_first=True, bidirectional=True)
        self.lstm_linear2 = nn.Sequential( 
            # torch.nn.Conv1d(in_channels=self.d_model*2, out_channels=self.d_model,
            #         kernel_size=1,stride=1,padding=0),
            nn.SiLU(),
            nn.Dropout(configs.dropout), 
        )

        # self.l = torch.nn.Conv1d(in_channels=self.d_model*2, out_channels=self.d_model,
        #             kernel_size=1,stride=1,padding=0)


        self.ffn_norm = torch.nn.LayerNorm(self.d_model)
        self.rnn_ffn = nn.Sequential(
            torch.nn.Conv1d(in_channels=self.d_model, out_channels=configs.d_ff,
                    kernel_size=1,stride=1,padding=0,groups=1),
            torch.nn.SiLU(),
            torch.nn.Dropout(configs.dropout),
            torch.nn.Conv1d(in_channels=configs.d_ff, out_channels=self.d_model,
                    kernel_size=1,stride=1,padding=0,groups=1),
        )

        # self.rnn_norm = torch.nn.LayerNorm(self.d_model//8)
        # self.rnn = torch.nn.GRU(input_size=self.d_model//8, hidden_size=self.d_model,
        #                         num_layers=1,batch_first=True, bidirectional=False)
        # self.rnn_linear = nn.Sequential( 
        #     torch.nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model//8,
        #             kernel_size=1,stride=1,padding=0),
        #     nn.SiLU(),
        #     nn.Dropout(configs.dropout), 
        # )
        # self.rnn_linear2 = nn.Sequential( 
        #     torch.nn.Conv1d(in_channels=self.d_model//8, out_channels=self.d_model//8,
        #             kernel_size=1,stride=1,padding=0),
        #     nn.SiLU(),
        #     nn.Dropout(configs.dropout), 
        # )
        # self.mamba = Mamba(
        #             d_model = configs.d_model,
        #             d_state = 16,
        #             d_conv = 2,
        #             expand = 1,
        #         )
        # self.mamba_norm = torch.nn.LayerNorm(self.d_model)
        # self.mamba2 = Mamba(
        #             d_model = configs.d_model,
        #             d_state = 16,
        #             d_conv = 2,
        #             expand = 1,
        #         )


    def forward(self, x):
        batch, seq, channel = x.shape

        x = torch.cat((x, x[:,:,:(self.c_patch*self.n_patch-channel)]), dim=-1)
        x = x.reshape(batch, self.d_model, self.c_patch, self.n_patch)
        kv = x
        x = x.reshape(batch*self.n_patch, self.d_model, self.c_patch).permute(0,2,1)

        x1, x2 = self.lstm(self.norm_lstm(x))
        x1 = x1[:,:,:self.d_model] + x1[:,:,-self.d_model:]
        x1 = self.lstm_linear(x1.permute(0,2,1))
        x1 = x1.reshape(batch, self.d_model, self.c_patch, self.n_patch)
        x2 = torch.sum(x2.permute(1,0,2), dim=1, keepdim=True).reshape(batch, self.n_patch, self.d_model)
        x21, x22 = self.lstm2((x2)) 
        # x22 = torch.sum(x22.permute(1,0,2), dim=1, keepdim=True)

        x21 = x21[:,:,:self.d_model] + x21[:,:,-self.d_model:]
        x21 = self.lstm_linear2(x21.permute(0,2,1)).unsqueeze(-2)
        x = kv + torch.mul(self.linear(kv), (torch.mul(x21, x1) )) 
        x = x.reshape(batch, self.d_model, self.c_patch*self.n_patch).contiguous()[:,:,:channel]


        # x = x.permute(0,2,1)
        # x = self.mamba_norm(x)
        # x = self.mamba(x) + self.mamba2(x.flip(dims=[1])).flip(dims=[1])
        # x = x.permute(0,2,1)

        x = x + self.rnn_ffn(self.ffn_norm(x.permute(0,2,1)).permute(0,2,1)) 

        # x1 = x.reshape(batch*channel, 8, 64)
        # x = x1 + torch.mul(x1, self.rnn_linear(self.rnn(self.rnn_norm(x1))[0].permute(0,2,1)).permute(0,2,1))
        # x = x.reshape(batch, self.d_model, channel)
        # # x = x + self.ffn_norm(x.permute(0,2,1)).permute(0,2,1)

        return x



class Embedding(nn.Module):
    def __init__(self, configs):
        super(Embedding, self).__init__()
        if configs.freq == 't':
            self.enc_in = configs.enc_in+5
        elif configs.freq == 'h':
            self.enc_in = configs.enc_in+4
        elif configs.freq == 'd':
            self.enc_in = configs.enc_in+3
        else:
            self.enc_in = configs.enc_in
        # self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model


        self.temporal_function = 'patch'
        if self.temporal_function == 'down':
            self.kernel = configs.patch
            self.layers = len(self.kernel)
            # self.layernorm = torch.nn.LayerNorm(self.enc_in)
            self.down_sample = torch.nn.ModuleList([torch.nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                    kernel_size=self.kernel[i],stride=self.kernel[i],padding=0, groups=self.enc_in)
                    for i in range(self.layers)])
            self.temporal_down = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.seq_len//self.kernel[i], self.enc_in]),
                # torch.nn.LayerNorm(self.enc_in),
                torch.nn.Conv1d(in_channels=self.seq_len//self.kernel[i], out_channels=configs.d_model,
                    kernel_size=1,stride=1,padding=0),
                # nn.Linear(self.seq_len//self.kernel[i], configs.d_model ),
                nn.SiLU(),
                nn.Dropout(configs.dropout),
                torch.nn.Conv1d(in_channels=configs.d_model, out_channels=self.seq_len,
                    kernel_size=1,stride=1,padding=0),
                # nn.Linear(configs.d_model, self.seq_len),
                # nn.Dropout(configs.dropout),
            ) for i in range(self.layers)])
            # self.linear_down = nn.Linear(self.seq_len, configs.d_model) 
                # for i in range(self.layers)])
            self.linear_down = torch.nn.Conv1d(in_channels=self.seq_len, out_channels=configs.d_model,
                    kernel_size=1,stride=1,padding=0)


        if self.temporal_function == 'patch':
            self.temporal_patch = nn.Sequential(
                RMSNorm([self.enc_in,self.seq_len]),
                nn.Linear(self.seq_len, configs.d_model),
                nn.SiLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, self.d_model),
                nn.Dropout(configs.dropout)
            )
            self.patch = configs.patch
            self.patch_num = [self.seq_len // i for i in self.patch]
            self.decomp = torch.nn.ModuleList([series_decomp(i+1) for i in self.patch])
            self.temporal1 = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch_num[i],self.patch[i]]),
                nn.Linear(self.patch[i], self.patch[i]*4),
                nn.SiLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch[i]),
                nn.Dropout(configs.dropout)
            ) for i in range(len(self.patch))])
            self.temporal2 = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch[i],self.patch_num[i]]),
                nn.Linear(self.patch_num[i], self.patch[i]*4),
                nn.SiLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch_num[i]),
                nn.Dropout(configs.dropout)
            )  for i in range(len(self.patch))])
            self.temporal1_season = torch.nn.ModuleList([nn.Sequential(
                RMSNorm([self.patch_num[i],self.patch[i]]),
                nn.Linear(self.patch[i], self.patch[i]*4),
                nn.SiLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch[i]),
                nn.Dropout(configs.dropout)
            ) for i in range(len(self.patch))])
            self.temporal2_season = torch.nn.ModuleList([nn.Sequential(
                # RMSNorm([self.patch[i],self.patch_num[i]]),
                nn.Linear(self.patch_num[i], self.patch[i]*4),
                nn.SiLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch_num[i]),
                nn.Dropout(configs.dropout)
            )  for i in range(len(self.patch))])
            self.linear_patch = torch.nn.ModuleList([nn.Linear(self.seq_len, self.d_model) 
                    for i in range(len(self.patch))])


    def forward(self, x):
        B, L, D = x.shape
        if self.temporal_function == 'patch':
            add = torch.zeros([B, D, self.d_model], device=x.device)
            for i in range(len(self.patch)):
                if self.patch[i] == 1:
                    add = add +  self.temporal_patch((x).transpose(1, 2)) 
                else:
                    season, x_group = self.decomp[i](x)
                    x_group = x
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
                    add = add + self.linear_patch[i]((x_group + season).permute(0,2,1))  
            x = add.permute(0,2,1)

        if self.temporal_function == 'down':
            # x = x.permute(0,2,1)
            # add = torch.zeros([B, D, self.d_model], device=x.device)
            # x1 = self.layernorm(x)
            for i in range(self.layers):
                # tmp = torch.nn.AvgPool1d(kernel_size=self.kernel[i])(x) 
                tmp = self.down_sample[i](x.permute(0,2,1)).permute(0,2,1)
                tmp = self.temporal_down[i](tmp)
                x = x + tmp
            x = self.linear_down(x)
                # add = add + tmp
            # x = x.permute(0,2,1)
        
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.use_norm = configs.use_norm
        self.freq = configs.freq
        self.d_model = configs.d_model


        # self.linear_trans = torch.nn.Conv1d(in_channels=configs.seq_len, out_channels=configs.d_model,
        #             kernel_size=1, stride=1, groups=1)

        self.emb_layer = 1
        self.emb = nn.ModuleList([Embedding(configs)
                                    for i in range(self.emb_layer)])
        self.model = nn.ModuleList([ResBlock_RNN2(configs, seq_len=self.seq_len, groups=configs.d_model // int(math.pow(2,i)))
                                    for i in range(configs.e_layers)])

        self.projection = nn.Linear(configs.d_model, configs.pred_len)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch, seq, channel = x_enc.shape

        # print(x_mark_enc)

        if self.freq == 't' or self.freq == 'h' or self.freq == 'd':
            x_enc = torch.cat((x_enc, x_mark_enc), dim=-1)

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # if self.freq == 't' or self.freq == 'h' or self.freq == 'd':
        #     x_enc = torch.cat((x_enc, x_mark_enc), dim=-1)

        # x_enc = self.linear_trans(x_enc)
        for i in range(self.emb_layer):
            x_enc = self.emb[i](x_enc)
        
        # x_enc = self.linear_trans(x_enc)

        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection((x_enc).transpose(1, 2)).transpose(1, 2)
        
        if self.use_norm:
            enc_out = enc_out * stdev + means

        return enc_out[:,:,:self.enc_in], x_enc

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        _, L, N = x_enc.shape

        # x = x_enc
        # for i in range(len(self.group)):
        #     # x_enc = x_enc + torch.mul(self.emb[i](x_enc), self.sigmoid(self.emb2[i](x_enc)))
        #     x_enc = x_enc + self.emb[i](x_enc)  

        x_enc = self.linear_trans(x_enc)  
        
        
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        # enc_out = x_enc
        enc_out = self.projection((x_enc ).transpose(1, 2)).transpose(1, 2)
        
        enc_out = enc_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        enc_out = enc_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        return enc_out


    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, x_enc = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]   # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

