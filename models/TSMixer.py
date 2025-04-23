import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi

from layers.SelfAttention_Family import AttentionLayer, ProbAttention, FullAttention
import torch.nn.init as init
# from models.Revin import RevIN

from mambapy.mamba import Mamba, MambaConfig

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

def ledoit_wolf_shrinkage(cov_matrix, shrinkage_target=None):
    """
    Perform Ledoit-Wolf shrinkage on a covariance matrix.
    
    Args:
        cov_matrix (torch.Tensor): Sample covariance matrix of shape (n, n).
        shrinkage_target (torch.Tensor, optional): Target covariance matrix, usually the identity matrix
            or a diagonal matrix with the average variance.
    
    Returns:
        torch.Tensor: Shrinked covariance matrix.
    """
    n = cov_matrix.shape[0]

    # Calculate the empirical covariance and its trace
    trace_c = torch.trace(cov_matrix)
    covariance_trace = trace_c / n
    
    # If no shrinkage target is provided, use the identity matrix
    if shrinkage_target is None:
        shrinkage_target = torch.eye(n).cuda() * covariance_trace
    
    # Calculate the shrinkage coefficient
    diff = cov_matrix - shrinkage_target
    numerator = torch.norm(diff, p='fro')**2
    denominator = (torch.norm(cov_matrix, p='fro')**2 + torch.norm(shrinkage_target, p='fro')**2)
    
    # Calculate the shrinkage intensity
    shrinkage_intensity = numerator / denominator
    shrinkage_intensity = min(shrinkage_intensity, 1.0)  # Ensure it does not exceed 1.0

    # Perform the shrinkage
    shrinked_cov = (1 - shrinkage_intensity) * cov_matrix + shrinkage_intensity * shrinkage_target
    return shrinked_cov






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


class ResBlock(nn.Module):
    def __init__(self, configs, seq_len=96):
        super(ResBlock, self).__init__()

        self.channel_patch = int(math.sqrt(configs.enc_in))+1
        self.channel_num = self.channel_patch
        self.enc_in = configs.enc_in
        self.seq_len = seq_len
        self.d_model = configs.d_model

        self.channel_function = 'no'
        # self.channel_function = 'MLP'
        # self.channel_function = 'MLP_group'
        self.channel_function = 'RNN'
        # self.channel_function = 'Mamba'
        # self.channel_function = 'attention'

        self.temporal_function = 'normal'
        # self.temporal_function = 'temporal_group'

        if self.temporal_function == 'temporal_group':
            self.patch = [1,4,12,24]
            self.patch_num = [self.seq_len // i for i in self.patch]
            self.decomp = torch.nn.ModuleList([series_decomp(i+1) for i in self.patch])
            self.moe = MoE(self.seq_len, len(self.patch), 0.1)
            # self.w = torch.ones(len(self.patch), requires_grad=True)
            self.temporal1 = torch.nn.ModuleList([nn.Sequential(
                nn.Linear(self.patch[i], self.patch[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch[i]),
                nn.Dropout(configs.dropout)
            ) for i in range(len(self.patch))])
            self.channel_bn = torch.nn.ModuleList([torch.nn.BatchNorm2d(configs.enc_in) for i in range(len(self.patch))])
            self.channel_bn2 = torch.nn.ModuleList([torch.nn.BatchNorm2d(configs.enc_in) for i in range(len(self.patch))])
            self.temporal2 = torch.nn.ModuleList([nn.Sequential(
                nn.Linear(self.patch_num[i], self.patch_num[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch_num[i]*4, self.patch_num[i]),
                nn.Dropout(configs.dropout)
            )  for i in range(len(self.patch))])


            self.temporal1_season = torch.nn.ModuleList([nn.Sequential(
                nn.Linear(self.patch[i], self.patch[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch[i]*4, self.patch[i]),
                nn.Dropout(configs.dropout)
            ) for i in range(len(self.patch))])
            self.channel_bn_season = torch.nn.ModuleList([torch.nn.BatchNorm2d(configs.enc_in) for i in range(len(self.patch))])
            self.channel_bn2_season = torch.nn.ModuleList([torch.nn.BatchNorm2d(configs.enc_in) for i in range(len(self.patch))])
            self.temporal2_season = torch.nn.ModuleList([nn.Sequential(
                nn.Linear(self.patch_num[i], self.patch_num[i]*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.patch_num[i]*4, self.patch_num[i]),
                nn.Dropout(configs.dropout)
            )  for i in range(len(self.patch))])
            self.linear = torch.nn.ModuleList([torch.nn.Linear(self.seq_len, self.seq_len) for i in range(len(self.patch))])

        if self.temporal_function == 'normal':
            self.temporal = nn.Sequential(
                RMSNorm([self.enc_in,self.seq_len]),
                nn.Linear(self.seq_len, configs.d_model),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model, self.seq_len),
                # nn.ReLU(),
                # nn.Dropout(configs.dropout),
                # nn.Linear(self.seq_len, configs.d_model),
                # nn.ReLU(),
                # nn.Dropout(configs.dropout),
                # nn.Linear(configs.d_model, self.seq_len),
                nn.Dropout(configs.dropout)
            )


        if self.channel_function == 'Mamba':
            self.norm = RMSNorm([self.enc_in,self.seq_len])
            self.config = MambaConfig(d_model=self.seq_len, n_layers=1)
            self.mamba = Mamba(self.config)
            self.norm2 = RMSNorm([self.enc_in,self.seq_len])
            self.mamba2 = Mamba(self.config)
            self.linear = torch.nn.Linear(self.seq_len, self.seq_len)
        if self.channel_function == 'MLP_group':
            self.channel1 = nn.Sequential(
                nn.Linear(self.channel_patch, self.channel_patch*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.channel_patch*4, self.channel_patch),
                nn.Dropout(configs.dropout)
            )
            # self.silu = torch.nn.ReLU()
            # self.sigmoid = torch.nn.Sigmoid()
            self.seq_bn = torch.nn.BatchNorm2d(configs.seq_len)
            self.seq_bn2 = torch.nn.BatchNorm2d(configs.seq_len)
            # self.seq_bn = RMSNorm([self.channel_patch, self.channel_num])
            # self.seq_bn2 = RMSNorm([self.channel_num, self.channel_patch])
            self.channel2 = nn.Sequential(
                nn.Linear(self.channel_num, self.channel_num*4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.channel_num*4, self.channel_num),
                nn.Dropout(configs.dropout)
            ) 
        if self.channel_function == 'RNN':
            self.norm = RMSNorm([self.enc_in,self.seq_len])
            self.linear1 = nn.Sequential(
                # torch.nn.Linear(self.seq_len, self.seq_len),
                torch.nn.SiLU(),
                torch.nn.Dropout(configs.d2),
            )
            self.lstm = torch.nn.LSTM(input_size=self.seq_len,
                                    hidden_size=self.d_model ,
                                    num_layers=1,  
                                    batch_first=True, bidirectional=True)
            self.pro = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(configs.d2),
                torch.nn.Linear(self.d_model*2, configs.seq_len),
                nn.SiLU(),
                nn.Dropout(configs.d2),               
                # torch.nn.Sigmoid(),
                # torch.nn.Softmax(dim=-1),  # softmax -1 好像和sigmoid差不多
            )

        if self.channel_function == 'MLP':
            self.final_linear = nn.Sequential(
                RMSNorm([self.seq_len,self.enc_in]),
                nn.Linear(self.enc_in , self.enc_in//4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.enc_in//4, self.enc_in//1),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.enc_in//1, self.enc_in//4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.enc_in//4, self.enc_in//1),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.enc_in//1, self.enc_in//4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.enc_in//4, self.enc_in//1),
                # nn.ReLU(),
                # nn.Dropout(configs.dropout),
                # nn.Linear(self.enc_in//1, self.enc_in//4),
                # nn.ReLU(),
                # nn.Dropout(configs.dropout),
                # nn.Linear(self.enc_in//4, self.enc_in),
                nn.Dropout(configs.dropout),
            )  
        if self.channel_function == 'attention':
            self.attn = AttentionLayer(
                FullAttention(mask_flag=False,attention_dropout=0.1),
                d_model=configs.seq_len,
                n_heads=configs.n_heads)

            self.norm = torch.nn.LayerNorm(self.seq_len)


    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape

        if self.temporal_function == 'temporal_group':
            # result = torch.zeros([B, L, D], device=x.device)
            weight = self.moe(x.permute(0,2,1)) 
            result = []
            for i in range(len(self.patch)):
                if self.patch[i] != 1:
                    season, x_group = self.decomp[i](x)
                else:
                    season = torch.zeros([B, self.seq_len, D], device=x.device)
                    x_group = x
                # season, x_group = self.decomp[i](x)
                x_group = x_group.permute(0,2,1)
                x_group = x_group.reshape(B, D, self.patch_num[i], self.patch[i])
                x_group = x_group + self.temporal1[i](self.channel_bn[i](x_group))
                # x_group =  self.channel_bn[i](x_group)
                x_group = x_group.permute(0,1,3,2)
                x_group = x_group + self.temporal2[i](self.channel_bn2[i](x_group))
                # x_group =  self.channel_bn2[i](x_group)
                x_group = x_group.permute(0,1,3,2)
                x_group = x_group.reshape(B, D, -1) 
                x_group = x_group.permute(0,2,1)
                
                season = season.permute(0,2,1)
                season = season.reshape(B, D, self.patch_num[i], self.patch[i])
                season = season + self.temporal1_season[i](self.channel_bn_season[i](season))
                # x_group =  self.channel_bn[i](x_group)
                season = season.permute(0,1,3,2)
                season = season + self.temporal2_season[i](self.channel_bn2_season[i](season))
                # x_group =  self.channel_bn2[i](x_group)
                season = season.permute(0,1,3,2)
                season = season.reshape(B, D, -1) 
                season = season.permute(0,2,1)

                # result = result + x_group + season
                # result.append(self.linear[i]((x_group).permute(0,2,1)))
                result.append(self.linear[i]((x_group + season).permute(0,2,1)))
            # x = result
            x =  (torch.matmul(torch.stack(result, dim=-1),weight.unsqueeze(-1)) ).squeeze(-1).permute(0,2,1)
        
        if self.temporal_function == 'normal':
            x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)

        if self.channel_function == 'MLP':
            x = x + self.final_linear(x)
        if self.channel_function == 'MLP_group':
            x = torch.cat((x, x[:,:,:(self.channel_patch*self.channel_num-self.enc_in)]), dim=-1)
            x = x.reshape(B, L, self.channel_patch, self.channel_num)
            x = x + self.channel1(self.seq_bn(x))
            # x = self.seq_bn(x)
            x = x.permute(0,1,3,2)
            x = x + self.channel2(self.seq_bn2(x))
            # x = self.seq_bn2(x)
            x = x.permute(0,1,3,2)
            x = x.reshape(B, L, -1)[:,:,:self.enc_in]
        if self.channel_function == 'RNN':
            x = x.permute(0,2,1)
            # h0 = torch.zeros(2, B, self.seq_len//2, device=x.device)  # 隐藏状态
            # c0 = torch.zeros(2, B, self.seq_len//2, device=x.device)  # 细胞状态
            h0 = torch.randn(2, B, self.d_model , device=x.device)  # 随机隐藏状态
            c0 = torch.randn(2, B, self.d_model , device=x.device)  # 随机细胞状态           
            # h0 = x.mean(dim=1).unsqueeze(0).repeat(2,1,1)  # 随机隐藏状态
            # c0 = x.mean(dim=1).unsqueeze(0).repeat(2,1,1)  # 随机细胞状态 
            x = x + torch.mul(self.linear1(x), self.pro(self.lstm(self.norm(x), (h0,c0))[0]))
            x = x.permute(0,2,1)
        if self.channel_function == 'Mamba':
            x = x.permute(0,2,1)
            # x1 = torch.flip(x, dim=1)
            # x = self.linear(self.mamba(x))
            x = x + self.linear(self.mamba(self.norm(x)) + torch.flip(self.mamba2(self.norm2(torch.flip(x, dims=[1]))), dims=[1]))
            # x = self.linear(x)
            x = x.permute(0,2,1)
        if self.channel_function == 'attention':
            x = x.permute(0,2,1)
            x = x + self.attn(x, x, x, attn_mask=None)[0]
            x = self.norm(x)
            x = x.permute(0,2,1)
 
        # x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        # self.decomp = series_decomp(13)
        # self.decomp3 = series_decomp(25)
        self.model = nn.ModuleList([ResBlock(configs, seq_len=self.seq_len)
                                    for _ in range(configs.e_layers)])
        # self.model_trend = nn.ModuleList([ResBlock(configs, seq_len=self.seq_len)
        #                             for _ in range(configs.e_layers)])
        # self.model1 = nn.ModuleList([ResBlock(configs, seq_len=self.seq_len//4)
        #                             for _ in range(configs.e_layers)])
        # self.model2 = nn.ModuleList([ResBlock(configs, seq_len=self.seq_len//12)
        #                             for _ in range(configs.e_layers)])
        # self.model3 = nn.ModuleList([ResBlock(configs, seq_len=self.seq_len//24 )
        #                             for _ in range(configs.e_layers)])                                                                        
        # self.model3_trend = nn.ModuleList([ResBlock(configs, seq_len=self.seq_len//24 )
        #                             for _ in range(configs.e_layers)]) 
        # self.model2 = nn.ModuleList([ResBlock(configs)
        #                             for _ in range(configs.e_layers)])
        # self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        # self.projection1 = nn.Linear(configs.seq_len//4, configs.pred_len)
        # self.projection2 = nn.Linear(configs.seq_len//12, configs.pred_len)
        # self.projection3 = nn.Linear(configs.seq_len//24, configs.pred_len)

        # self.patch = [24,12,4]
        # self.decomp = nn.ModuleList([series_decomp(self.patch[i]+1)
        #                 for i in range(configs.e_layers)])
 
        # self.moe = MoE(self.seq_len, (configs.e_layers), 0.1)    
        # self.w = torch.ones(len(self.patch), requires_grad=True) 

        # self.linear = torch.nn.Linear(self.seq_len, configs.d_model)
        # self.linear2 = torch.nn.Linear(self.seq_len, configs.d_model)
        # self.softmax = torch.nn.Softmax(dim=-1)


        # self.lstm = torch.nn.LSTM(input_size=1,
        #                 hidden_size=1,
        #                 num_layers=1, dropout=0.1, 
        #                 batch_first=True, bidirectional=False)
        # self.revin = RevIN(self.enc_in)

        # self.chafen = torch.nn.Linear(self.seq_len+1, self.pred_len)

        # self.norm_method = 'last_value'
        if configs.use_norm:
            self.norm_method = 'revin'
        else:
            self.norm_method = 'no'

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch, seq, channel = x_enc.shape

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


        # x_enc1 = torch.cat((x_enc[:,:1,:], x_enc), dim=1)
        # result = self.chafen(x_enc1.permute(0,2,1)).permute(0,2,1)
        # x_enc = x_enc - x_enc1[:,:seq, :]

        # min1 = (torch.max(x_enc, dim=1, keepdim=True)[0] + torch.min(x_enc, dim=1, keepdim=True)[0])/2
        # x_enc = x_enc - min1

        # min1 = (x_enc[:,-1:,:]   )/1
        # x_enc = x_enc - min1

        # x_enc = self.revin(x_enc, 'norm')
        # x3 = x_enc
        # means2 = x_enc.mean(1, keepdim=True).detach()       
        # x_enc = x_enc - means2
        # stdev2 = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev2

        # x1 = torch.nn.AvgPool1d(kernel_size=4, stride=4, padding=0)(x_enc.permute(0,2,1)).permute(0,2,1)
        # x2 = torch.nn.AvgPool1d(kernel_size=12, stride=12, padding=0)(x_enc.permute(0,2,1)).permute(0,2,1)
        # x3 = torch.nn.AvgPool1d(kernel_size=24, stride=24, padding=0)(x_enc.permute(0,2,1)).permute(0,2,1)


        # # 打乱顺序
        # x_enc = x_enc.permute(0,2,1)
        # indices = torch.randperm(x_enc.size(0))
        # x_enc = x_enc[indices]
        # x_enc = x_enc.permute(0,2,1)

        # x_enc = self.linear(x_enc.permute(0,2,1)).permute(0,2,1)

        # x_enc1 = self.linear(x_enc.permute(0,2,1)).permute(0,2,1)
        # x_enc2 = self.linear2(x_enc.permute(0,2,1)).permute(0,2,1)

        # shrinked_covs = []      
        # for i in range(batch):
        #     sample_cov = torch.cov(x_enc1[i].T)  # 转置以使每列为一个变量
        #     # 使用 Ledoit-Wolf 方法进行收缩
        #     shrinked_cov = ledoit_wolf_shrinkage(sample_cov).to(x_enc.device)
        #     shrinked_covs.append(shrinked_cov)
        # shrinked_covs = torch.stack(shrinked_covs)
        # shrinked_covs = self.softmax(shrinked_covs)

        # x_enc = torch.matmul(shrinked_covs, x_enc2.permute(0,2,1)).permute(0,2,1)

        # weight = self.moe(x_enc.permute(0,2,1)) 
        # result = []
        # x: [B, L, D]
        # x = x_enc
        # enc_out = torch.zero
        # enc_out = torch.zeros([batch, self.pred_len, channel], device=x_enc.device)
        # season, trend = self.decomp(x_enc)
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
            # x_enc = self.revin(x_enc, 'norm')

            # season, trend = self.decomp(x_enc)
            # season = self.model[i](season)
            # trend = self.model_trend[i](trend)
            # x_enc = season + trend
            # x_enc = self.revin(x_enc, 'denorm')

            # season = self.model2[i](season)
            # x_enc = trend + season
            # enc_out = enc_out + self.projection[i]((trend + season).transpose(1, 2)).transpose(1, 2)
            # result.append(self.projection[i]((x_enc).transpose(1, 2)))
        # for i in range(self.layer):
        #     # season, trend = self.decomp[i](x_enc)
        #     x1 = self.model1[i](x1)
        # for i in range(self.layer):
        #     # season, trend = self.decomp[i](x_enc)
        #     x2 = self.model2[i](x2)
        # for i in range(self.layer):
        #     x3 = self.model3[i](x3)

            # season, trend = self.decomp3(x3)
            # season = self.model3[i](season)
            # trend = self.model3_trend[i](trend)
            # x3 = season + trend     
        # x_enc = x_enc.permute(0,2,1).reshape(batch*channel,seq, 1)
        # x_enc = self.lstm(x_enc)[0]
        # x_enc = x_enc.reshape(batch, channel, seq).permute(0,2,1)

        # # 恢复顺序
        # reverse_indices = torch.argsort(indices)
        # x_enc = x_enc[reverse_indices]


        # enc_out = (torch.matmul(torch.stack(result, dim=-1),weight.unsqueeze(-1)) ).squeeze(-1).permute(0,2,1)
        
        # x_enc = season + trend
        enc_out = self.projection((x_enc ).transpose(1, 2)).transpose(1, 2)
        # enc_out1 = self.projection1((x1 ).transpose(1, 2)).transpose(1, 2)        
        # enc_out2 = self.projection2((x2 ).transpose(1, 2)).transpose(1, 2)
        # enc_out3 = self.projection3((x3 ).transpose(1, 2)).transpose(1, 2)
        # enc_out = (enc_out + enc_out2 + enc_out3 )
        # enc_out = enc_out *stdev2 + means2 
        # enc_out = enc_out + means2 
        if  self.norm_method == 'revin':
            enc_out = enc_out * stdev2 + means2 
        if self.norm_method == 'last_value':
            enc_out = enc_out + min1 
        # enc_out = self.revin(enc_out , 'denorm')

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
