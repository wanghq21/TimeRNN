
from typing import Callable, Optional
from einops import rearrange, repeat
import torch
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
# from models.attention import *
# from models.layers import *

import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt

class MaskAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(MaskAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # scores = scores if mask == None else scores * mask
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        A = A if mask == None else A * mask
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()


class MaskAttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    input:
        queries: (bs, L, d_model)
        keys: (_, S, d_model)
        values: (bs, S, d_model)
        mask: (L, S)
    return: (bs, L, d_model)

    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(MaskAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            mask,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out) # B, L, d_model
    



class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: BLC
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
        return res, moving_mean  #output: BLC
    
    
    
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        print(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x




class _Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model):
        super(_Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster = nn.Linear(d_model*2, 1)
        
        
    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, cluster_emb.shape[-1])      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = int(bn/self.n_vars)
        x_emb_batch = x_emb.repeat(self.n_cluster, 1)   
        cluster_emb_batch = torch.repeat_interleave(cluster_emb, bn, dim=0)
        out = torch.cat([x_emb_batch, cluster_emb_batch], dim=-1)
        prob = F.sigmoid(self.cluster(out)).squeeze(-1).reshape(self.n_cluster, bs, self.n_vars).permute(1,2,0)
        # prob: [bs, n_vars, n_cluster]
        prob_avg = torch.mean(prob, dim=0)      #[n_vars, n_cluster]
        prob_avg = F.softmax(prob_avg, dim=-1)
        return prob_avg


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        # self.fc3 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class _Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device, epsilon=0.05):
        super(Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        # linear_layer = [nn.Linear(seq_len, d_model), nn.ReLU(), nn.Linear(d_model, d_model)]
        # self.linear = MLP(seq_len, d_model)
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device) #nn.Parameter(torch.rand(n_cluster, in_dim * out_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        
        
    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        n_vars = x.shape[-1]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, self.d_model)      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1) 
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_temp)   #[bs*n_vars, n_cluster]
        num_var_pc = torch.sum(mask, dim=0)
        adpat_cluster = torch.matmul(x_emb.transpose(0,1), mask)/(num_var_pc + 1e-6)  #[d_model, n_cluster]
        cluster_emb = cluster_emb + adpat_cluster.transpose(0,1)
        prob_avg = torch.mean(prob, dim=0)      #[n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        return prob_avg, cluster_emb
    
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern

class Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device, epsilon=0.05):
        super(Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        # linear_layer = [nn.Linear(seq_len, d_model), nn.ReLU(), nn.Linear(d_model, d_model)]
        # self.linear = MLP(seq_len, d_model)
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device) #nn.Parameter(torch.rand(n_cluster, in_dim * out_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.p2c = CrossAttention(d_model, n_heads=1)
        
        
    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        n_vars = x.shape[-1]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, self.d_model)      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1) 
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)
        prob_avg = torch.mean(prob, dim=0)    #[n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_avg)   #[bs, n_vars, n_cluster]

        x_emb_ = x_emb.reshape(bs, n_vars,-1)
        cluster_emb_ = cluster_emb.repeat(bs,1,1)
        cluster_emb = self.p2c(cluster_emb_, x_emb_, x_emb_, mask=mask.transpose(0,1))

        return prob_avg, cluster_emb
    
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern


def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):   #[n_vars, n_cluster]
    Q = torch.exp(out / epsilon)
    sum_Q = torch.sum(Q, dim=1, keepdim=True) 
    Q = Q / (sum_Q)
    return Q




def cluster_aggregator(var_emb, mask):
    '''
        var_emb: (bs*patch_num, nvars, d_model)
        mask: (nvars, n_cluster)
        return: (bs*patch_num, n_cluster, d_model)
    '''
    num_var_pc = torch.sum(mask, dim=0)
    var_emb = var_emb.transpose(1,2)
    cluster_emb = torch.matmul(var_emb, mask)/(num_var_pc + 1e-6)
    cluster_emb = cluster_emb.transpose(1,2)
    return cluster_emb


    
class CrossAttention(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    input:
        queries: (bs, L, d_model)
        keys: (_, S, d_model)
        values: (bs, S, d_model)
        mask: (L, S)
    return: (bs, L, d_model)

    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(CrossAttention, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            mask,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return out # B, L, d_model


# Cell
class Patch_backbone(nn.Module):
    def __init__(self, args, device,**kwargs):
        super().__init__()
        self.n_cluster = args.n_cluster
        self.revin_layer = RevIN(args.enc_in, affine=True, subtract_last=False)
        self.out_len = args.pred_len
        self.n_vars = args.enc_in
        # Patching
        self.patch_len = args.patch_len
        self.max_seq_len = args.max_seq_len
        self.stride = args.stride
        self.padding_patch = args.padding_patch
        self.n_layers = args.e_layers
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.d_ff = args.d_ff
        self.attn_dropout = args.dropout
        self.dropout_rate = args.dropout
        self.pre_norm = args.pre_norm
        self.individual = args.individual
        self.device = device
        
        patch_num = int((args.seq_len - args.patch_len)/args.stride + 1)
        if self.padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, args.stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TV_Encoder(self.n_vars, patch_num=patch_num, patch_len=self.patch_len, max_seq_len=self.max_seq_len, 
                                   n_cluster=self.n_cluster, n_layers=self.n_layers, d_model=self.d_model, n_heads=self.n_heads, 
                                   d_k=None, d_v=None, d_ff=self.d_ff, attn_dropout=self.attn_dropout, dropout=self.dropout_rate,
                                   pre_norm=self.pre_norm, store_attn=False, pe='zeros', learn_pe=True, **kwargs)

        # Head
        self.head_nf = self.d_model * patch_num
        self.pretrain_head = args.pretrain_head
        self.use_norm = args.use_norm
        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, self.n_vars, fc_dropout=0.0) # custom head passed as a partial func with all its kwargs
        else:
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, self.out_len, head_dropout=0.0, n_cluster=self.n_cluster, device=self.device)
        
    
    def forward(self, z, cls_emb, prob):
        '''
        input:
            z: [bs, nvars, seq_len]
            cls_emb: [n_cluster, d_model]
            prob: [nvars, n_cluster]
        return: 
            z: [bs, nvars, target_window]
            cls_emb: [n_cluster, d_model]
        '''    
        if self.use_norm:                                                
            # norm
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs, nvars, patch_num, patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs, nvars, patch_len, patch_num]
        
        # model
        z, cls_emb = self.backbone(z, cls_emb, prob)                                        # z: [bs, nvars, d_model, patch_num]
        z = self.head(z, prob)                                                              # z: [bs, nvars, target_window] 
        
        if self.use_norm: 
            # denorm
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z, cls_emb
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0, n_cluster=1, device=None):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual == "i":
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        elif self.individual == "c":
            self.flatten = nn.Flatten(start_dim=-2)
            self.cluster_linear = Cluster_wise_linear(n_cluster, n_vars, nf, target_window, device)
            self.dropout = nn.Dropout(head_dropout)
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x, prob=None):                                 # x: [bs, nvars, d_model, patch_num]; prob: [nvars, n_cluster]
        if self.individual == "i":
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs, d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs, target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs, nvars, target_window]
        elif self.individual == "c":
            x = self.flatten(x)                           # x: [bs, nvars, d_model * patch_num]
            x = self.cluster_linear(x, prob)
            x = self.dropout(x)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x            #x: [bs, nvars, target_window]
        
        
    
    
class TV_Encoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024, n_cluster=1, 
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., store_attn=False, pre_norm=False,
                 pe='zeros', learn_pe=True, **kwargs):
        
        
        super().__init__()
        self.n_vars = c_in
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)       
        self.seq_len = q_len

            
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = nn.ModuleList([TimeVarAttentionLayer(c_in, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, 
                                                            store_attn=False, norm=norm, attn_dropout=attn_dropout, 
                                                            dropout=dropout, pre_norm=pre_norm, n_cluster=n_cluster) 
                                      for i in range(n_layers)])


    def forward(self, x, cls_emb, prob) -> Tensor:                                              
        # x: [bs, nvars, patch_len, patch_num]
        # cls_emb: [n_cluster, d_model]
        # prob: [nvars, n_cluster]
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs, nvars, patch_num, patch_len]
        x = self.W_P(x)                                                          # x: [bs, nvars, patch_num, d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars, patch_num, d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars, patch_num, d_model]

        # Encoding
        z = u
        for mod in self.encoder: 
            z, cls_emb = mod(z, cls_emb, prob)                                   # z: [bs * nvars, patch_num, d_model]
        
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs, nvars, patch_num, d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs, nvars, d_model, patch_num]
        
        return z, cls_emb
            

# class Cluster_wise_linear(nn.Module):
#     def __init__(self, n_cluster, n_vars, in_dim, out_dim, device):
#         super().__init__()
#         self.n_cluster = n_cluster
#         self.n_vars = n_vars
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.cw_weight = Parameter(torch.empty((n_cluster, in_dim * out_dim)))           #nn.Parameter(torch.rand(n_cluster, in_dim * out_dim), requires_grad=True)
#         nn.init.kaiming_uniform_(self.cw_weight, a=math.sqrt(5))
        
#         self.bias = Parameter(torch.empty((n_cluster, out_dim)))
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.cw_weight)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         nn.init.uniform_(self.bias, -bound, bound)

        
#     def forward(self, x, prob):
#         # x: [bs, n_vars, in_dim]
#         # prob: [n_vars, n_cluster]
#         # return: [bs, n_vars, out_dim]
#         bsz = x.shape[0]
#         # prob = self.concrete_bern(prob)
#         output = torch.mm(prob, self.cw_weight) #[n_vars, in_dim*out_dim]
#         bias = torch.mm(prob, self.bias)  #[n_vars, out_dim]
#         bias = bias.repeat(bsz, 1)
        
#         # cluster_weights_batch = cluster_weights.expand(bsz, self.n_vars, cluster_weights.shape[-1])
#         # cluster_weights_batch = cluster_weights_batch.reshape(-1, cluster_weights.shape[-1])
#         # cluster_weights_batch = cluster_weights_batch.reshape(-1, self.in_dim, self.out_dim)   #[bs*n_var, in_dim, out_dim]
#         # x = x.reshape(-1, self.in_dim).unsqueeze(1)  #[bs*n_vars, 1, in_dim]
#         # output = torch.bmm(x, cluster_weights_batch).squeeze(1) #[bs*n_vars, out_dim]
        
#         x = x.unsqueeze(-2)  #[bs, n_vars, 1, in_dim]
#         output = output.reshape(self.n_vars, self.in_dim, self.out_dim)
#         output = torch.matmul(x, output).reshape(-1, self.out_dim)   #[bs*n_vars, out_dim]
        
    
#         output = output + bias
#         output = output.reshape(bsz, -1, self.out_dim)
#         return output
    
#     def concrete_bern(self, prob, temp = 0.07):
#         random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
#         random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
#         prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
#         prob_bern = ((prob + random_noise) / temp).sigmoid()
#         return prob_bern
        



class Cluster_wise_linear(nn.Module):
    def __init__(self, n_cluster, n_vars, in_dim, out_dim, device):
        super().__init__()
        self.n_cluster = n_cluster
        self.n_vars = n_vars
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linears = nn.ModuleList()
        for i in range(n_cluster):
            self.linears.append(nn.Linear(in_dim, out_dim))

        
    def forward(self, x, prob):
        # x: [bs, n_vars, in_dim]
        # prob: [n_vars, n_cluster]
        # return: [bs, n_vars, out_dim]
        bsz = x.shape[0]
        output = []
        for layer in self.linears:
            output.append(layer(x))
        output = torch.stack(output, dim=-1).to(x.device)  #[bsz, n_vars,  out_dim, n_cluster]
        prob = prob.unsqueeze(-1)  #[n_vars, n_cluster, 1]
        output = torch.matmul(output, prob).reshape(bsz, -1, self.out_dim)   #[bsz, n_vars, out_dim]
        return output
    


class TimeVarAttentionLayer(nn.Module):
    def __init__(self, n_vars, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, pre_norm=False, n_cluster=1):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.self_attn = MaskAttentionLayer(d_model, n_heads, d_keys=d_k, d_values=d_v, dropout=attn_dropout)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.norm_var_1 = nn.LayerNorm(d_model)
        self.norm_var_2 = nn.LayerNorm(d_model)
        self.p2c = MaskAttentionLayer(d_model, n_heads, d_keys=d_k, d_values=d_v, dropout=attn_dropout)
        self.c2p = MaskAttentionLayer(d_model, n_heads, d_keys=d_k, d_values=d_v, dropout=attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.n_vars = n_vars
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.n_cluster = n_cluster
        self.ff_var = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

    def forward(self, src, clusters, prob) -> Tensor:
        '''
        Time axis: normalization -> attention -> Add & drop -> normalization -> FFN -> Add & drop
        Variates axis:  attention -> Add & drop -> normalization -> FFN -> Add & drop -> normalization

        input:
            src: [bs * nvars, patch_num, d_model]
            clusters: [n_cluster, d_model]
            prob: [nvars, n_cluster]
        return:
            output: [bs * nvars, patch_num, d_model]
            cluster_emb:  [n_cluster, d_model]
        '''
        bs = int(src.shape[0]/self.n_vars)
        # 1. Normalization
        if self.pre_norm:
            src = self.norm_attn(src)
        # 2. Multi-Head attention 
        src2 = self.self_attn(src, src, src)
        # 3. Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # 4. Feed-forward Normalization
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## 5. Feed-Forward
        src2 = self.ff(src)
        ## 6. Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)                # src: (bs * nvars, patch_num, d_model)'
        
        output = src
        
        # ipdb.set_trace()
        
        # output = rearrange(output, '(b nvars) n_patch d_model -> (b n_patch) nvars d_model', b=bs)  # (bs*patch_num, nvars, d_model)
        cluster_emb = clusters.expand(output.shape[0], clusters.shape[0], clusters.shape[1])       # (bs*patch_num, n_cluster, d_model)
        # mask = self.concrete_bern(prob)
        # # cluster_emb = self.p2c(cluster_emb, output, output, mask=mask)
        # cluster_emb = cluster_aggregator(output, mask)                              # (bs*patch_num, n_cluster, d_model
        # patch_emb = self.c2p(output, cluster_emb, cluster_emb)
        # output = output + self.dropout(patch_emb)                                      # (bs*patch_num, nvars, d_model)
        
        # output = self.norm_var_1(output)
        
        # output = output + self.dropout(self.ff_var(output))
        # output = self.norm_var_2(output)                                            # (bs*patch_num, nvars, d_model)
        # output = rearrange(output, '(b n_patch) nvars d_model -> (b nvars) n_patch d_model', b=bs)
        cluster_emb = cluster_emb.mean(0)
        return output, cluster_emb


    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern