
# Cell
import torch
from torch import nn
import torch.nn.functional as F
from layers.RevIN import RevIN

from math import ceil
import os
import pickle
import numpy as np
import pandas as pd
import math
import random
import tqdm
from einops import rearrange, repeat, reduce
from torch import einsum
from torch.nn.utils import weight_norm


class PreNorm(nn.Module): 
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.8):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head *  heads 
        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.heads        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) / self.d_k
        
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out),attn
       
    


class c_Transformer(nn.Module):           ##Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.8):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, c_Attention(dim,  heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)) 
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x_n,attn=attn(x)
            x = x_n + x
            x = ff(x) + x
        return x,attn





class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout , patch_dim, horizon, d_model):
        super().__init__()
        
        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim),nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.mlp_head = nn.Linear(dim, d_model)#horizon)
        

    def forward(self, x):
        
        x = self.to_patch_embedding(x)
        x,attn = self.transformer(x)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x#,attn


# NystromTrans

def exists(val):
    return val is not None

# NystromParam
def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z



class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, num_landmarks,
                 pinv_iterations, eps, dropout=0.5):
        super().__init__()
        inner_dim = dim_head * heads  ##32(4*8)
        project_out = not (heads == 1 and dim_head == dim)

        ## NystromParam
        self.eps = eps
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        ## NystromParam

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  ##better to dim to dim*3

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        #         if self.residual:
        #             out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out



class Trans_C_nys(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout , patch_dim, horizon, d_model):
        super().__init__()
        
        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim),nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Nystromformer(dim, depth, heads, dim_head, mlp_dim, num_landmarks=5,pinv_iterations = 6,eps=1e-8,dropout=dropout)
        
        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x):
        
        x = self.to_patch_embedding(x)
        #x = self.dropout(x)
        x = self.transformer(x)
        x = self.mlp_head(x).squeeze()
        return x



class Nystromformer(nn.Module):  ##Register the blocks into whole network
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_landmarks,
                 pinv_iterations, eps, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, num_landmarks=num_landmarks,
                                       pinv_iterations=pinv_iterations, eps=eps, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x



class Fredformer_backbone(nn.Module):
    def __init__(self, ablation:int,  mlp_drop:float, use_nys:int, output:int, mlp_hidden:int,cf_dim:int,cf_depth :int,cf_heads:int,cf_mlp:int,cf_head_dim:int,cf_drop:float,c_in:int, context_window:int, target_window:int, patch_len:int, stride:int,  d_model:int, 
                head_dropout = 0, padding_patch = None,individual = False, revin = True, affine = True, subtract_last = False, **kwargs):
        
        super().__init__()
        self.use_nys = use_nys
        self.ablation = ablation
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in)
        self.output = output
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.targetwindow=target_window
        self.horizon = self.targetwindow
        patch_num = int((context_window - patch_len)/stride + 1)
        self.norm = nn.LayerNorm(patch_len)
        #print("depth=",cf_depth)
        # Backbone 
        self.re_attn = True
        if self.use_nys==0:
            self.fre_transformer = Trans_C(dim = cf_dim,depth = cf_depth, heads = cf_heads, mlp_dim = cf_mlp, dim_head = cf_head_dim, dropout = cf_drop, patch_dim = patch_len*2 , horizon = self.horizon*2, d_model=d_model*2)
        else:
            self.fre_transformer = Trans_C_nys(dim = cf_dim,depth = cf_depth, heads = cf_heads, mlp_dim = cf_mlp, dim_head = cf_head_dim, dropout = cf_drop, patch_dim = patch_len*2 , horizon = self.horizon*2, d_model=d_model*2)
        
        
        # Head
        self.head_nf_f  = d_model * 2 * patch_num #self.horizon * patch_num#patch_len * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=head_dropout)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window, head_dropout=head_dropout)
        
        self.ircom = nn.Linear(self.targetwindow*2,self.targetwindow)
        self.rfftlayer = nn.Linear(self.targetwindow*2-2,self.targetwindow)
        self.final = nn.Linear(self.targetwindow*2,self.targetwindow)

        #break up R&I:
        self.get_r = nn.Linear(d_model*2,d_model*2)
        self.get_i = nn.Linear(d_model*2,d_model*2)
        self.output1 = nn.Linear(target_window,target_window)


        #ablation
        self.input = nn.Linear(c_in,patch_len*2)
        self.outpt = nn.Linear(d_model*2,c_in)
        self.abfinal = nn.Linear(patch_len*patch_num,target_window)

    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]

        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
        
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag
        

        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z1: [bs x nvars x patch_num x patch_len]
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)                         # z2: [bs x nvars x patch_num x patch_len]                                                                 

        #for channel-wise_1
        z1 = z1.permute(0,2,1,3)
        z2 = z2.permute(0,2,1,3)


        # model shape
        batch_size = z1.shape[0]
        patch_num  = z1.shape[1]
        c_in       = z1.shape[2]
        patch_len  = z1.shape[3]
        
        #proposed
        z1 = torch.reshape(z1, (batch_size*patch_num,c_in,z1.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]
        z2 = torch.reshape(z2, (batch_size*patch_num,c_in,z2.shape[-1]))                            # z: [bs * patch_num,nvars, patch_len]

        z = self.fre_transformer(torch.cat((z1,z2),-1))
        z1 = self.get_r(z)
        z2 = self.get_i(z)
        

        z1 = torch.reshape(z1, (batch_size,patch_num,c_in,z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size,patch_num,c_in,z2.shape[-1]))
        

        z1 = z1.permute(0,2,1,3)                                                                    # z1: [bs, nvars， patch_num, horizon]
        z2 = z2.permute(0,2,1,3)

        z1 = self.head_f1(z1)                                                                    # z: [bs x nvars x target_window] 
        z2 = self.head_f2(z2)                                                                    # z: [bs x nvars x target_window]
        
        z = torch.fft.ifft(torch.complex(z1,z2))
        zr = z.real                                              
        zi = z.imag
        z = self.ircom(torch.cat((zr,zi),-1))


        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears1 = nn.ModuleList()
            #self.linears2 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, target_window))
                #self.linears2.append(nn.Linear(target_window, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears1[i](z)                    # z: [bs x target_window]
                #z = self.linears2[i](z)                    # z: [target_window x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
            #x = self.linear1(x)
            #x = self.linear2(x) + x
            #x = self.dropout(x)
        return x
    
class Flatten_Head_t(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout):
        super().__init__()
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, nf)
        self.linear2 = nn.Linear(nf, nf)
        self.linear3 = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        
        x = self.flatten(x)
        x = F.relu(self.linear1(x)) + x
        x = F.relu(self.linear2(x)) + x
        
        x = self.linear3(x)
        return x