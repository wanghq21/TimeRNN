import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.layers import *
# from models.attention import *
from layers.patch_layer import *


def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):   #[n_vars, n_cluster]
    Q = torch.exp(out / epsilon)
    sum_Q = torch.sum(Q, dim=1, keepdim=True) 
    Q = Q / (sum_Q)
    return Q



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




class Model(nn.Module):
    def __init__(self, args, baseline = False, if_decomposition=False):
        super(Model, self).__init__()
        self.n_vars = args.enc_in
        self.in_len = args.seq_len
        self.out_len = args.pred_len
        self.patch_len = args.patch_len
        self.n_cluster = args.n_cluster
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.individual = args.individual
        self.baseline = baseline
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if if_decomposition:
            self.decomp_module = series_decomp(kernel_size=25)
            self.encoder_trend = Patch_backbone(args, device=self.device)
            self.encoder_res = Patch_backbone(args, device=self.device)
        if self.individual == "c":
            self.Cluster_assigner = Cluster_assigner(self.n_vars, self.n_cluster, self.in_len, self.d_model, device=self.device)
            self.cluster_emb = self.Cluster_assigner.cluster_emb
        else:
            self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(self.device)
        self.encoder = Patch_backbone(args, device=self.device)
        self.decomposition = if_decomposition
        self.cluster_prob = None
        
    def forward(self, x_seq,  x_mark_enc, x_dec, x_mark_dec, mask=None, if_update=False):       #[bs, seq_len, n_vars]
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        if self.individual == "c":
            self.cluster_prob, cluster_emb_1 = self.Cluster_assigner(x_seq, self.cluster_emb)      #[n_vars, n_cluster]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x_seq)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res, cls_emb_res = self.encoder_res(res_init, self.cluster_emb, self.cluster_prob)
            trend, cls_emd_trend = self.encoder_trend(trend_init,  self.cluster_emb, self.cluster_prob)
            out = res + trend
            cluster_emb = (cls_emb_res + cls_emd_trend)/2
            if if_update and self.individual == "c":
                self.cluster_emb = nn.Parameter(cluster_emb_1, requires_grad=True)
            out = out.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x_seq = x_seq.permute(0,2,1)
            out, cls_emb = self.encoder(x_seq, self.cluster_emb, self.cluster_prob)
            if if_update and self.individual == "c":
                self.cluster_emb = nn.Parameter(cluster_emb_1, requires_grad=True)
            out = out.permute (0,2,1)
        return base + out[:, :self.out_len, :]   #[bs, out_len, n_vars]