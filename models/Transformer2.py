import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import DSAttention, FullAttention
from layers.SelfAttention_Family import AttentionLayer, ProbAttention, FullAttention1
from layers.Embed import DataEmbedding
import numpy as np
from layers.Embed import PatchEmbedding2, PositionalEmbedding


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.d_model = configs.d_model
        self.output_attention = configs.output_attention
        self.dropout = torch.nn.Dropout(0.1)
       # Encoder
        self.encoder = torch.nn.ModuleList([EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, alpha=configs.alpha1), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)])


        # self.encoder = AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads)
     

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            # self.decoder = Decoder(
            #     [
            #         DecoderLayer(
            #             AttentionLayer(
            #                 FullAttention(True, configs.factor, attention_dropout=configs.dropout,
            #                               output_attention=False),
            #                 configs.d_model, configs.n_heads),
            #             AttentionLayer(
            #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
            #                               output_attention=False),
            #                 configs.d_model, configs.n_heads),
            #             configs.d_model,
            #             configs.d_ff,
            #             dropout=configs.dropout,
            #             activation=configs.activation,
            #         )
            #         for l in range(configs.d_layers)
            #     ],
            #     norm_layer=torch.nn.LayerNorm(configs.d_model),
            #     projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            # )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

        self.attn_layers = configs.e_layers
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # padding = stride
        # self.patch_embedding = PatchEmbedding2(
        #             configs.d_model,patch_len, stride, padding, self.enc_in, configs.dropout)        
        # self.encoder = torch.nn.ModuleList([AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads)
        #                 for i in range(self.attn_layers)])

        self.layer1 = 1
        self.encoder1 = torch.nn.ModuleList([AttentionLayer(
                        FullAttention1(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, alpha=configs.alpha2), configs.seq_len, configs.n_heads)
                        for i in range(self.layer1)])
        self.encoder2 = torch.nn.ModuleList([AttentionLayer(
                        FullAttention1(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, alpha=configs.alpha2), configs.seq_len, configs.n_heads)
                        for i in range(self.attn_layers)])
        
        # self.norm = torch.nn.ModuleList([torch.nn.LayerNorm(self.d_model) for i in range(self.attn_layers)])
        self.pred = torch.nn.Linear(self.seq_len, self.pred_len)
        # self.pred1 = torch.nn.Linear(self.seq_len, self.pred_len)
        # self.p1 = torch.nn.Linear(self.seq_len, self.seq_len) 
        # self.p2 = torch.nn.ModuleList([torch.nn.Linear(self.seq_len, self.seq_len) for i in range(self.attn_layers)])        
        self.norm1 = torch.nn.ModuleList([torch.nn.LayerNorm(self.seq_len) 
                                        for i in range(self.layer1)])
        # self.norm2 = torch.nn.ModuleList([torch.nn.LayerNorm(self.seq_len) for i in range(self.attn_layers)])

        # self.pro = torch.nn.Linear(self.d_model, self.enc_in)
        # self.conv = torch.nn.ModuleList([torch.nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model,kernel_size=kernel,stride=kernel)
        #                 for kernel in [12,24]])
        # self.conv_trans = torch.nn.ModuleList([torch.nn.ConvTranspose1d(in_channels=self.d_model, out_channels=self.d_model,kernel_size=kernel,stride=kernel)
        #                 for kernel in [12,24]])

        # self.conv = torch.nn.ModuleList([torch.nn.Linear(self.seq_len, self.seq_len)
        #                 for i in range(self.attn_layers)])
        # self.l1 = torch.nn.ModuleList([torch.nn.Linear(self.d_model, self.d_model)
        #                 for i in range(self.attn_layers)])
        # self.conv_trans = torch.nn.ModuleList([torch.nn.Linear(self.seq_len, self.seq_len)
        #                 for kernel in [12,12,12,12]])
        # self.pro = torch.nn.Linear(self.d_model, self.enc_in)
        # self.pred1 = torch.nn.Linear(self.seq_len, self.pred_len)

        # self.pos1 = PositionalEmbedding(self.seq_len)
        # self.juedui_pos = torch.nn.Linear(1, self.seq_len, bias=False)
        self.juedui_pos = torch.nn.Embedding(num_embeddings=1000, embedding_dim=self.seq_len)
        # self.pos2 = PositionalEmbedding(self.seq_len)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch, seq, channel = x_enc.shape

        means2 = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means2
        stdev2 = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev2

        # Embedding
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        x_enc = x_enc.permute(0,2,1) 

        pos = self.juedui_pos(torch.arange(1,self.enc_in+1, 1).unsqueeze(0).to(x_enc.device))
        # print(pos.shape)
        # pos = self.pos1(x_enc)
        # x_enc = self.p1(x_enc)
        x_enc = x_enc + pos 

        # Embedding
        x_enc = x_enc.permute(0,2,1) 
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        x_enc = x_enc.permute(0,2,1) 

        enc_out = enc_out.permute(0,2,1)
        for i in range(self.layer1):
            # enc_out = self.norm1[i](enc_out)
            enc_out = enc_out + self.dropout(self.encoder1[i](enc_out, x_enc, x_enc, attn_mask=None)[0])
            enc_out = self.norm1[i](enc_out)
        enc_out = enc_out.permute(0,2,1)

        # enc_out = self.norm2[0](enc_out.permute(0,2,1)).permute(0,2,1)

        # x_enc = self.p1(x_enc)
        for i in range(self.attn_layers):
            # enc_out = self.norm2[i](enc_out.permute(0,2,1)).permute(0,2,1)
            enc_out = self.encoder[i](enc_out)[0]
            # enc_out = self.norm2[i](enc_out.permute(0,2,1)).permute(0,2,1)
            
            # x_enc = self.p2[i](x_enc)
            enc_out = enc_out.permute(0,2,1)
            # x_enc = x_enc.permute(0,2,1)
            # x_enc = self.norm2[i](x_enc)
            x_enc = x_enc + self.dropout(self.encoder2[i](x_enc, enc_out, enc_out, attn_mask=None)[0])
            # x_enc = self.norm2[i](x_enc)
            enc_out = enc_out.permute(0,2,1)
        x_enc = x_enc.permute(0,2,1)

        dec_out =  self.pred(x_enc.permute(0,2,1)).permute(0,2,1) 
        dec_out = dec_out * stdev2 + means2

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

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
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
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




# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta
#         )
#         out = out.view(B, L, -1)

#         return self.out_projection(out), attn

# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)

#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)

#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         if self.output_attention:
#             return V.contiguous(), A
#         else:
#             return V.contiguous(), None

