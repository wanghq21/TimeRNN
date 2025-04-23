import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.d_model = configs.d_model
        self.output_attention = configs.output_attention
        self.dropout = torch.nn.Dropout(0.1)
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
        
        
        self.attn_layers = 1
        self.encoder1 = torch.nn.ModuleList([AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.seq_len, configs.n_heads)
                        for i in range(self.attn_layers)])
        self.encoder2 = torch.nn.ModuleList([AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.seq_len, configs.n_heads)
                        for i in range(self.attn_layers)])

        self.norm = torch.nn.LayerNorm(self.d_model)
        self.pred = torch.nn.Linear(self.seq_len, self.pred_len)
        self.p1 = torch.nn.ModuleList([torch.nn.Linear(self.seq_len, self.seq_len) for i in range(self.attn_layers)])
        self.p2 = torch.nn.ModuleList([torch.nn.Linear(self.seq_len, self.seq_len) for i in range(self.attn_layers)])
        self.norm1 = torch.nn.ModuleList([torch.nn.LayerNorm(self.seq_len) for i in range(self.attn_layers)])
        self.norm2 = torch.nn.ModuleList([torch.nn.LayerNorm(self.seq_len) for i in range(self.attn_layers)])

        self.pro = torch.nn.Linear(self.d_model, self.enc_in)


    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        batch, seq, channel = x_enc.shape
        means2 = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means2
        stdev2 = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev2
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        x_enc = x_enc.permute(0,2,1)   

        # enc_out = enc_out.permute(0,2,1)
        # for i in range(self.attn_layers):
        #     # x_enc = self.p2[i](x_enc)
        #     enc_out = enc_out + self.dropout(self.encoder1[i](enc_out, x_enc, x_enc, attn_mask=None)[0])
        #     enc_out = self.norm1[i](enc_out)
        # enc_out = enc_out.permute(0,2,1)

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        # print(enc_out.shape)

        enc_out = enc_out.permute(0,2,1)
        # x_enc = self.p1(x_enc)
        for i in range(self.attn_layers):
            x_enc = self.p1[i](x_enc)
            x_enc = x_enc + self.dropout(self.encoder2[i](x_enc, enc_out, enc_out, attn_mask=None)[0])
            x_enc = self.norm2[i](x_enc)
        x_enc = x_enc.permute(0,2,1)
        dec_out = self.pred(x_enc.permute(0,2,1)).permute(0,2,1)

        # dec_out = self.pro(self.pred(enc_out.permute(0,2,1)).permute(0,2,1))

        dec_out = dec_out * stdev2 + means2

        return dec_out  # [B, L, D]
    
    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
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
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
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
