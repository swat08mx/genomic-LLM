import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Model(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)  
        emb = self.pos_encoding(emb)  
        emb = emb.transpose(0, 1)  
        out = self.transformer(emb)
        out = out.transpose(0, 1)
        logits = self.mlm_head(out)
        return logits

vocab_size = len(list(vocab.keys()))
d_model = 512
num_heads = 16
num_layers = 6
max_seq_length = 201

if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"
