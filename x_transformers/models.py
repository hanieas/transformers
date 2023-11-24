from torch import nn
from torch.nn.functional import log_softmax
from .base import LayerNorm, PositionwiseFeedForward, PositionalEncoding, Embeddings
from .utils import clones
from .decoder import DecoderLayer
from .attention import MultiHeadedAttention
from .generators import generate


class BaseDecoderGenerator(nn.Module):
    def __init__(self, N, d_model, d_ff, vocab_size,  heads, dropout, max_len=5000):
        super(BaseDecoderGenerator, self).__init__()
        self.d_model = d_model
        self.decoder = Decoder(
            N=N,
            layer=DecoderLayer(
                size=d_model,
                dropout=dropout,
                feed_forward=PositionwiseFeedForward(
                    d_model=d_model, d_ff=d_ff, dropout=dropout),
                self_attn=MultiHeadedAttention(
                    h=heads, d_model=d_model, dropout=dropout),
            ),
        )
        self.in_embed = Embeddings(d_model, vocab_size)
        self.pe = PositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len)
        self.out_embed = nn.Linear(d_model, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask):
        x = self.in_embed(x)
        x = self.pe(x)
        x = self.decoder.forward(x=x, tgt_mask=mask)
        logits = self.out_embed(x)
        probs = log_softmax(logits, dim=-1)
        return logits, probs

    def generate(self, x, iterations, stop_token=None, temperature=None, top_p=None, top_k=None):
        return generate(self, x, iterations, stop_token=stop_token, temperature=temperature, top_p=top_p, top_k=top_k)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), tgt_mask, memory=memory, src_mask=src_mask)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, tgt_mask, memory=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, tgt_mask, memory=memory, src_mask=src_mask)
        return self.norm(x)
