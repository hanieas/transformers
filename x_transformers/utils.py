import torch
from torch import nn
import copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(
        attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src=None, tgt=None, pad=2):
        if src is not None:
            self.src = src
            self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            try:
                self.tgt = tgt[:, :-1]
            except:
                print(tgt)
                raise
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask