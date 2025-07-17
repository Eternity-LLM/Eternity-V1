from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import torch

world_size = 1
rank = 0
use_deepseek = True

@dataclass
class ModelArgs:
    # embedding
    dim:int = 7168
    vocab_size:int = 129280
    emb_lora_rank:int = 0


class ParallelEmbedding(nn.Module):
    # Embedding layer with parallelism support across distributed processes and low-rank adaptation.

    # Args:
    #     vocab_size (int) : vocabulary size.
    #     dim (int) : Embedding dimension.
    #     init_weight (optional, torch.tensor or nn.Parameter) : DeepSeek-R1-0528 pretrained params.
    #     lora_rank (int)

    def __init__(self, vocab_size:int, dim:int, init_weight = None, lora_rank:int = 256) -> None :
        super().__init__()
        self.vocabs = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0
        self.part_vocab_size = vocab_size // world_size
        self.st_idx = rank * self.part_vocab_size
        self.en_idx = self.st_idx + self.part_vocab_size
        
        self.weight = nn.Parameter(init_weight[self.st_idx:(self.en_idx+1), :]) if init_weight else nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        if use_deepseek:
        	self.weight.requires_grad = False
        
        if lora_rank:
            self.A = nn.Parameter(torch.empty(self.part_vocab_size, lora_rank))
            self.B = nn.Parameter(torch.empty(lora_rank, self.part_vocab_size))
        return None
    def forward(self, x:torch.Tensor):
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        if use_deepseek:
            y = F.embedding(x, self.weight + torch.matmul(self.A, self.B))
        else:
        	y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y
