
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
import torch

@dataclass
class ModelArgs:
    # model
    d_model:int = 7168
    max_seq_len:int = 4096*4
    max_batch_size:int = 8
    # mla (multi-head latent attention)
    lora_rank:int = 1024
    num_heads:int = 16
    v_head_dim:int = 128

world_size = 1

class ParallelEmbedding(nn.Module):
    '''
    This is a layer from DeepSeek-V3 (https://github.com/deepseek-ai/deepseek-v3/)

    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    '''
    def __init__(self, vocab_size:int, dim:int, rank:int = 0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        '''
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y

class Linear(nn.Linear):
    pass

class MLA(nn.Module):
    '''
    Multi-Head Latent Attention
    '''
    def __init__(self, args:ModelArgs) -> None :
        assert args.d_model % args.num_heads == 0
        self.d_heads = args.d_model // args.num_heads
        self.v_head_dim = args.v_head_dim
        self.dim = args.d_model
        self.lora_rank = args.lora_rank
        self.n_heads = args.num_heads
        self.scale = self.d_heads ** 0.5

        self.query = Linear(self.dim, self.lora_rank, bias = False)
        self.key_value = Linear(self.dim, self.lora_rank, bias = False)
        self.value_b = Linear(self.lora_rank, self.v_head_dim * self.n_heads, bias = False)
        self.wo = Linear(self.v_head_dim * self.n_heads, self.dim)
