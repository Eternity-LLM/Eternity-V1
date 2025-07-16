from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from dataclasses import dataclass
import torch

world_size = 1
rank = 0
use_deepseek = True
@dataclass
class ModelArgs:
    # embedding
    dim:int = 7168



class ParallelEmbedding(nn.Module):
    # Embedding layer with parallelism support across distributed processes.

    # Args:
    #     vocab_size (int) : vocabulary size.
    #     dim (int) : Embedding dimension.
    #     fine_tune_from_deepseek (bool).
    #     init_weight (torch.tensor or nn.Parameter) : DeepSeek-R1-0528 pretrained params.
    #     lora_rank (int)

    def __init__(self, vocab_size:int, dim:int, init_weight = None, lora_rank:int = 256) -> None :
        super().__init__()
        self.vocabs = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0
        self.part_vocab_size = vocab_size // world_size
        self.st_idx = rank * self.part_vocab_size
        self.en_idx = self.st_idx + self.part_vocab_size
        assert (fine_tune_from_deepseek and init_weight != None) or not fine_tune_from_deepseek
        if use_deepseek:
            self.weight = init_weight[self.st_idx:self.en_idx, :]
            self.weight.requires_grad = False
            self.A = nn.Parameter(torch.empty(self.part_vocab_size, lora_rank))
            self.B = nn.Parameter(torch.empty(lora_rank, self.part_vocab_size))
        else:
            self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        return None

