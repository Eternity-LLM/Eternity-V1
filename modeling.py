from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from dataclasses import dataclass
import torch

@dataclass
class ModelArgs:
    # embedding
    dim:int = 7168
    world_size:int = 1
    fine_tune_from_deepseek:bool = True

class ParallelEmbedding(nn.Module):
    # Embedding layer with parallelism support across distributed processes.

    # Args:
    #     vocab_size (int) : vocabulary size.
    #     dim (int) : Embedding dimension.

    def __init__(self, vocab_size:int, dim:int) -> None :
        super().__init__()