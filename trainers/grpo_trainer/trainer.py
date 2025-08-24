from .reward_models import *
from ...muon import Muon
from ...modeling import StateFormer

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List
from transformers import AutoTokenizer

class GRPOTrainer:
    def __init__(
            self,
            model:StateFormer,
            optimizer:Muon,
            epsilon:float,
            beta:float,
            n_groups:int,
            reward_fn:RewardModel,
            tokenizer:AutoTokenizer
    ) -> None:
        self.model = model
        self.optim = optimizer
        self.__min = 1 - epsilon
        self.__max = 1 + epsilon
        self.beta = beta
        self.n_groups = n_groups
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
    def __sampling(self, tokens:torch.Tensor, eos_id:int, temp:float=0.8, padding_mask:Optional[torch.Tensor] = None) -> List[str]:
        outputs = []
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        prompt_len = tokens.shape[1]
        if prompt_len > self.model.max_seq_len:
            raise RuntimeError(f'Input sequence length ({prompt_len}) must not be larger than maximum sequence length ({self.model.max_seq_len}).')
        if temp <= 0:
            raise ValueError(f'Invalid softmax temperature {temp}.')
        for i in range(self.n_groups):
            inputs = torch.zeros(
                tokens.shape[0],
                self.model.max_seq_len
            )
            inputs[:, :prompt_len] = tokens
            prev_pos = 0
            for cur_pos in range(prompt_len, self.model.max_seq_len):
                logits = self.model.forward(tokens=inputs[:, prev_pos:cur_pos], start_pos=prev_pos, padding_mask=padding_mask)
                logits = logits / temp
                probs = torch.softmax(logits, dim=-1)
                next_token = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)[:, -1]
                inputs[:, cur_pos] = next_token
                if next_token[0, 0] == eos_id:
                    inputs = inputs[:, :cur_pos+1]
                    break
            output = self.tokenizer.decode(input[0], skip_special_tokens=True)
            outputs.append(output)
        return outputs
    
    def __compute_reward(self, quesion:str, truth:str, output:List[str])->List[float]:
        return self.reward_fn(question=quesion, results=output, truth=truth)
    
    def __compute_advantage(self, rewards:List[float]) -> torch.Tensor:
        rewards = torch.tensor(rewards)
        r_mean, r_std = torch.mean(rewards), torch.std(rewards)
        advantages = (rewards - r_mean) / r_std
        return advantages, r_mean, r_std