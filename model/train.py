import torch
from torch import nn
from modeling import ModelArgs, StateFormer
import modeling
from muon import Muon

# Still developing, not completed.
args = ModelArgs()
model = StateFormer(args)
device = torch.device('cuda')
model.to(device)

@modeling.cold_start(model)
def cold_start():
    pass

