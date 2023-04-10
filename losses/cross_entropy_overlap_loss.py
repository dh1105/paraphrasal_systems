import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class CrossEntropyOverlapLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyOverlapLoss, self).__init__()

    def forward(self, inputs, outputs, eps=1):
        softmax = nn.Softmax(dim=-1)
        outputs = softmax(outputs)

        overlap_loss = nn.CrossEntropyLoss(ignore_index=0)
        outputs = outputs.view(-1, outputs.size(-1))
        inputs = inputs.view(-1)
        ce_loss = overlap_loss(outputs, inputs)
        # ce_loss = torch.log(ce_loss)
                
        return -ce_loss