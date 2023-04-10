import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()

    def forward(self, inputs, outputs, eps=1e-10):
        inputs = inputs.view(-1)
        outputs = outputs.view(-1, outputs.shape[-1])

        mask = inputs != 0
        mask = mask.type(torch.int)
        mask = mask.unsqueeze(-1)
        inputs = F.one_hot(inputs, num_classes=outputs.shape[-1])
        inputs = inputs.double()
        inputs.requires_grad = True
        inputs = inputs * mask # Apply a padding mask such that all padding in the one-hot vectors is 0
        # softmax = nn.Softmax(dim=-1)
        # outputs = softmax(outputs)
        outputs = torch.nn.functional.gumbel_softmax(outputs)

        intersection = torch.abs(inputs * outputs)
        sum = torch.abs(inputs) + torch.abs(outputs)
        union = sum - intersection
        jacc = (intersection + eps)/(union + eps)
        jacc = jacc.mean()

        jaccard_loss = jacc
                
        return jaccard_loss