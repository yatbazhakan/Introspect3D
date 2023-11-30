import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

class FocalLoss(nn.Module):
    def __init__(self, weight: Optional[Tensor] = None, gamma: float = 0, reduction: str = 'mean', ignore_index: int = -100):
        super(FocalLoss, self).__init__()
        if weight is not None and not isinstance(weight, Tensor):
            raise TypeError("alpha must be of type Tensor or None")
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(weight=None, reduction='none', ignore_index=ignore_index)

    def forward(self, inputs, targets):
        if inputs.ndim > 2:
            # Flatten the inputs and targets
            c = inputs.shape[1]
            inputs = inputs.permute(0, *range(2, inputs.ndim), 1).reshape(-1, c)
            targets = targets.view(-1)

        unignored_mask = targets != self.ignore_index
        targets = targets[unignored_mask]
        if len(targets) == 0:
            return torch.tensor(0.)
        inputs = inputs[unignored_mask]

        log_p = F.log_softmax(inputs, dim=-1)
        ce = self.nll_loss(log_p, targets)

        if self.weight is not None:
            weight = self.weight[targets]
            ce = weight * ce

        all_rows = torch.arange(len(inputs))
        log_pt = log_p[all_rows, targets]
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
