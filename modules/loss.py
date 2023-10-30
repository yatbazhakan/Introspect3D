
import torch.nn as nn
import torch.nn.functional as F
import torch
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0, reduction='none', mode='sigmoid'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        assert mode in ['sigmoid', 'softmax'], "mode should be either 'sigmoid' or 'softmax'"
        self.mode = mode

    def forward(self, inputs, targets):
        if self.mode == 'sigmoid':
            targets = targets.reshape(targets.shape[0], -1)
            p = torch.sigmoid(inputs)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
            p_t = p * targets + (1 - p) * (1 - targets)
            loss = ce_loss * ((1 - p_t) ** self.gamma)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        elif self.mode == 'softmax':
            if(inputs.shape[0] != 1):
                inputs.squeeze_()
                targets.squeeze_()
            # targets_onehot = F.one_hot(targets, num_classes=inputs.size(-1))
            ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
            p = F.softmax(inputs, dim=1)
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = ce_loss * ((1 - p_t) ** self.gamma)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()