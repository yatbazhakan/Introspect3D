import torch
import torchmetrics
from sklearn.metrics import roc_curve
import numpy as np

class TPRatFPR(torchmetrics.Metric):
    def __init__(self, fpr_threshold=0.05, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.fpr_threshold = fpr_threshold
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_scores", default=[], dist_reduce_fx="cat")

    def update(self, y_true: torch.Tensor, y_scores: torch.Tensor):
        self.y_true.append(y_true)
        self.y_scores.append(y_scores[:, 1])  # Assuming y_scores is 2D and class of interest is the second column

    def compute(self):
        y_true = torch.cat(self.y_true, dim=0).numpy()
        y_scores = torch.cat(self.y_scores, dim=0).numpy()

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        print(fpr,"\n", tpr)
        idx  = np.where(fpr <= self.fpr_threshold)[0][-1]  # Get the last index where FPR is below the threshold
        #idx = np.argmin(np.abs(fpr - self.fpr_threshold))
        return tpr[idx]

class FPRatTPR(torchmetrics.Metric):
    def __init__(self, tpr_threshold=0.95, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tpr_threshold = tpr_threshold
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_scores", default=[], dist_reduce_fx="cat")

    def update(self, y_true: torch.Tensor, y_scores: torch.Tensor):
        self.y_true.append(y_true)
        self.y_scores.append(y_scores[:, 1])  # Assuming y_scores is 2D and class of interest is the second column

    def compute(self):
        y_true = torch.cat(self.y_true, dim=0).numpy()
        y_scores = torch.cat(self.y_scores, dim=0).numpy()

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        print(fpr,"\n", tpr)
        idx = np.where(tpr >= self.tpr_threshold)[0][0]
        # idx = np.argmin(np.abs(tpr - self.tpr_threshold))
        return fpr[idx]
