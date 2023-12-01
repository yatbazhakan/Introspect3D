import torch
import torchmetrics

class TPRatFPR5andFPRatTPR95(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_scores", default=[], dist_reduce_fx="cat")

    def update(self, y_true: torch.Tensor, y_scores: torch.Tensor):
        # Update states
        self.y_true.append(y_true)
        self.y_scores.append(y_scores)
    def calculate_tpr_fpr_metrics(self,y_true, y_scores):
        """
        Calculate TPR@FPR5 and FPR@TPR95 metrics.

        Args:
        y_true (Tensor): True binary labels in range {0, 1}.
        y_scores (Tensor): Target scores, confidence values, or non-thresholded measure of decisions.

        Returns:
        tuple: (tpr_at_fpr5, fpr_at_tpr95)
        """
        # Sort scores and corresponding truth values
        sorted_indices = torch.argsort(y_scores, descending=True)
        sorted_scores = y_scores[sorted_indices]
        sorted_true = y_true[sorted_indices]

        # Compute TPR and FPR at each threshold
        tprs, fprs = [], []
        for threshold in sorted_scores:
            preds = y_scores >= threshold
            tp = torch.sum((preds == 1) & (y_true == 1)).float()
            fp = torch.sum((preds == 1) & (y_true == 0)).float()
            fn = torch.sum((preds == 0) & (y_true == 1)).float()
            tn = torch.sum((preds == 0) & (y_true == 0)).float()

            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)

            tprs.append(tpr)
            fprs.append(fpr)

        # Convert lists to tensors for easier manipulation
        tprs = torch.tensor(tprs)
        fprs = torch.tensor(fprs)

        # Find the closest FPR to 0.05 (5%)
        idx_fpr5 = torch.argmin(torch.abs(fprs - 0.05))
        tpr_at_fpr5 = tprs[idx_fpr5]

        # Find the closest TPR to 0.95 (95%)
        idx_tpr95 = torch.argmin(torch.abs(tprs - 0.95))
        fpr_at_tpr95 = fprs[idx_tpr95]

        return tpr_at_fpr5.item(), fpr_at_tpr95.item()
    def compute(self):
        # Concatenate all batches
        y_true = torch.cat(self.y_true, dim=0)
        y_scores = torch.cat(self.y_scores, dim=0)

        # Compute the metrics
        return self.calculate_tpr_fpr_metrics(y_true, y_scores)