import torch
from base_metric import Metric

class MLMAccuracy(Metric):
    """Tính độ chính xác cho tác vụ điền từ (MLM)"""
    def __str__(self):
        return "mlm_accuracy"

    def compute(self, preds: torch.Tensor, labels: torch.Tensor):
        # Chỉ tính toán trên các vị trí không phải -100 (vị trí bị mask)
        mask = labels != -100
        correct = (preds[mask] == labels[mask]).float()
        return correct.mean().item() if correct.numel() > 0 else 0.0