"""Weighted Binary Cross Entropy loss for imbalanced multi-label classification."""
from collections import Counter
from typing import Dict, List
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss applying per-class positive weights."""

    def __init__(self, pos_weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("pos_weights", pos_weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        weights = self.pos_weights.to(logits.device)
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weights, reduction=reduction
        )


def calculate_class_weights(
    genres_lists: List[List[str]],
    label_encoder: MultiLabelBinarizer,
) -> Dict[str, float]:
    """Calculates balanced class weights: n_samples / (n_classes * class_count)."""
    counts = Counter(genre for row in genres_lists for genre in row)
    total_samples = max(len(genres_lists), 1)
    num_classes = max(len(label_encoder.classes_), 1)

    return {
        genre: total_samples / (num_classes * max(counts.get(genre, 1), 1))
        for genre in label_encoder.classes_
    }


def get_pos_weights_tensor(
    genres_lists: List[List[str]],
    label_encoder: MultiLabelBinarizer,
    device: torch.device,
) -> torch.Tensor:
    """Returns a float32 tensor of positive class weights ordered by label_encoder.classes_."""
    class_weights = calculate_class_weights(genres_lists, label_encoder)
    ordered = [class_weights[genre] for genre in label_encoder.classes_]
    return torch.tensor(ordered, dtype=torch.float32, device=device)
