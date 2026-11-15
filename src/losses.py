"""Vectorized Triplet Margin Loss implementation."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLoss(nn.Module):
    """Vectorized Triplet Margin Loss from scratch."""

    def __init__(
        self,
        margin: float = 0.2,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """Initializes the TripletMarginLoss."""
        raise NotImplementedError

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Computes triplet margin loss over a batch of mined embeddings."""
        raise NotImplementedError