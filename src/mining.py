"""Online triplet mining strategies including Semi-Hard negative selection."""

from typing import Literal

import torch


def compute_pairwise_distances(embeddings: torch.Tensor, squared: bool = True) -> torch.Tensor:
    """Computes the (B, B) pairwise Euclidean distance matrix for L2-normalized embeddings."""
    raise NotImplementedError

def get_anchor_positive_and_negative_masks(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds boolean masks for valid positive and negative pairs in a batch."""
    raise NotImplementedError


def mine_semi_hard_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    fallback: Literal["hardest", "none"] = "hardest",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mines online semi-hard negative triplets from a mini-batch.

    For each valid anchor-positive pair ``(a, p)``, searches for negatives ``n`` lying
    inside the semi-hard margin band:

        D(a, p) < D(a, n) < D(a, p) + margin

    where ``D(., .)`` is the squared Euclidean distance.
    """
    raise NotImplementedError
