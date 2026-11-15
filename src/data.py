"""Dataset loaders, image transforms, and identity-aware PK batch sampling."""

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Protocol

import torchvision.transforms as transforms
from torch.utils.data import Sampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _HasTargets(Protocol):
    """Structural protocol for datasets exposing integer class labels via `.targets`."""

    targets: Sequence[int]


def get_train_transforms(
    image_size: tuple[int, int] = (128, 128),
) -> transforms.Compose:
    """Returns data augmentation transforms for training face embeddings."""
    raise NotImplementedError

def get_eval_transforms(
    image_size: tuple[int, int] = (128, 128),
) -> transforms.Compose:
    """Returns deterministic preprocessing transforms for validation, gallery, and probe."""
    raise NotImplementedError


class PKBatchSampler(Sampler[list[int]]):
    """Samples P distinct identities with K images each per mini-batch."""
    raise NotImplementedError
