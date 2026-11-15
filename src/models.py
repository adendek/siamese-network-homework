"""Embedding models for face recognition and deep metric learning."""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FaceEmbeddingNet(nn.Module):
    """Backbone neural network with projection head for hypersphere metric embeddings.

    Adapts a vision backbone (e.g. ResNet, MobileNet) to output an L2-normalized
    d-dimensional feature vector on the unit hypersphere.
    """
    raise NotImplementedError
