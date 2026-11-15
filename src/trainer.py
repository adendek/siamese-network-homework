"""Training pipeline for deep metric face embedding learning."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics import evaluate_verification, generate_evaluation_pairs
from src.mining import mine_semi_hard_triplets


@dataclass
class EpochStats:
    """Container for per-epoch training and validation metrics.

    Attributes:
        train_loss: Average triplet margin loss across processed training batches.
        active_triplets_ratio: Fraction in ``[0.0, 1.0]`` of mined triplets that violated
            the margin (``d(a, p) - d(a, n) + margin > 0``).
        val_accuracy: Best 1:1 verification accuracy on validation pairs.
        val_auc: ROC Area Under the Curve on validation pairs.
        val_eer: Equal Error Rate on validation pairs (lower is better).
        val_margin: Separation gap ``mean_neg_dist - mean_pos_dist`` on validation pairs.
        epoch: 1-based epoch index.
        is_best: ``True`` if this epoch achieved the best monitored validation score so far.
    """

    train_loss: float
    active_triplets_ratio: float
    val_accuracy: float
    val_auc: float
    val_eer: float
    val_margin: float
    epoch: int = 0
    is_best: bool = False


class MetricTrainer:
    """Coordinates model training, online triplet mining, optimization, and validation.

    Args:
        model: Embedding network mapping ``(B, C, H, W)`` images to L2-normalized
            ``(B, d)`` embeddings. If the model exposes ``.backbone`` and
            ``.projection_head`` submodules, differential learning rates are applied.
        criterion: Triplet loss module taking ``(anchors, positives, negatives)``.
        train_loader: Training ``DataLoader`` (typically backed by ``PKBatchSampler``).
        val_loader: Validation ``DataLoader`` over held-out identities.
        device: Target compute device (e.g., ``torch.device('cuda')``, ``'mps'``, or ``'cpu'``).
        margin: Triplet separation margin ``alpha > 0``. If ``None``, inferred from
            ``criterion.margin`` (defaulting to ``0.2``).
        backbone_lr: Learning rate for pretrained backbone layers. Defaults to ``1e-5``.
        head_lr: Learning rate for the randomly initialized projection head.
            Defaults to ``5e-4``.
        weight_decay: AdamW weight decay coefficient. Defaults to ``1e-4``.
        epochs: Total number of training epochs. Defaults to ``15``.
        val_num_pairs: Number of balanced positive/negative pairs sampled during
            ``validate()``. Defaults to ``1000``.
        val_seed: Deterministic seed used when sampling validation pairs. Defaults to ``42``.
        mining_fallback: Fallback strategy passed to ``mine_semi_hard_triplets``
            (``'hardest'`` or ``'none'``). Defaults to ``'hardest'``.
        checkpoint_dir: Optional directory path where ``best_model.pt`` and
            ``last_checkpoint.pt`` are saved. If ``None``, checkpoints are not written to disk.
        monitor: Validation metric used to select the best checkpoint (``'eer'`` minimizes
            Equal Error Rate; ``'auc'`` maximizes ROC-AUC). Defaults to ``'eer'``.

    Raises:
        ValueError: If ``epochs <= 0``, ``margin <= 0``, or ``margin`` disagrees with
            ``criterion.margin``.

    Notes:
        - **Why differential learning rates are used:** Pretrained backbones (ResNet18,
          MobileNetV3) already encode rich low-level facial edges and textures; updating
          them with a small LR (``1e-5``) prevents catastrophic forgetting, while the
          newly initialized linear projection head needs a larger LR (``5e-4``) to align
          features onto the unit hypersphere.
        - **Why ``active_triplets_ratio`` decreases over training:** Early in training,
          embeddings are poorly separated and most mined triplets violate the margin
          (ratio near ``1.0``). As clusters separate, fewer triplets produce non-zero
          loss, indicating convergence.
        - **Best-checkpoint restoration:** At the end of ``fit()`` (including if stopped
          early via ``KeyboardInterrupt``), the best validation weights are automatically
          reloaded into ``self.model`` when ``checkpoint_dir`` is set.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        device: torch.device,
        margin: float | None = 0.2,
        backbone_lr: float = 1e-5,
        head_lr: float = 5e-4,
        weight_decay: float = 1e-4,
        epochs: int = 15,
        val_num_pairs: int = 1000,
        val_seed: int = 42,
        mining_fallback: Literal["hardest", "none"] = "hardest",
        checkpoint_dir: str | Path | None = None,
        monitor: Literal["eer", "auc"] = "eer",
    ) -> None:
        raise NotImplementedError

    def save_checkpoint(self, filename: str, epoch: int, stats: EpochStats) -> Path:
        """Saves model, optimizer, scheduler, and epoch metrics to ``checkpoint_dir``."""
        if self.checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is None; cannot save checkpoints.")
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_score": self.best_score,
                "monitor": self.monitor,
                "stats": stats,
            },
            path,
        )
        return path

    def load_checkpoint(self, checkpoint_path: str | Path, load_optimizer: bool = True) -> int:
        """Restores model (and optionally optimizer/scheduler) state from disk."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if load_optimizer and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_score = ckpt.get("best_score", self.best_score)
        return int(ckpt.get("epoch", 0))

    def train_epoch(self) -> tuple[float, float]:
        """Runs one training epoch with online semi-hard negative mining.

        Returns:
            Tuple ``(avg_loss, active_triplets_ratio)`` across processed batches.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def validate(self) -> EpochStats:
        """Evaluates 1:1 face verification metrics on held-out validation identities."""
        raise NotImplementedError

    def fit(self, resume_from: str | Path | None = None) -> list[EpochStats]:
        """Runs the full training loop across epochs with validation and checkpointing.

        Args:
            resume_from: Optional path to a saved checkpoint to resume training from.

        Returns:
            List of ``EpochStats`` recorded at the end of each epoch.
        """
        raise NotImplementedError

