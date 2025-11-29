import os
import torch
from datetime import datetime
from pathlib import Path
from typing import Type, Tuple

from cvrunner.utils.logger import get_cv_logger
from cvrunner.experiment import BaseExperiment, DataBatch, MetricType
from cvrunner.runner import BaseRunner
from torch.utils.data import Dataset
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.data.dataset import WaymoDataset
from src.grpo.reward import PathRewardWithCollision
from src.wayformer.wayformer import build_wayformer

from runner.wayformer_runner import WayformerRunner
from experiments.base_experiments.grpo_experiment import GRPOExperiment


class GRPOWithCollisionKL(GRPOExperiment):
    @property
    def checkpoint_path(self) -> str:
        return "/home/leo/Projects/ds190/wayformer/checkpoints/15000_100_epochs_20251127_004614/checkpoint_step_11232.pt"

    @property
    def reward_class(self):
        return PathRewardWithCollision

    @property
    def old_probs_recompute_freq(self) -> int:
        return 1

    @property
    def wandb_runname(self) -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"grpo_run5_full_dataset_kl_collision_{now}"
        return f"sanity_check_{now}"

    @property
    def beta(self) -> float:
        return 0.05

    @property
    def num_epochs(self) -> int:
        return 10

    @property
    def val_freq(self) -> int:
        return 2

    @property
    def batch_size(self) -> int:
        return 64

    def build_optimizer_scheduler(
        self,
        model: torch.nn.Module,
        len_dataloader: int = 0
    ) -> Tuple[Optimizer, _LRScheduler]:
        optimizer = AdamW(
            model.parameters(),
            lr=4e-4,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs * len_dataloader
        )

        # scheduler = torch.optim.lr_scheduler.PolynomialLR(
        #     optimizer,
        #     total_iters=self.num_epochs * len_dataloader,
        #     power=0.95
        # )

        return optimizer, scheduler
