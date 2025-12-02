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
from src.wayformer.wayformer import build_wayformer

from runner.wayformer_runner import WayformerRunner
from experiments.base_experiments.wayformer_experiment import WayformerExperiment


class BaselineExperiment(WayformerExperiment):
    @property
    def base_data_folder(self) -> str:
        return "/data/scenarionet_format/training/"

    @property
    def base_val_data_folder(self) -> str:
        return "/data/scenarionet_format/training/"

    @property
    def sanity_check(self) -> bool:
        return False

    @property
    def wandb_runname(self) -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"15000_80_epochs_XL_{now}"
        return f"sanity_check_{now}"

    @property
    def num_epochs(self) -> int:
        return 80

    @property
    def val_freq(self) -> int:
        return 2

    @property
    def batch_size(self) -> int:
        return 64

    #####################################
    # Model congiguration               #
    #####################################
    @property
    def num_layers(self) -> int:
        return 4

    @property
    def num_decoder_layers(self) -> int:
        return 6

    @property
    def d_model(self) -> int:
        return 512

    @property
    def dim_feedforward(self) -> int:
        return 2048

    @property
    def nhead(self) -> int:
        return 8

    def build_optimizer_scheduler(
        self,
        model: torch.nn.Module,
        len_dataloader: int = 0
    ) -> Tuple[Optimizer, _LRScheduler]:
        optimizer = AdamW(
            model.parameters(),
            lr=5e-4,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs * len_dataloader
        )

        return optimizer, scheduler
