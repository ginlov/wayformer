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
from experiments.wayformer_experiment import WayformerExperiment


class SanityExperiment(WayformerExperiment):
    @property
    def sanity_check(self) -> bool:
        return True 

    @property
    def wandb_runname(self) -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sanity_check_{now}"

    @property
    def num_epochs(self) -> int:
        return 1000

    @property
    def val_freq(self) -> int:
        return 1

    @property
    def batch_size(self) -> int:
        return 2
