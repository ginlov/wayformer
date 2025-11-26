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
    def checkpont_path(self) -> str:
        return "/home/leo/Projects/ds190/wayformer/checkpoints/new_dataset_scheduler1/checkpoint_step_4446.pt"

    @property
    def sanity_check(self) -> bool:
        return True 

    @property
    def wandb_runname(self) -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return "fine_tune_classfiication_head"
        return f"sanity_check_{now}"

    @property
    def num_epochs(self) -> int:
        return 5

    @property
    def val_freq(self) -> int:
        return 1

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
            lr=5e-4,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs * len_dataloader
        )

        return optimizer, scheduler

    def build_model(self) -> torch.nn.Module:
        model = build_wayformer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.num_layers,
            self.dropout,
            self.fusion,
            self.num_latents,
            self.attention_type,
            self.num_decoder_layers,
            self.num_modes,
            self.dataset_config
        )
        checkpoint = torch.load(self.checkpont_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        for name, param in model.named_parameters():
            if 'gmm_likelihood_projection' not in name:
                param.requires_grad = False
        return model

