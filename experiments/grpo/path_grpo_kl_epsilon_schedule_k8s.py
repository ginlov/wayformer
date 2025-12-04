import os
import torch
import pickle
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
from src.grpo.reward import PathRewardWithCollision, PathReward
from src.wayformer.wayformer import build_wayformer

from runner.wayformer_runner import WayformerRunner
from experiments.base_experiments.grpo_experiment import GRPOExperiment

class EpsilonSchedule:
    def __init__(self, final_epsilon: float, total_iters: int):
        self.final_epsilon = final_epsilon
        self.total_iters = total_iters
        step_size = (1.0 - final_epsilon) / total_iters
        self.epsilon_values = [1.0 - step_size * i for i in range(total_iters)]
        self.current_step = 0

    @property
    def epsilon(self) -> float:
        if self.current_step < self.total_iters:
            eps = self.epsilon_values[self.current_step]
            return eps
        else:
            return self.final_epsilon

    def step(self):
        if self.current_step < self.total_iters:
            self.current_step += 1

class GRPOWithCollisionKL(GRPOExperiment):
    @property
    def epsilon(self):
        return EpsilonSchedule(
            final_epsilon=0.05,
            total_iters=5000
        )

    @property
    def num_train_samples(self) -> int | None:
        return None

    @property
    def base_data_folder(self) -> str:
        return "/data/scenarionet_format/training/"

    @property
    def base_val_data_folder(self) -> str:
        return "/data/scenarionet_format/training/"

    @property
    def checkpoint_path(self) -> str:
        # return "/home/leo/Projects/ds190/wayformer/checkpoints/15000_100_epochs_20251127_004614/checkpoint_step_11232.pt"
        return "/data/checkpoints/wayformer/cvrunner-job-6d6gt/checkpoint_step_7020.pt"

    @property
    def reward_class(self):
        return PathReward

    @property
    def old_probs_recompute_freq(self) -> int:
        return 4

    @property
    def sanity_check(self) -> bool:
        return False

    @property
    def wandb_runname(self) -> str:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"grpo_run5_mismatches_kl_path_{now}"

    @property
    def beta(self) -> float:
        return 0.02

    @property
    def num_epochs(self) -> int:
        return 40

    @property
    def val_freq(self) -> int:
        return 2

    @property
    def batch_size(self) -> int:
        return 128

    def build_dataset(self, partition: str):
        dataset = super().build_dataset(partition)
        if partition == 'train':
            with open('assets/grpo/top1_mismatches.pkl', 'rb') as f:
                training_tracks = pickle.load(f)
            # Sample a subset of tracks
            # The total number of tracks is len(training_tracks)
            # Select 50% from training_tracks
            # And 50% from dataset.tracks which are not in training_tracks
            final_tracks = training_tracks.copy()
            final_tracks = final_tracks[:len(training_tracks)//2]
            remaining_tracks = [track for track in dataset.tracks if track not in training_tracks]
            num_additional_tracks = len(training_tracks) // 2
            if num_additional_tracks > 0:
                additional_tracks = remaining_tracks[:num_additional_tracks]
                # get 50% from final_tracks and 50% from additional_tracks
                final_tracks.extend(additional_tracks)
            dataset.tracks = final_tracks
            dataset.weights = []
        return dataset

    def train_step(
        self,
        model: torch.nn.Module,
        data_batch: DataBatch,
        loss_function: torch.nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        device: torch.device
    ):
        output = super().train_step(
            model,
            data_batch,
            loss_function,
            optimizer,
            lr_scheduler,
            device
        )
        loss_function.epsilon.step()
        return output


    def build_optimizer_scheduler(
        self,
        model: torch.nn.Module,
        len_dataloader: int = 0
    ) -> Tuple[Optimizer, _LRScheduler]:
        optimizer = AdamW(
            model.parameters(),
            lr=12e-4,
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
