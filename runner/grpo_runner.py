import itertools
from typing import TYPE_CHECKING, Any
import os
import random

import torch
from torch.utils.data import DataLoader
from cvrunner.runner import TrainRunner
from cvrunner.utils.logger import get_cv_logger

from src.data.dataset import WaymoSampler, GRPOSampler
from src.data.utils import collate_fn
from src.wayformer.utils import cal_grad_norm, cal_param_norm
from src.grpo.reward import PathReward

if TYPE_CHECKING:
    from experiments.grpo_experiment import GRPOExperiment

logger = get_cv_logger()

class GRPORunner(TrainRunner):
    def __init__(
        self,
        experiment: type["GRPOExperiment"]
    ):
        super().__init__(experiment=experiment)
        self.load_checkpoint()

        # I want to freeze all layers except the gmm likelihood projection head
        for name, param in self.model.named_parameters():
            if "gmm_likelihood_projection" not in name:
                param.requires_grad = False

        self.optimizer, self.lr_scheduler = self.experiment.build_optimizer_scheduler(
            model=self.model,
            len_dataloader = len(self.train_dataloader)
        )
        logger.info(f"Training dataset has {len(self.train_dataloader.dataset)} samples.")
        logger.info(f"Validation dataset has {len(self.val_dataloader.dataset)} samples.")
        logger.info(f"Number of training steps per epoch: {len(self.train_dataloader)}.")
        logger.info(f"Number of validation steps per epoch: {len(self.val_dataloader)}.")
        # Compute number of trainable parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {num_params}.")

        logger.info("Precomputing reference probabilities for GRPO.")
        self.ref_probs = {}
        original_dataset = self.experiment.build_dataset(partition='train')
        original_dataloader = DataLoader(
            self.experiment.build_dataset(partition='train'),
            batch_size=self.experiment.batch_size,
            shuffle=False,
            num_workers=10,
            collate_fn=collate_fn,
            sampler=WaymoSampler(original_dataset)
        )
        self.sampler = GRPOSampler(
            original_dataset,
            15000
        )
        self.train_dataloader = DataLoader(
            original_dataset,
            batch_size=self.experiment.batch_size,
            shuffle=False,
            num_workers=10,
            collate_fn=collate_fn,
            sampler=self.sampler
        )
        self.compute_reference_prob(original_dataloader)

        logger.info("Initilize path reward")
        self.path_reward = self.experiment.reward_class()

        # For this only, frequency to recompute old probablities
        self.epoch = 0

        # Get checkpoint folder for consistency
        self.checkpoint_folder = os.path.join('checkpoints', logger._wandb_runname) \
                            if logger._wandb_runname else os.path.join('checkpoints', self.experiment.wandb_runname)
        logger.info(f"Checkpoint folder: {self.checkpoint_folder}")

    def compute_reference_prob(
        self,
        dataloader
    ):
        logger.info("Computing reference probabilities for GRPO.")
        self.model.eval()
        with torch.no_grad():
            self.ref_probs = self.experiment.compute_ref_probs(
                self.model,
                dataloader,
                self.device
            )
        logger.info("Reference probabilities computation completed.")

    def run(self):
        logger.info(f"Starting training for {self.experiment.num_epochs} epochs.")
        num_epochs = self.experiment.num_epochs
        val_freq = self.experiment.val_freq
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}.")
            self.train_epoch_start()
            self.train_epoch()
            self.epoch += 1
            self.train_epoch_end()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed.")

            if (epoch + 1) % val_freq == 0:
                logger.info(f"Starting validation at epoch {epoch + 1}.")

                self.val_epoch_start()
                self.val_epoch()
                self.val_epoch_end()
                self.checkpoint()
                logger.info(f"Validation at epoch {epoch + 1} completed.")

    def train_epoch_start(self):
        super().train_epoch_start()
        # Each epoch, limit to 1000 samples, RL does not need full dataset
        if self.epoch % self.experiment.old_probs_recompute_freq == 0:
            self.sampler.refresh()
            self.model.eval()
            with torch.no_grad():
                self.old_probs = self.experiment.compute_ref_probs(
                    self.model,
                    self.train_dataloader,
                    self.device
                )
            self.model.train()

    def train_epoch(self):
        num_step = len(self.train_dataloader)
        for i, data in enumerate(self.train_dataloader):
            logger.info(f"Training step {i + 1}/{num_step}.")
            idx = data['idx']
            data['ref_probs'] = torch.stack([self.ref_probs[_id] for _id in idx], dim=0)
            data['old_probs'] = torch.stack([self.old_probs[_id] for _id in idx], dim=0)
            data['reward_fn'] = self.path_reward
            self.train_step(data)

    def train_step(
        self,
        data_batch: Any,
    ):
        super().train_step(data_batch)
        with torch.no_grad():
            param_norm = cal_param_norm(self.model)
            logger.log_metrics(param_norm, local_step=self.step)

            grad_norm = cal_grad_norm(self.model)
            logger.log_metrics(grad_norm, local_step=self.step)

            logger.log_metrics({'lr': self.optimizer.param_groups[0]['lr']}, local_step=self.step)

    def val_epoch(self):
        with torch.no_grad():
            for data in self.val_dataloader:
                data['reward_fn'] = self.path_reward
                self.val_step(data)

    def val_step(
        self,
        data_batch: Any,
    ):
        with torch.no_grad():
            outputs = self.experiment.val_step(
                model=self.model,
                data_batch=data_batch,
                loss_function=self.loss_function,
                device=self.device
            )

            self.val_metrics.update({k: v for k, v in outputs.items() if 'val/' in k})

    def checkpoint(self):
        super().checkpoint()
        logger.info(f"Checkpoint saved at step {self.step}.")
        checkpoint_path = os.path.join('checkpoints', self.checkpoint_folder, f'checkpoint_step_{self.step}.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'step': self.step,
        }, checkpoint_path)

    def load_checkpoint(self):
        logger.info(f"Loading checkpoint from f{self.experiment.checkpoint_path}")
        checkpoint = torch.load(self.experiment.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state dict loaded.")
