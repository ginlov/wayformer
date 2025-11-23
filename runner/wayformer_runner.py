import itertools
from typing import TYPE_CHECKING, Any

import torch
from cvrunner.runner import TrainRunner
from cvrunner.utils.logger import get_cv_logger

from src.wayformer.utils import cal_grad_norm, cal_param_norm

if TYPE_CHECKING:
    from experiments.wayformer_experiment import WayformerExperiment

logger = get_cv_logger()

class WayformerRunner(TrainRunner):
    def __init__(
        self,
        experiment: type["WayformerExperiment"]
    ):
        super().__init__(experiment=experiment)
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
        logger.info("WayformerRunner initialized.")

    def run(self):
        logger.info(f"Starting training for {self.experiment.num_epochs} epochs.")
        num_epochs = self.experiment.num_epochs
        val_freq = self.experiment.val_freq
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}.")
            self.train_epoch_start()
            self.train_epoch()
            self.train_epoch_end()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed.")

            if (epoch + 1) % val_freq == 0:
                logger.info(f"Starting validation at epoch {epoch + 1}.")

                self.val_epoch_start()
                self.val_epoch()
                self.val_epoch_end()
                logger.info(f"Validation at epoch {epoch + 1} completed.")

    def train_epoch(self):
        num_step = len(self.train_dataloader)
        for i, data in enumerate(self.train_dataloader):
            logger.info(f"Training step {i + 1}/{num_step}.")
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
            logger.log_images(
                list(data_batch['idx']),
                list(outputs['images']),
                self.step
            )

    def val_epoch_end(self):
        logger.log_metrics(self.val_metrics.summary(), local_step=self.step)

