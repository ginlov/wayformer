from typing import TYPE_CHECKING, Any
import os

import torch
from cvrunner.runner import DistributedTrainRunner
from cvrunner.utils.logger import get_cv_logger
import cvrunner.utils.distributed as dist_utils

from src.utils import cal_grad_norm, cal_param_norm

if TYPE_CHECKING:
    from experiments.base_experiments.wayformer_experiment import WayformerExperiment

logger = get_cv_logger()

class WayformerRunner(DistributedTrainRunner):
    def __init__(
        self,
        experiment: type["WayformerExperiment"]
    ):
        super().__init__(experiment=experiment)

        ## Log dataset and model information
        logger.info("Training dataset has %d samples.", len(self.train_dataloader.dataset))
        logger.info("Validation dataset has %d samples.", len(self.val_dataloader.dataset))
        logger.info("Number of training steps per epoch: %d.", len(self.train_dataloader))
        logger.info("Number of validation steps per epoch: %d.", len(self.val_dataloader))

        # Compute number of trainable parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Number of trainable parameters: %d.", num_params)
        logger.info("WayformerRunner initialized.")

        if dist_utils.is_main_process():
            # Get checkpoint folder for consistency
            self.checkpoint_folder = os.path.join(self.experiment.base_checkpoint_folder, logger._wandb_runname) \
                            if logger._wandb_runname else os.path.join(self.experiment.base_checkpoint_folder, self.experiment.wandb_runname)
            logger.info("Checkpoint folder: %s", self.checkpoint_folder)

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
            logger.log_metrics(param_norm, step=self.step, stdout=False)

            grad_norm = cal_grad_norm(self.model)
            logger.log_metrics(grad_norm, step=self.step, stdout=False)

            logger.log_metrics({'lr': self.optimizer.param_groups[0]['lr']}, step=self.step)

    def val_step(
        self,
        data_batch: Any,
    ):
        with torch.no_grad():
            outputs = self.experiment.val_step(
                model=self.model,
                data_batch=data_batch,
                loss_function=self.loss_function,
                criterion=self.criterion,
                device=self.device
            )

            self.val_metrics.update({k: v for k, v in outputs.items() if 'val/' in k})
            logger.log_images(
                list(data_batch['idx']),
                list(outputs['images']),
                self.step
            )

    def checkpoint(self):
        super().checkpoint()
        logger.info(f"Checkpoint saved at local step {dist_utils.get_global_step(self.step)}.")
        checkpoint_path = os.path.join(self.checkpoint_folder, f'checkpoint_step_{self.step}.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'step': self.step,
        }, checkpoint_path)
