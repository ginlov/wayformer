from abc import ABC
from typing import Type, Tuple, Dict, Any, Callable, Literal
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from cvrunner.runner import BaseRunner
from cvrunner.utils.logger import get_cv_logger
from cvrunner.experiment import BaseExperiment, DataBatch, MetricType
from src.wayformer.config import DatasetConfig
from src.wayformer.metrics import WaymoMetrics
from src.wayformer.wayformer import build_wayformer
from src.wayformer.loss import WayformerLoss
from src.data.dataset import WaymoDataset, WaymoSampler, DistributedWaymoSampler
from src.data.utils import collate_fn, pad_in_case_empty_context
from src.data.visualize import visualize_scene

from runner.wayformer_runner import WayformerRunner

import cvrunner.utils.distributed as dist_utils

logger = get_cv_logger()

class WayformerExperiment(BaseExperiment, ABC):
    def __init__(self):
        pass

    @property
    def base_checkpoint_folder(self) -> str:
        return "/data/checkpoints/wayformer/"

    @property
    def val_freq(self) -> int:
        return 2

    def runner_cls(self) -> Type[BaseRunner]:
        return WayformerRunner

    @property
    def sanity_check(self) -> bool:
        return False

    @property
    def wandb_project(self) -> str:
        return "Wayformer"

    @property
    def base_data_folder(self) -> str:
        return "/home/leo/data/scenario_format_waymo/training/"

    @property
    def base_val_data_folder(self) -> str:
        return "/home/leo/data/scenario_format_waymo/training/"

    @property
    def batch_size(self) -> int:
        return 32

    @property
    def weight_decay(self) -> float:
        return 10e-3

    @property
    def num_epochs(self) -> int:
        return 30

    @property
    def dataset_config(self) -> DatasetConfig:
        return DatasetConfig()

    ########################################
    # MODEL CONFIGURATION                  #
    ########################################
    @property
    def d_model(self) -> int:
        return 256

    @property
    def nhead(self) -> int:
        return 8

    @property
    def dim_feedforward(self) -> int:
        return 512

    @property
    def num_layers(self) -> int:
        return 2

    @property
    def dropout(self) -> float:
        return 0.1

    @property
    def fusion(self) -> str:
        return "late"

    @property
    def num_latents(self) -> int:
        return 64

    @property
    def attention_type(self) -> str:
        return "latent"

    @property
    def num_decoder_layers(self) -> int:
        return 4

    @property
    def num_modes(self) -> int:
        return 6

    @property
    def num_likelihoods_proj_layers(self) -> int:
        return 1

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
            self.num_likelihoods_proj_layers,
            self.dataset_config
        )
        return model

    def build_dataset(self, partition: Literal['train', 'test', 'val']) -> WaymoDataset:
        dataset = WaymoDataset(
            base_folder=self.base_data_folder if partition == "train"\
                    else self.base_val_data_folder,
            partition=partition
        )
        if self.sanity_check:
            dataset.tracks = dataset.tracks[:self.batch_size*4]
            if partition == "train":
                dataset.weights = dataset.weights[:self.batch_size*4]
        return dataset

    def build_dataloader(self, partition: Literal['train', 'test', 'val']) -> DataLoader:
        dataset = self.build_dataset(partition=partition)

        if partition == 'train':
            sampler = DistributedWaymoSampler(dataset, shuffle= partition == "train")
        else:
            sampler = WaymoSampler(dataset, shuffle= partition == "train")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=10,
            collate_fn=collate_fn
        )
        return dataloader

    def build_loss_function(self) -> torch.nn.Module:
        return WayformerLoss()

    def build_optimizer_scheduler(
        self,
        model: torch.nn.Module,
        len_dataloader: int = 0
    ) -> Tuple[Optimizer, Any]:
        optimizer = AdamW(
            model.parameters(),
            lr=5e-4,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs * len_dataloader
        )
        # scheduler = torch.optim.lr_scheduler.ConstantLR(
        #     optimizer,
        #     factor=1.0,
        #     total_iters=self.num_epochs * len_dataloader
        # )
        #
        return optimizer, scheduler

    def build_criterion(self) -> Callable:
        return WaymoMetrics()

    def train_step(
        self,
        model: torch.nn.Module,
        data_batch: DataBatch,
        loss_function: torch.nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        device: torch.device
    ) -> MetricType:
        optimizer.zero_grad()
        data_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()}

        # Pad agent_interaction_features if no surrounding agents
        # Pad road_features if no road
        # Pad traffic_light_features if no traffic light
        agent_interaction_features, agent_interaction_mask = pad_in_case_empty_context(
            data_batch['agent_interaction_features'],
            data_batch['agent_interaction_mask'],
        )

        road_features, road_mask = pad_in_case_empty_context(
            data_batch['road_features'],
            data_batch['road_mask'],
        )

        traffic_light_features, traffic_light_mask = pad_in_case_empty_context(
            data_batch['traffic_light_features'],
            data_batch['traffic_light_mask'],
        )

        output = model(
            data_batch['agent_features'],
            agent_interaction_features,
            road_features,
            traffic_light_features,
            ~data_batch['agent_mask'],
            ~agent_interaction_mask,
            ~road_mask,
            ~traffic_light_mask,
        ) # (Axnum_modesxft_tsx4, Axnum_modesx1)

        label_pos = data_batch['label_pos']
        label_mask = data_batch['label_mask']

        loss = loss_function(
            label_pos,
            label_mask,
            output
        )

        loss['loss/loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        lr_scheduler.step()
        return {k: v.item() for k, v in loss.items()}

    def val_step(
        self,
        model: torch.nn.Module,
        data_batch: DataBatch,
        loss_function: torch.nn.Module,
        criterion: torch.nn.Module | None = None,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        data_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()}

        # Just in case, we don't have any traffic light, road or surrounding agent_features
        # Padding with zeros and setting masks to False
        agent_interaction_features, agent_interaction_mask = pad_in_case_empty_context(
            data_batch['agent_interaction_features'],
            data_batch['agent_interaction_mask'],
        )
        road_features, road_mask = pad_in_case_empty_context(
            data_batch['road_features'],
            data_batch['road_mask'],
        )
        traffic_light_features, traffic_light_mask = pad_in_case_empty_context(
            data_batch['traffic_light_features'],
            data_batch['traffic_light_mask'],
        )

        output = model(
            data_batch['agent_features'],
            agent_interaction_features,
            road_features,
            traffic_light_features,
            ~data_batch['agent_mask'],
            ~agent_interaction_mask,
            ~road_mask,
            ~traffic_light_mask,
        ) # (Axnum_modesxft_tsx4, Axnum_modesx1)

        label_pos = data_batch['label_pos']
        label_mask = data_batch['label_mask']
        loss = loss_function(
            label_pos,
            label_mask,
            output
        )
        
        metrics = {'val/' + k.split('/')[1]: v.item() for k, v in loss.items()}

        metrics['images'] = visualize_scene(
            data_batch,
            output,
            label_pos,
            label_mask
        )

        if criterion is not None:
            criterion_output = criterion(
                data_batch,
                output
            )

            metrics.update({"val/" + k: v for k, v in criterion_output.items()})
        return metrics

    def load_checkpoint(self) -> None:
        pass

    def save_checkpoint(self) -> None:
        pass
