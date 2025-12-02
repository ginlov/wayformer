from abc import ABC
from typing import Type, Tuple, Dict, Any
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm

from cvrunner.runner import BaseRunner
from cvrunner.utils.logger import get_cv_logger
from cvrunner.experiment import BaseExperiment, DataBatch, MetricType
from src.wayformer.config import DatasetConfig
from src.wayformer.wayformer import build_wayformer
from src.wayformer.loss import WayformerLoss
from src.grpo.loss import GRPOLoss
from src.grpo.reward import PathReward, PathRewardWithCollision
from src.data.dataset import WaymoDataset, GRPOSampler, WaymoSampler
from src.data.utils import collate_fn, visualize_scene, pad_in_case_empty_context

from runner.grpo_runner import GRPORunner

logger = get_cv_logger()

class GRPOExperiment(BaseExperiment, ABC):
    def __init__(self):
        pass

    @property
    def base_checkpoint_folder(self) -> str:
        return "/data/checkpoints/grpo_wayformer/"

    @property
    def num_train_samples(self) -> int | None:
        return 15000

    @property
    def num_val_samples(self) -> int | None:
        return None

    @property
    def reward_class(self):
        return PathRewardWithCollision

    @property
    def old_probs_recompute_freq(self) -> int:
        return 3

    @property
    def checkpoint_path(self) -> str:
        return "/home/leo/Projects/ds190/wayformer/checkpoints/new_dataset_scheduler1/checkpoint_step_4446.pt"

    @property
    def epsilon(self) -> float:
        return 0.2

    @property
    def beta(self) -> float:
        return 0.01

    @property
    def val_freq(self) -> int:
        return 2

    def runner_cls(self) -> Type[BaseRunner]:
        return GRPORunner

    @property
    def sanity_check(self) -> bool:
        return False

    @property
    def wandb_project(self) -> str:
        return "GRPO_Wayformer"

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

    def build_model(self) -> torch.nn.Module:
        return build_wayformer(
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

    def build_dataset(self, partition: str) -> WaymoDataset:
        dataset = WaymoDataset(
            base_folder=self.base_data_folder if partition == "train" else self.base_val_data_folder,
            partition=partition
        )
        dataset.weights = []
        if self.sanity_check:
            dataset.tracks = dataset.tracks[:self.batch_size * 2]
        return dataset

    def build_dataloader(self, partition: str) -> DataLoader:
        dataset = self.build_dataset(partition=partition)

        sampler = GRPOSampler(
            dataset,
            subset_size=self.num_train_samples if partition == 'train' else self.num_val_samples
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=10,
            collate_fn=collate_fn
        )
        return dataloader

    def build_loss_function(self) -> torch.nn.Module:
        return GRPOLoss(self.epsilon, self.beta)

    def build_optimizer_scheduler(
        self,
        model: torch.nn.Module,
        len_dataloader: int = 0
    ) -> Tuple[Optimizer, _LRScheduler]:
        optimizer = AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=self.weight_decay
        )

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.num_epochs * len_dataloader
        # )
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=self.num_epochs * len_dataloader
        )

        return optimizer, scheduler

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

        # Pad surrounding environment features
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

        rewards = data_batch['reward_fn'](
            label_pos,
            label_mask,
            output,
            data_batch['agent_future_width'],
            data_batch['other_agents_future_pos'],
            data_batch['other_agents_future_mask'],
            data_batch['other_agents_future_width']
        ) # [A, num_modes]

        loss = loss_function(
            rewards["total_reward"],
            data_batch['ref_probs'],
            data_batch['old_probs'],
            output[1],
        )

        loss["loss/loss"].backward()
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

        # Pad surrounding environment features
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

        rewards = data_batch['reward_fn'](
            label_pos,
            label_mask,
            output,
            data_batch['agent_future_width'],
            data_batch['other_agents_future_pos'],
            data_batch['other_agents_future_mask'],
            data_batch['other_agents_future_width']
        ) # [A, num_modes]

        rewards = {k: v * output[1] for k, v in rewards.items()}

        metrics = {'val/' + k: v.mean().item() for k, v in rewards.items()}
        if criterion is not None:
            criterion_output = criterion(
                data_batch,
                output
            )
            metrics.update({"val/"+k: v for k, v in criterion_output.items()})
        # metrics['images'] = visualize_scene(
        #     data_batch,
        #     output,
        #     label_pos,
        #     label_mask
        # )
        return metrics

    def load_checkpoint(self) -> None:
        pass

    def save_checkpoint(self) -> None:
        pass

    def compute_ref_probs(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = 'cpu'
    ) -> Dict[Any, torch.Tensor]:
        ref_probs = {}
        for data_batch in tqdm(dataloader):
            data_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_batch.items()}

            # Pad surrounding environment features
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

            idx = data_batch['idx']
            gmm_likelihoods = output[1]  # [A, num_modes]
 
            ref_probs.update({
                item: gmm_likelihoods[i].detach() for i, item in enumerate(idx)
            })
        return ref_probs
