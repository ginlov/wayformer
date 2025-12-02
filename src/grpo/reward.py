import torch

from typing import Tuple
from src.wayformer.metrics import check_collision_for_trajectory, collision_per_timestep

class PathReward(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        targets: torch.Tensor, # [A, ts, 2]
        target_mask: torch.Tensor, # [A, ts]
        predictions: Tuple[torch.Tensor, torch.Tensor], # [A, num_modes, ts, 4], [A, num_modes]
        *args,
        **kwargs
    ):
        target_mask = target_mask.float()
        # Find nearest mode to target of each agent
        traj_preds, _ = predictions
        A, num_modes, ts, _ = traj_preds.shape
        targets_expanded = targets.unsqueeze(1).expand(-1, num_modes, -1, -1) # [A, num_modes, ts, 2]

        # Compute L2 errors for each mode, apply target mask
        target_mask_expanded = target_mask.unsqueeze(1).expand(-1, num_modes, -1) # [A, num_modes, ts]
        assert target_mask_expanded.shape == traj_preds.shape[:3], f"{target_mask_expanded.shape} vs {traj_preds.shape[:3]}"
        traj_preds = traj_preds * target_mask_expanded.unsqueeze(-1) # [A, num_modes, ts, 4]
        targets_expanded = targets_expanded * target_mask_expanded.unsqueeze(-1) # [A, num_modes, ts, 2]

        # Find best mode for each agent
        l2_errors = torch.norm(traj_preds[..., :2] - targets_expanded, dim=-1) # [A, num_modes, ts]
        total_l2_errors = l2_errors.sum(dim=-1) # [A, num_modes]

        # Boost up top 2 rewards
        sorted_l2_errors, sorted_indices = torch.sort(total_l2_errors, dim=-1) # [A, num_modes]
        rewards = -total_l2_errors
        rewards += (sorted_indices == 0).float() * 7.0
        rewards += (sorted_indices == 1).float() * 4.0

        return {"total_reward":rewards}


class PathRewardWithCollision(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        targets: torch.Tensor, # [A, ts, 2]
        target_mask: torch.Tensor, # [A, ts]
        predictions: Tuple[torch.Tensor, torch.Tensor], # [A, num_modes, ts, 4], [A, num_modes]
        agent_fut_width: torch.Tensor, # [A, ts]
        other_fut_trajs: torch.Tensor, # [A, ts, n, 2]
        other_fut_masks: torch.Tensor, # [A, ts, n]
        other_fut_widths: torch.Tensor, # [A, n]
    ):
        target_mask = target_mask.float()
        # Find nearest mode to target of each agent
        traj_preds, _ = predictions
        A, num_modes, ts, _ = traj_preds.shape
        targets_expanded = targets.unsqueeze(1).expand(-1, num_modes, -1, -1) # [A, num_modes, ts, 2]

        # Compute L2 errors for each mode, apply target mask
        target_mask_expanded = target_mask.unsqueeze(1).expand(-1, num_modes, -1) # [A, num_modes, ts]
        assert target_mask_expanded.shape == traj_preds.shape[:3], f"{target_mask_expanded.shape} vs {traj_preds.shape[:3]}"
        traj_preds = traj_preds * target_mask_expanded.unsqueeze(-1) # [A, num_modes, ts, 4]
        targets_expanded = targets_expanded * target_mask_expanded.unsqueeze(-1) # [A, num_modes, ts, 2]

        # Find best mode for each agent
        l2_errors = torch.norm(traj_preds[..., :2] - targets_expanded, dim=-1) # [A, num_modes, ts]
        total_l2_errors = l2_errors.sum(dim=-1) # [A, num_modes]

        # Boost up top 2 rewards
        sorted_l2_errors, sorted_indices = torch.sort(total_l2_errors, dim=-1) # [A, num_modes]
        l2_rewards = -total_l2_errors
        l2_rewards += (sorted_indices == 0).float() * 7.0
        l2_rewards += (sorted_indices == 1).float() * 4.0


        # Check for collision
        traj_width = agent_fut_width.unsqueeze(1).expand(-1, num_modes, -1) # [A, num_modes, ts]

        collision_matrix = collision_per_timestep(
            traj_preds[..., :2], # [A, num_modes, ts, 2]
            target_mask_expanded, # [A, num_modes, ts]
            traj_width, # [A, num_modes, ts]
            other_fut_trajs,
            other_fut_masks,
            other_fut_widths, # [A, n]
            collision_threshold=0.3
        ) # [A, num_modes, m, ts]
        collision_matrix = collision_matrix.sum(dim=(-2, -1)) # [A, num_modes]

        total = l2_rewards - collision_matrix * 5
        return {"total_reward": total, "collision_penalty": -collision_matrix * 5, "l2_error_penalty": l2_rewards}

