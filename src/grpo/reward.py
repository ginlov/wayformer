import torch

from typing import Tuple

def check_collision_for_trajectory(
    trajectories_of_interest: torch.Tensor,  # [B, k, m+1, 2]
    masks_of_interest: torch.Tensor,         # [B, k, m+1]
    widths_of_interest: torch.Tensor,        # [B, k] or [B, k, m+1] - width of vehicles for trajectories of interest
    other_trajectories: torch.Tensor,        # [B, n, m+1, 2]
    other_masks: torch.Tensor,               # [B, n, m+1]
    other_widths: torch.Tensor,              # [B, n] or [B, n, m+1] - width of vehicles for other trajectories
    collision_threshold: float = 0.0         # additional safety margin beyond vehicle widths
):
    """
    Check collisions between k trajectories of interest and n other trajectories in batch.
    
    Args:
        trajectories_of_interest: [B, k, m+1, 2] tensor of k trajectories to check per batch
        masks_of_interest: [B, k, m+1] boolean tensor, True for valid points
        widths_of_interest: [B, k] or [B, k, m+1] tensor of widths for trajectories of interest
        other_trajectories: [B, n, m+1, 2] tensor of n other trajectories per batch
        other_masks: [B, n, m+1] boolean tensor, True for valid points
        other_widths: [B, n] or [B, n, m+1] tensor of widths for other vehicles
        collision_threshold: additional safety margin in meters (default 0.0)
    
    Returns: collision_matrix: [B, k, n] boolean tensor where collision_matrix[b, i, j] = True 
                         if trajectories_of_interest[b, i] collides with other_trajectories[b, j]
    """
    assert trajectories_of_interest.dim() == 4, "trajectories_of_interest must be [B, k, m+1, 2]"
    assert masks_of_interest.dim() == 3, "masks_of_interest must be [B, k, m+1]"
    assert other_trajectories.dim() == 4, "other_trajectories must be [B, n, m+1, 2]"
    assert other_masks.dim() == 3, "other_masks must be [B, n, m+1]"
    assert trajectories_of_interest.shape[0] == other_trajectories.shape[0], f"Batch sizes must match, got {trajectories_of_interest.shape[0]} and {other_trajectories.shape[0]}"
    assert trajectories_of_interest.shape[2] == other_trajectories.shape[2], f"Trajectory lengths must match got {trajectories_of_interest.shape[2]} and {other_trajectories.shape[2]}"

    masks_of_interest = masks_of_interest.bool()
    B = trajectories_of_interest.shape[0]
    k = trajectories_of_interest.shape[1]
    n = other_trajectories.shape[1]
    
    # Expand dimensions for pairwise comparison
    traj_interest = trajectories_of_interest.unsqueeze(2)  # [B, k, 1, m+1, 2]
    other_traj = other_trajectories.unsqueeze(1)  # [B, 1, n, m+1, 2]
    
    # Compute pairwise distances
    distances = torch.norm(traj_interest - other_traj, dim=-1)  # [B, k, n, m+1]
    
    # Expand masks for pairwise comparison
    mask_interest = masks_of_interest.unsqueeze(2)  # [B, k, 1, m+1]
    mask_other = other_masks.unsqueeze(1)  # [B, 1, n, m+1]
    valid_both = mask_interest & mask_other  # [B, k, n, m+1]
    
    # Compute collision threshold based on vehicle widths
    # Two vehicles collide if their center distance < (width1 + width2) / 2
    
    # Handle widths_of_interest
    if widths_of_interest.dim() == 2:
        # widths_of_interest is [B, k], expand to [B, k, 1, 1]
        widths_interest = widths_of_interest.unsqueeze(2).unsqueeze(3)  # [B, k, 1, 1]
    else:
        # widths_of_interest is [B, k, m+1], expand to [B, k, 1, m+1]
        widths_interest = widths_of_interest.unsqueeze(2)  # [B, k, 1, m+1]
    
    # Handle other_widths
    if other_widths.dim() == 2:
        # other_widths is [B, n], expand to [B, 1, n, 1]
        widths_other = other_widths.unsqueeze(1).unsqueeze(3)  # [B, 1, n, 1]
    else:
        # other_widths is [B, n, m+1], expand to [B, 1, n, m+1]
        widths_other = other_widths.unsqueeze(1)  # [B, 1, n, m+1]
    
    # Compute combined width for all pairs
    combined_width = (widths_interest + widths_other) / 2  # [B, k, n, 1] or [B, k, n, m+1]
    
    # Add safety margin
    effective_threshold = combined_width + collision_threshold  # [B, k, n, 1] or [B, k, n, m+1]
    
    # Check collisions: distance < effective threshold at any valid timestep
    collision_at_timestep = (distances < effective_threshold) & valid_both  # [B, k, n, m+1]
    collision_matrix = collision_at_timestep.any(dim=3)  # [B, k, n]
    
    return collision_matrix


class PathReward(torch.nn.Module):
    def __init__(self, offset=8.0):
        super().__init__()
        self.offset = offset

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

        return {"total_reward":-(total_l2_errors - self.offset)}


class PathRewardWithCollision(torch.nn.Module):
    def __init__(self, offset=8.0):
        super().__init__()
        self.offset = offset

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

        # Check for collision
        traj_width = agent_fut_width.unsqueeze(1).expand(-1, num_modes, -1) # [A, num_modes, ts]
        collision_matrix = check_collision_for_trajectory(
            trajectories_of_interest=traj_preds[..., :2], # [A, num_modes, ts, 2]
            masks_of_interest=target_mask_expanded, # [A, num_modes, ts]
            widths_of_interest=traj_width, # [A, num_modes, ts]
            other_trajectories=other_fut_trajs,
            other_masks=other_fut_masks,
            other_widths=other_fut_widths, # [A, n]
            collision_threshold=0.0
        ) # [A, num_modes, n]

        collision_matrix = collision_matrix.sum(dim=-1)

        total = -(total_l2_errors - self.offset) - collision_matrix * 100
        return {"total_reward": total, "collision_penalty": -collision_matrix * 100, "l2_error_penalty": -(total_l2_errors - self.offset)}

