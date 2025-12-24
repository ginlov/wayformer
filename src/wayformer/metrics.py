import torch
from typing import Dict, Any
from src.utils import cal_l2_dist

class WaymoMetrics(torch.nn.Module):
    def __init__(
        self,
        miss_rate_threshold: float = 2.0,
        collision_threshold: float = 0.3,
    ):
        super().__init__()
        self.miss_rate_threshold = miss_rate_threshold
        self.collision_threshold = collision_threshold

    def _min_distance_error(
        self,
        pred_trajs,
        gt_trajs,
        gt_masks,
    ) -> torch.Tensor:
        """
        Compute the minimum distance error between predicted trajectories and ground truth trajectories
        at each timestep.

        Args:
            pred_trajs (torch.Tensor): Predicted trajectories of shape (B, N, T, 2)
            gt_trajs (torch.Tensor): Ground truth trajectories of shape (B, T, 2)

        Returns:
            torch.Tensor: Minimum distance error of shape (B, T)
        """
        # Compute L2 distance between predicted and ground truth trajectories
        dists = cal_l2_dist(pred_trajs, gt_trajs, gt_masks, step_wise=True)  # (B, N, T)

        # Find the minimum distance error across all predicted trajectories
        min_dists, _ = torch.min(dists, dim=1)  # (B, T)

        # Masking out invalid timesteps
        # gt_masks is guaranteed to be not None
        # I want value of min_dists equal to -1 where gt_masks is False
        min_dists = min_dists * gt_masks.float() + (-1.0) * (~gt_masks).float()

        return min_dists

    def _miss_rate(
        self,
        pred_trajs,
        gt_trajs,
        gt_masks,
    ) -> torch.Tensor:
        """
        Compute the miss rate between predicted trajectories and ground truth trajectories
        at each timestep based on a distance threshold.

        Args:
            pred_trajs (torch.Tensor): Predicted trajectories of shape (B, N, T, 2)
            gt_trajs (torch.Tensor): Ground truth trajectories of shape (B, T, 2)
            threshold (float): Distance threshold to consider a miss

        Returns:
            torch.Tensor: Miss rate of shape (B, T)
        """
        # Compute L2 distance between predicted and ground truth trajectories
        dists = cal_l2_dist(pred_trajs, gt_trajs, gt_masks, step_wise=True)  # (B, N, T)

        # Determine misses based on the threshold
        misses = (dists > self.miss_rate_threshold).all(dim=1).float()  # (B, T)

        # Masking out invalid timesteps
        # gt_masks is guaranteed to be not None
        # I want value of misses equal to -1 where gt_masks is False
        misses = misses * gt_masks.float() + (-1.0) * (~gt_masks).float()

        return misses

    def _min_average_distance_error(
        self,
        pred_trajs,
        gt_trajs,
        gt_masks
    ) -> torch.Tensor:
        """
        Compute the minimum average distance error between predicted trajectories and ground truth trajectories
        over the entire trajectory.

        Args:
            pred_trajs (torch.Tensor): Predicted trajectories of shape (B, N, T, 2)
            gt_trajs (torch.Tensor): Ground truth trajectories of shape (B, T, 2)

        Returns:
            torch.Tensor: Minimum average distance error of shape (B,)
        """
        # Compute L2 distance between predicted and ground truth trajectories
        dists = cal_l2_dist(pred_trajs, gt_trajs, gt_masks, step_wise=True)  # (B, N, T)

        # Compute average distance error over the trajectory
        # Masking out invalid timesteps
        # gt_masks is guaranteed to be not None
        valid_lengths = gt_masks.sum(dim=1).unsqueeze(1)  # (B, 1)
        dists = dists * gt_masks.unsqueeze(1).float()  # (B, N, T)
        sum_dists = dists.sum(dim=2)  # (B, N)
        avg_dists = sum_dists / valid_lengths  # (B, N)

        # Find the minimum average distance error across all predicted trajectories
        min_avg_dists, _ = torch.min(avg_dists, dim=1)  # (B,)

        return min_avg_dists

    def _mean_average_precision(
        self,
        pred_trajs,
        gt_trajs
    ) -> torch.Tensor | None:
        """
        Compute the mean average precision between predicted trajectories and ground truth
        trajectories over the entire trajectory.

        Args:
            pred_trajs (torch.Tensor): Predicted trajectories of shape (B, N, T, 2)
            gt_trajs (torch.Tensor): Ground truth trajectories of shape (B, T, 2)

        Returns:
            torch.Tensor: Minimum average precision of shape (B,)
        """
        raise NotImplementedError("Mean Average Precision is not implemented yet. Need to partition\
                                  the trajectories into different behavioral buckets.")

    def _overlap(
        self,
        trajectories_of_interest: torch.Tensor, # [B, 1, T, 2]
        masks_of_interest: torch.Tensor,         # [B, 1, T]
        widths_of_interest: torch.Tensor,        # [B, 1] or [B, 1, T] - width of vehicles
        other_trajectories: torch.Tensor,        # [B, m, T, 2]
        other_masks: torch.Tensor,               # [B, m, T]
        other_widths: torch.Tensor,              # [B, m] or [B, m, T] - width of vehicles
    ) -> torch.Tensor:
        """
        Check overlaps between n trajectories of interest and m other trajectories in batch.
        
        Args:
            trajectories_of_interest: [B, 1, T, 2] tensor of n trajectories to check per batch
            masks_of_interest: [B, 1, T] boolean tensor, True for valid points
            widths_of_interest: [B, 1] or [B, 1, T] tensor of widths for trajectories of interest
            other_trajectories: [B, m, T, 2] tensor of m other trajectories per batch
            other_masks: [B, m, T] boolean tensor, True for valid points
            other_widths: [B, m] or [B, m, T] tensor of widths for other vehicles

        Returns: overlap_matrix: [B, n, m] boolean tensor where overlap_matrix[b, i, j] = True 
                             if trajectories_of_interest[b, i] overlaps with other_trajectories[b, j]
        """
        collision_matrix = collision_per_timestep(
            trajectories_of_interest,
            masks_of_interest,
            widths_of_interest,
            other_trajectories,
            other_masks,
            other_widths,
            collision_threshold=self.collision_threshold
        )  # [B, n, m, T]
        overlap_matrix = collision_matrix.any(dim=-2).squeeze(1)  # [B, T]
        overlap = overlap_matrix.sum(dim=-1) / overlap_matrix.shape[-1]
        return overlap

    def _min_fde(
        self,
        pred_trajs,
        gt_trajs,
        gt_masks
    ):
        """
        Compute the minimum final distance error between predicted trajectories and ground truth trajectories
        at the final timestep.

        Args:
            pred_trajs (torch.Tensor): Predicted trajectories of shape (B, N, T, 2)
            gt_trajs (torch.Tensor): Ground truth trajectories of shape (B, T, 2)

        Returns:
            torch.Tensor: Minimum final distance error of shape (B,)
        """
        B, N, T, _ = pred_trajs.shape

        # Expand ground truth trajectories to match predicted trajectories shape
        gt_trajs_expanded = gt_trajs.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, T, 2)

        # Checking where is the last valid timestep for each batch
        # Should be the last index where gt_masks is True
        # Not summing along dim=1 since there could be some False values in between
        idx = torch.where(
            gt_masks,
            torch.arange(gt_masks.size(1), device=gt_masks.device).unsqueeze(0),
            -1
        )

        last_idx = idx.max(dim=1).values    # shape [B]

        # Compute L2 distance at the final timestep
        final_dists = torch.norm(
            pred_trajs[
                torch.arange(B, device=pred_trajs.device).unsqueeze(1),
                torch.arange(N, device=pred_trajs.device).unsqueeze(0),
                last_idx.unsqueeze(1),
                :]-
            gt_trajs_expanded[
                torch.arange(B, device=pred_trajs.device).unsqueeze(1),
                torch.arange(N, device=pred_trajs.device).unsqueeze(0),
                last_idx.unsqueeze(1),
                :],
            dim=-1
        )  # (B, N)

        # Find the minimum final distance error across all predicted trajectories
        min_final_dists, _ = torch.min(final_dists, dim=1)  # (B,)

        return min_final_dists
    
    def _brier_min_fde(
        self,
        pred_trajs,
        gt_trajs,
        pred_probs,
        gt_masks
    ):
        """
        Compute the Brier Minimum Final Distance Error between predicted trajectories and ground truth trajectories
        at the final timestep, weighted by predicted probabilities.

        Args:
            pred_trajs (torch.Tensor): Predicted trajectories of shape (B, N, T, 2)
            gt_trajs (torch.Tensor): Ground truth trajectories of shape (B, T, 2)
            pred_probs (torch.Tensor): Predicted probabilities of shape (B, N)

        Returns:
            torch.Tensor: Brier Minimum Final Distance Error of shape (B,)
        """
        B, N, T, _ = pred_trajs.shape

        # Expand ground truth trajectories to match predicted trajectories shape
        gt_trajs_expanded = gt_trajs.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, T, 2)

        # Checking where is the last valid timestep for each batch
        # Should be the last index where gt_masks is True
        # Not summing along dim=1 since there could be some False values in between
        idx = torch.where(
            gt_masks,
            torch.arange(gt_masks.size(1), device=gt_masks.device).unsqueeze(0),
            -1
        )

        last_idx = idx.max(dim=1).values    # shape [B]

        # Compute L2 distance at the final timestep
        final_dists = torch.norm(
            pred_trajs[
                torch.arange(B, device=pred_trajs.device).unsqueeze(1),
                torch.arange(N, device=pred_trajs.device).unsqueeze(0),
                last_idx.unsqueeze(1),
                :] - 
            gt_trajs_expanded[
                torch.arange(B, device=pred_trajs.device).unsqueeze(1),
                torch.arange(N, device=pred_trajs.device).unsqueeze(0),
                last_idx.unsqueeze(1),
                :],
            dim=-1
        )  # (B, N)

        # Find the minimum weighted final distance error across all predicted trajectories
        min_final_dists, best_indices = torch.min(final_dists, dim=1)  # (B,)

        # Brier
        brier = (1 - pred_probs[torch.arange(B), best_indices]) ** 2  # (B,)
        brier_min_final_dists = min_final_dists + brier  # (B,)

        return brier_min_final_dists

    @torch.no_grad()
    def forward(
        self,
        data_batch,
        output
    ) -> Dict[str, Any]:
        """
        Compute Waymo metrics given a data batch and model output.
        """
        metrics = {}
        T = data_batch['label_pos'].shape[1]
        min_distance_error = self._min_distance_error(
            output[0][:, :, :, :2],
            data_batch['label_pos'],
            data_batch['label_mask']
        ) # [B, T]

        miss_rate = self._miss_rate(
            output[0][:, :, :, :2],
            data_batch['label_pos'],
            data_batch['label_mask']
        ) # [B, T]

        min_average_distance_error = self._min_average_distance_error(
            output[0][:, :, :, :2],
            data_batch['label_pos'],
            data_batch['label_mask']
        ) # [B]

        most_likely_indices = torch.argmax(output[1], dim=1)  # [B]
        most_likely_trajs = output[0][
            torch.arange(output[0].shape[0], device=output[0].device),
            most_likely_indices,
            :, :2
        ].unsqueeze(1)  # [B, 1, T, 2]

        overlap = self._overlap(
            most_likely_trajs,
            data_batch['label_mask'].unsqueeze(1),
            data_batch['agent_future_width'].unsqueeze(1),
            data_batch['other_agents_future_pos'],
            data_batch['other_agents_future_mask'],
            data_batch['other_agents_future_width'],
        ) # [B]

        min_fde = self._min_fde(
            output[0][:, :, :, :2],
            data_batch['label_pos'],
            data_batch['label_mask']
        ) # [B]

        brier_min_fde = self._brier_min_fde(
            output[0][:, :, :, :2],
            data_batch['label_pos'],
            output[1],
            data_batch['label_mask']
        ) # [B]

        for i in range(T):
            # format timestep to 2 ditits
            # filter out minus one values in min_distance_error and miss_rate
            valid_min_distance_error = min_distance_error[:, i][min_distance_error[:, i] >= 0]
            valid_miss_rate = miss_rate[:, i][miss_rate[:, i] >= 0]
            if valid_min_distance_error.numel() > 0:
                metrics[f'min_distance_error_t{str(i+1).zfill(2)}'] = valid_min_distance_error.mean().item()

            if valid_miss_rate.numel() > 0:
                metrics[f'miss_rate_t{str(i+1).zfill(2)}'] = valid_miss_rate.mean().item()
            
        # Filter out minus one values in min_average_distance_error
        valid_min_average_distance_error = min_average_distance_error[min_average_distance_error >= 0]
        valid_overlap = overlap[overlap >= 0]
        valid_min_fde = min_fde[min_fde >= 0]
        valid_brier_min_fde = brier_min_fde[brier_min_fde >= 0]

        if valid_min_average_distance_error.numel() > 0:
            metrics['min_average_distance_error'] = valid_min_average_distance_error.mean().item()
        if valid_overlap.numel() > 0:
            metrics['overlap'] = valid_overlap.mean().item()
        if valid_min_fde.numel() > 0:
            metrics['min_fde'] = valid_min_fde.mean().item()
        if valid_brier_min_fde.numel() > 0:
            metrics['brier_min_fde'] = valid_brier_min_fde.mean().item()
        return metrics

def collision_per_timestep(
    trajectories_of_interest: torch.Tensor,  # [B, n, T, 2]
    masks_of_interest: torch.Tensor,        # [B, n, T]
    widths_of_interest: torch.Tensor,        # [B, n] or [B, n, T] - width of vehicles for trajectories of interest
    other_trajectores: torch.Tensor,       # [B, m, T, 2]
    other_masks: torch.Tensor,              # [B, m, T]
    other_widths: torch.Tensor,             # [B, m] or [B, m, T] - width of vehicles for other trajectories
    collision_threshold: float = 0.0         # additional safety margin beyond vehicle widths
) -> torch.Tensor: # [B, n, m, T]
    """
    Check collisions between n trajectories of interest and m other trajectories in batch per timestep.
    
    Args:
        trajectories_of_interest: [B, n, T, 2] tensor of n trajectories to check per batch
        masks_of_interest: [B, n, T] boolean tensor, True for valid points
        widths_of_interest: [B, n] or [B, n, T] tensor of widths for trajectories of interest
        other_trajectores: [B, m, T, 2] tensor of m other trajectories per batch
        other_masks: [B, m, T] boolean tensor, True for valid points
        other_widths: [B, m] or [B, m, T] tensor of widths for other vehicles
        collision_threshold: additional safety margin in meters (default 0.0)

    Returns: collision_matrix: [B, n, m, T] boolean tensor where collision_matrix[b, i, j, t] = True 
                         if trajectories_of_interest[b, i] collides with other_trajectores[b, j] at timestep t
    """
    assert trajectories_of_interest.dim() == 4, "trajectories_of_interest must be [B, n, T, 2]"
    assert masks_of_interest.dim() == 3, "masks_of_interest must be [B, n, T]"
    assert other_trajectores.dim() == 4, "other_trajectores must be [B, m, T, 2]"
    assert other_masks.dim() == 3, "other_masks must be [B, m, T]"
    assert trajectories_of_interest.shape[0] == other_trajectores.shape[0], f"Batch sizes must match, got {trajectories_of_interest.shape[0]} and {other_trajectores.shape[0]}"
    assert trajectories_of_interest.shape[2] == other_trajectores.shape[2], f"Trajectory lengths must match got {trajectories_of_interest.shape[2]} and {other_trajectores.shape[2]}"

    masks_of_interest = masks_of_interest.bool()
    
    # Expand dimensions for pairwise comparison
    traj_interest = trajectories_of_interest.unsqueeze(2)  # [B, n, 1, T, 2]
    other_traj = other_trajectores.unsqueeze(1)  # [B, 1, m, T, 2]
    
    # Compute pairwise distances
    distances = torch.norm(traj_interest - other_traj, dim=-1)  # [B, n, m, T]
    
    # Expand masks for pairwise comparison
    mask_interest = masks_of_interest.unsqueeze(2)  # [B, n, 1, T]
    mask_other = other_masks.unsqueeze(1)  # [B, 1, m, T]
    valid_both = mask_interest & mask_other  # [B, n, m, T]
    
    # Compute collision threshold based on vehicle widths
    # Two vehicles collide if their center distance < (width1 + width2) / 2
    # Handle widths_of_interest
    if widths_of_interest.dim() == 2:
        # widths_of_interest is [B, n], expand to [B, n, 1, 1]
        widths_interest = widths_of_interest.unsqueeze(2).unsqueeze(3)  # [B, n, 1, 1]
    else:
        # widths_of_interest is [B, n, T], expand to [B, n, 1, T]
        widths_interest = widths_of_interest.unsqueeze(2)  # [B, n, 1, T]

    # Handle other_widths
    if other_widths.dim() == 2:
        # other_widths is [B, m], expand to [B, 1, m, 1]
        widths_other = other_widths.unsqueeze(1).unsqueeze(3)  # [B, 1, m, 1]
    else:
        # other_widths is [B, m, T], expand to [B, 1, m, T]
        widths_other = other_widths.unsqueeze(1)  # [B, 1, m, T]

    # Compute combined width for all pairwises
    combined_width = (widths_interest + widths_other) / 2  # [B, n, m, 1] or [B, n, m, T]
    # Add safety margin
    effective_threshold = combined_width + collision_threshold  # [B, n, m, 1] or [B, n, m, T]
    # Check collisions: distance < effective threshold at any valid timestep
    collision_matrix = (distances < effective_threshold) & valid_both  # [B, n, m, T]
    return collision_matrix
