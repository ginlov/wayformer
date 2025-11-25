import torch
import math

from typing import Tuple

class WayformerLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        targets: torch.Tensor, # [A, ts, 2]
        target_mask: torch.Tensor, # [A, ts]
        predictions: Tuple[torch.Tensor, torch.Tensor] # [A, num_modes, ts, 4], [A, num_modes]
    ):
        target_mask = target_mask.float()
        # Find nearest mode to target of each agent
        traj_preds, mode_probs = predictions
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
        best_mode_indices = torch.argmin(total_l2_errors, dim=-1) # [A]

        # Compute log likelihood loss
        # Get likelihood of the best mode
        bed_mode_likelihoods = mode_probs[torch.arange(A), best_mode_indices] # [A]
        log_likelihood_loss = -torch.log(bed_mode_likelihoods + 1e-9).mean() # scalar

        # Compute log likelihood of the true trajectory under the best mode
        # a mode is represented as (x, y, log_std_x, log_std_y)
        best_mode_preds = traj_preds[torch.arange(A), best_mode_indices] # [A, ts, 4]
        best_mode_log_std = best_mode_preds[..., 2:] # [A, ts, 2]
        best_mode_std = torch.exp(best_mode_log_std) # [A, ts, 2]
        best_mode_mean = best_mode_preds[..., :2] # [A, ts, 2]
        gauss_nll = 0.5 * (((targets - best_mode_mean) / best_mode_std) ** 2 + 2 * best_mode_log_std + torch.log(torch.tensor(2 * torch.pi))) # [A, ts, 2]

        # Apply target mask
        gauss_nll = gauss_nll * target_mask.unsqueeze(-1) # [A, ts, 2]
        traj_nll_loss = gauss_nll.mean(dim=-1).mean() # scalar

        # Weighted L2 std regularization
        # Further timestep should be regularized more heavily
        time_weights = torch.logspace(0.0, math.log(4.0), steps=ts).to(targets.device) # [ts]
        # Apply target mask
        time_weights = time_weights.unsqueeze(0) * target_mask.mean(dim=0) # [A, ts]
        with torch.no_grad():
            # Apply target mask
            l2_std_reg = torch.norm(best_mode_std * time_weights.unsqueeze(2), dim=-1).mean() # scalar

        # Weighted L2 mean regularization
        # Further timestep should be regularized more heavily
        l2_mean_reg = torch.norm((targets - best_mode_mean) * time_weights.unsqueeze(2), dim=-1).mean() # scalar

        total_loss = log_likelihood_loss + traj_nll_loss + 0.1 * l2_mean_reg
        return {'loss/loss': total_loss,
                'loss/classification_loss': log_likelihood_loss,
                'loss/regression_loss': traj_nll_loss,
                'loss/l2_std_reg_for_tracking': l2_std_reg,
                'loss/l2_mean_reg': 0.1*l2_mean_reg}

