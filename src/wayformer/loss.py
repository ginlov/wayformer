import math
import torch

from typing import Tuple

from src.utils import cal_l2_dist

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

        # Find best mode for each agent
        total_l2_errors = cal_l2_dist(traj_preds[...,:2], targets, target_mask) # [A, num_modes]
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
        with torch.no_grad():
            l2_mean_reg = torch.norm((targets - best_mode_mean) * time_weights.unsqueeze(2), dim=-1).mean() # scalar

        total_loss = log_likelihood_loss + traj_nll_loss
        return {'loss/loss': total_loss,
                'loss/classification_loss': log_likelihood_loss,
                'loss/regression_loss': traj_nll_loss,
                'loss/l2_std_reg_for_tracking': l2_std_reg,
                'loss/l2_mean_reg_for_tracking': l2_mean_reg}
