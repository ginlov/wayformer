import torch

from typing import Tuple

class WayformerLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor], # [A, num_modes, ts, 4], [A, num_modes]
        targets: torch.Tensor # [A, ts, 2]
    ):
        # Find nearest mode to target of each agent
        traj_preds, mode_probs = predictions
        A, num_modes, ts, _ = traj_preds.shape
        targets_expanded = targets.unsqueeze(1).expand(-1, num_modes, -1, -1) # [A, num_modes, ts, 2]
        l2_errors = torch.norm(traj_preds[..., :2] - targets_expanded, dim=-1) # [A, num_modes, ts]
        total_l2_errors = l2_errors.sum(dim=-1) # [A, num_modes]
        best_mode_indices = torch.argmin(total_l2_errors, dim=-1) # [A]

        # Compute log likelihood loss
        log_likelihood_loss = self.cross_entropy_loss(mode_probs, best_mode_indices)

        # Compute log likelihood of the true trajectory under the best mode
        # a mode is represented as (x, y, log_std_x, log_std_y)
        best_mode_preds = traj_preds[torch.arange(A), best_mode_indices] # [A, ts, 4]
        best_mode_log_std = best_mode_preds[..., 2:] # [A, ts, 2]
        best_mode_std = torch.exp(best_mode_log_std) # [A, ts, 2]
        best_mode_mean = best_mode_preds[..., :2] # [A, ts, 2]
        gauss_nll = 0.5 * (((targets - best_mode_mean) / best_mode_std) ** 2 + 2 * best_mode_log_std + torch.log(torch.tensor(2 * torch.pi))) # [A, ts, 2]
        traj_nll_loss = gauss_nll.sum(dim=-1).mean() # scalar

        total_loss = log_likelihood_loss + traj_nll_loss
        return total_loss
