import torch
from typing import Any, Dict

class GRPOMetrics(torch.nn.Module):
    def __init__(
        self,
        waymo_metrics: torch.nn.Module
    ):
        super().__init__()
        self.waymo_metrics = waymo_metrics

    @torch.no_grad
    def forward(
        self,
        data_batch,
        output
    ) -> Dict[str, Any]:
        metrics = self.waymo_metrics(
            data_batch,
            output
        )

        # Compute percentage of mismatches prediction
        out_traj = output[0] # [A, num_modes, ts, 4]
        gt_traj = data_batch['label_pos'] # [A, ts, 2]
        gt_mask = data_batch['label_mask'] # [A, ts]
        # Compute the exact best mode (lowest L2 error to ground-truth)
        distances = torch.norm(
            out_traj[:, :, :, :2] - gt_traj.unsqueeze(1), dim=-1
        ) # [A, num_modes, ts]
        distances = distances * gt_mask.unsqueeze(1)  # Mask invalid timesteps
        sum_distances = distances.sum(dim=-1)  # [A, num_modes]

        sorted_modes_dis = torch.argsort(sum_distances, dim=-1)  # [A, num_modes]
        sorted_modes_probs = torch.argsort(-output[1], dim=-1)  # [A, num_modes]

        mismatches = (sorted_modes_dis[:, 0] != sorted_modes_probs[:, 0]).float()  # [A]
        mismatch_rate = mismatches.mean().item()

        # Top 2 mismatches is the proportion of agents whose chosen mode (highest prob)
        # is not in the top 2 closest modes (lowest L2 error)
        top2_modes_dis = sorted_modes_dis[:, :2]  # [A, 2]
        top2_chosen = (sorted_modes_probs[:, 0].unsqueeze(1) == top2_modes_dis).any(dim=-1).float()  # [A]
        top2_mismatch_rate = (1.0 - top2_chosen).mean().item()

        # Average best trajectory probability
        prob_best_traj = torch.gather(
            output[1], 1, sorted_modes_dis[:, 0:1]
        ).squeeze(-1)  # [A]
        avg_prob_best_traj = prob_best_traj.mean().item()

        # Average top 2 trajectory probability
        prob_top2_traj = torch.gather(
            output[1], 1, sorted_modes_dis[:, :2]
        )  # [A, 2]
        avg_prob_top2_traj = prob_top2_traj.mean().item()

        # Average rank of the best trajectory
        rank_best_traj = (sorted_modes_probs == sorted_modes_dis[:, 0:1]).nonzero()[:, 1].float() + 1.0  # [A]
        avg_rank_best_traj = rank_best_traj.mean().item()

        # Average rank of the top 2 trajectories
        ranks_top2_traj = []
        for i in range(2):
            rank_topi_traj = (sorted_modes_probs == sorted_modes_dis[:, i:i+1]).nonzero()[:, 1].float() + 1.0  # [A]
            ranks_top2_traj.append(rank_topi_traj)
        ranks_top2_traj = torch.stack(ranks_top2_traj, dim=-1)  # [A, 2]
        avg_rank_top2_traj = ranks_top2_traj.mean().item()
        
        metrics.update({
            "mismatch_rate": mismatch_rate,
            "top2_mismatch_rate": top2_mismatch_rate,
            "avg_prob_best_traj": avg_prob_best_traj,
            "avg_prob_top2_traj": avg_prob_top2_traj,
            "avg_rank_best_traj": avg_rank_best_traj,
            "avg_rank_top2_traj": avg_rank_top2_traj
        })
        return metrics
