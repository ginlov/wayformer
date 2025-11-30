import torch
from cvrunner.utils.logger import get_cv_logger

logger = get_cv_logger()

class GRPOLoss(torch.nn.Module):
    def __init__(self, epsilon: float = 0.2, beta: float = 0.01):
        super().__init__()
        self.epsilon = epsilon
        self.beta = beta
        pass

    def compute_advantage(
        self,
        rewards: torch.Tensor,  # [A, num_modes]
    ) -> torch.Tensor:
        mean = rewards.mean(dim=1, keepdim=True)  # [A, 1]
        std = rewards.std(dim=1, keepdim=True) + 1e-9  # [A, 1]
        adv = (rewards - mean) / std  # [A, num_modes]
        return adv

    def forward(
        self,
        rewards: torch.Tensor, # [A, num_modes]
        ref_probs: torch.Tensor, # [A, num_modes]
        old_probs: torch.Tensor,  # [A, num_modes]
        curr_probs: torch.Tensor  # [A, num_modes]
    ):
        advantages = self.compute_advantage(rewards)  # [A, num_modes]

        # Compute ratio
        ratios = curr_probs/(old_probs + 1e-9)  # [A, num_modes]
        # Compute surrogate losses
        surr1 = ratios * advantages  # [A, num_modes]
        surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages  # [A, num_modes]
        # PPO loss
        ppo_loss = -torch.min(surr1, surr2).mean()  # scalar

        # KL divergence loss
        # all probabilities are not in log space
        kl_div = ref_probs * (torch.log(ref_probs + 1e-9) - torch.log(curr_probs + 1e-9))  # [A, num_modes]
        kl_div = kl_div.sum(dim=1).mean()  # scalar

        total_loss = ppo_loss + self.beta * kl_div
        return {"loss/loss": total_loss, "loss/ppo_loss": ppo_loss, "loss/kl_div": self.beta * kl_div}
