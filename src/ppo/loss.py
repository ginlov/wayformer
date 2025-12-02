import torch

class PPOLoss(torch.nn.Module):
    def __init__(self, clip_param: float, value_loss_coef: float, entropy_coef: float):
        super().__init__()
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def forward(
        self,
        rewards: torch.Tensor, # [B, T]
        values: torch.Tensor, # [B, T + 1]
        ref_probs: torch.Tensor, # [B, T]
        old_probs: torch.Tensor, # [B, T]
        curr_probs: torch.Tensor, # [B, T]
        masks: torch.Tensor, # [B, T]
    ):
        advantages, returns = compute_gae(rewards, values, masks, gamma=0.99, lam=0.95)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # Compute ratio
        ratios = curr_probs / (old_probs + 1e-9)  # [B, T]

        # Compute surrogate losses
        surr1 = ratios * advantages  # [B, T]
        surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages  # [B, T]

        # PPO loss
        ppo_loss = -torch.min(surr1, surr2).mean()  # scalar

        # Value loss
        value_loss = (returns - values[:, :-1]).pow(2).mean()  # scalar

        # Entropy loss
        entropy_loss = - (curr_probs * torch.log(curr_probs + 1e-9)).mean()  # scalar

        total_loss = ppo_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

        return {
            "loss/loss": total_loss,
            "loss/ppo_loss": ppo_loss,
            "loss/value_loss": self.value_loss_coef * value_loss,
            "loss/entropy_loss": self.entropy_coef * entropy_loss
        }


def compute_gae(
    rewards: torch.Tensor, # [B, T]
    values: torch.Tensor, # [B, T + 1]
    masks: torch.Tensor, # [B, T]
    gamma: float, # discount factor
    lam: float # GAE lambda
):
    B, T = rewards.size()
    advantages = torch.zeros(B, T, device=rewards.device)
    last_gae_lam = 0
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values[:, t + 1] * masks[:, t] - values[:, t]
        advantages[:, t] = last_gae_lam = delta + gamma * lam * masks[:, t] * last_gae_lam
    returns = advantages + values[:, :-1]
    return advantages, returns

