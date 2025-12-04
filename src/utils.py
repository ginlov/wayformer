import torch
from typing import Dict

def init_weights(module):
    """
    Initialize the weights of the model.

    Args:
        module (torch.nn.Module): The module to initialize.
    """
    xavier_uniform_ = [
        torch.nn.Conv2d,
        torch.nn.Linear,
        torch.nn.ConvTranspose2d,
        torch.nn.Conv1d,
        torch.nn.ConvTranspose1d,
    ]
    if any(isinstance(module, m) for m in xavier_uniform_):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.constant_(module.bias, 0)
        torch.nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0, std=0.01)

@torch.no_grad
def cal_param_norm(model: torch.nn.Module) -> Dict:
    """Compute parameter L2 norm

    Args:
        model (torch.nn.Module): DETR model

    Returns:
        Dict: norm of each layer in DETR model
    """
    param_norm = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            norm = torch.norm(param, p=2).item()
            param_norm[f"param_norm/{name}"] = norm

    return param_norm


@torch.no_grad
def cal_grad_norm(model: torch.nn.Module) -> Dict:
    """Computer gradient norm

    Args:
        optimizer (torch.optim.Optimizer): optimizer

    Returns:
        Dict: gradient norm dictionary
    """
    grad_norm = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = torch.norm(param.grad, p=2).item()
            grad_norm[f"grad_norm/{name}"] = norm
    return grad_norm

def cal_l2_dist(
    pred_trajs: torch.Tensor, # (B, num_modes, T, 2)
    gt_trajs: torch.Tensor, # (B, T, 2)
    mask: torch.Tensor, # (B, T)
    step_wise: bool = False
):
    """
    Calculate L2 distance between predicted trajectories and ground truth trajectories.

    Args:
        pred_trajs (torch.Tensor): Predicted trajectories of shape (B, num_modes, T, 2).
        gt_trajs (torch.Tensor): Ground truth trajectories of shape (B, T, 2).
        mask (torch.Tensor): Mask tensor of shape (B, T) indicating valid time steps.

    Returns:
        torch.Tensor: L2 distance of shape (B, num_modes).
    """
    B, num_modes, T, _ = pred_trajs.shape

    # Expand ground truth trajectories to match predicted trajectories shape
    gt_trajs_expanded = gt_trajs.unsqueeze(1).expand(-1, num_modes, -1, -1)  # (B, num_modes, T, 2)
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(-1, num_modes, -1, -1)  # (B, num_modes, T, 1)

    # Calculate squared differences
    # Use torch.norm
    l2_dist = torch.norm((pred_trajs - gt_trajs_expanded) * mask_expanded, dim=-1)  # (B, num_modes, T)

    if step_wise:
        return l2_dist  # (B, num_modes, T)

    # Sum over time dimension
    l2_dist = l2_dist.sum(dim=-1)  # (B, num_modes)

    return l2_dist

