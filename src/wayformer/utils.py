import torch

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

