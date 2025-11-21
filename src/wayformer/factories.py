import torch

def get_multihead_attention(d_model, nhead):
    """
    Returns a multihead attention module.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.

    Returns:
        torch.nn.Module: The multihead attention module.
    """
    return torch.nn.MultiheadAttention(
        embed_dim=d_model, num_heads=nhead, batch_first=True
    )

def build_positional_embedding(
    temporal_dimension: int,
    spatial_dimension: int,
    embedding_dim: int
):
    """
    Build a learnable positional embedding layer.

    Args:
        max_seq_len (int): Maximum sequence length.
        embedding_dim (int): Dimension of the embedding vectors.

    Returns:
        torch.nn.Embedding: Learnable positional embedding layer.
    """
    return torch.nn.Parameter(torch.zeros(temporal_dimension, spatial_dimension, embedding_dim))

