import torch
from src.wayformer.utils import init_weights
from src.wayformer.factories import get_multihead_attention, build_positional_embedding

class DecoderLayer(torch.nn.Module):
    """
    Single decoder layer consisting of self-attention, cross-attention, and feed-forward network.
    """
    def __init__(
        self,
        d_model:int,
        nhead:int,
        dim_feedforward:int,
        dropout:float =0.1,
    ):
        super().__init__()
        self.self_attn = get_multihead_attention(d_model, nhead)
        self.cross_attn = get_multihead_attention(d_model, nhead)
        self.conv1 = torch.nn.Conv1d(d_model, dim_feedforward, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(dim_feedforward, d_model, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.apply(init_weights)

    def forward(
        self,
        query: torch.Tensor, # B x L_Q x D
        key: torch.Tensor, # B x L_K x D
        value: torch.Tensor, # B x L_V x D
        positional_encoding: torch.Tensor | None =None,
        key_mask: torch.Tensor | None =None,
    ) -> torch.Tensor:
        """
        Forward pass for the DecoderLayer.
        Args:
            query (torch.Tensor): The query tensor of shape (batch_size, seq_length, d_model).
            key (torch.Tensor): The key tensor of shape (batch_size, seq_length, d_model).
            value (torch.Tensor): The value tensor of shape (batch_size, seq_length, d_model).
            positional_encoding (torch.Tensor | None): Positional encoding tensor.
            key_mask (torch.Tensor | None): Mask tensor for attention.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        # Self-attention
        self_attention_output = self.self_attn(query, query, query)[0]
        query = query + self.dropout(self_attention_output)
        query = self.norm1(query)

        # Cross-attention
        if positional_encoding is not None:
            key_with_pos = key + positional_encoding
            value_with_pos = value + positional_encoding
            cross_attention_output = self.cross_attn(query, key_with_pos, value_with_pos, key_mask)[0]
        else:
            cross_attention_output = self.cross_attn(query, key, value, key_mask)[0]
        query = query + self.dropout(cross_attention_output)
        query = self.norm2(query)

        # Feedforward with 1x1 conv: (N, S, E) -> (N, E, S) for Conv1d
        query_conv = query.transpose(1, 2)  # N x E x S
        query1 = self.conv2(self.dropout(torch.relu(self.conv1(query_conv))))
        query1 = query1.transpose(1, 2)  # N x S x E
        query = query + self.dropout(query1)
        return self.norm3(query)

class TrajectoryDecoder(torch.nn.Module):
    """
    Trajectory Decoder module for generating future trajectories based on encoded features.
    """
    def __init__(
        self,
        num_modes: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the TrajectoryDecoder.
        Args:
            num_modes (int): The number of trajectory modes to predict.
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int): The dimension of the feedforward network model.
            num_layers (int): The number of decoder layers.
            dropout (float): The dropout value.
        """
        super().__init__()
        self.mode_query = torch.nn.Parameter(torch.empty(num_modes, d_model))
        torch.nn.init.orthogonal_(self.mode_query)
        self.layers = torch.nn.ModuleList([
            DecoderLayer(
                d_model, nhead, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        memory: torch.Tensor, # B x S_mem x D
        positional_encoding: torch.Tensor | None =None, # B x S_mem x D
    ) -> torch.Tensor: # B x num_modes x D
        """
        Forward pass for the TrajectoryDecoder.
        Args:
            memory (torch.Tensor): The encoded features from the encoder.
            positional_encoding (torch.Tensor | None): Positional encoding tensor.

        Returns:
            torch.Tensor: The decoded trajectory modes.
        """        
        batch_size = memory.size(0)
        # Expand mode queries to batch size
        query = self.mode_query.unsqueeze(0).expand(batch_size, -1, -1)  # B x num_modes x D
        output = query
        for layer in self.layers:
            output = layer(output, memory, memory, positional_encoding)
        return output

