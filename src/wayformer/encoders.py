import torch
from typing import Literal
from src.wayformer.utils import init_weights
from src.wayformer.factories import get_multihead_attention, build_positional_embedding

class LateFusionSceneEncoder(torch.nn.Module):
    def __init__(self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
        num_latents: int = 16,
        attention_type: Literal["multi_axis", "latent", "factorized"] = "multi_axis"
    ):
        super().__init__()
        self.agent_hist_encoder = build_encoder(
            d_model,
            nhead,
            dim_feedforward,
            num_layers,
            dropout,
            num_latents,
            attention_type
        )
        self.road_encoder = build_encoder(
            d_model,
            nhead,
            dim_feedforward,
            num_layers,
            dropout,
            num_latents,
            attention_type
        )
        self.agent_int_encoder = build_encoder(
            d_model,
            nhead,
            dim_feedforward,
            num_layers,
            dropout,
            num_latents,
            attention_type
        )
        self.traffic_light_encoder = build_encoder(
            d_model,
            nhead,
            dim_feedforward,
            num_layers,
            dropout,
            num_latents,
            attention_type
        )

    def forward(
        self,
        agent_histories: torch.Tensor, # B x A x T_hist x 1 x D
        agent_interactions: torch.Tensor, # B x A x T_hist x S_i x D
        road_graphs: torch.Tensor, # B x A x 1 x S_r x D
        traffic_lights: torch.Tensor, # B x A x T_hist x S_tls x D
        agent_positional_encodings: torch.Tensor | None =None, # B x A x T_hist x 1 x D
        agent_interaction_positional_encodings: torch.Tensor | None =None, # B x A x 
                                                                            # T_hist x S_i x D
        road_positional_encodings: torch.Tensor | None =None, # B x A x 1 x S_r x D
        traffic_light_positional_encodings: torch.Tensor | None =None # B x A x T_hist x S_tls x D
    ) -> torch.Tensor: # B x A x sequence_length x D
        """
        Forward pass for the LateFusionSceneEncoder.
        Args:
            agent_histories (torch.Tensor): Tensor of agent historical trajectories.
            agent_interactions (torch.Tensor): Tensor of agent interaction features.
            road_graphs (torch.Tensor): Tensor of road graph features.
            traffic_lights (torch.Tensor): Tensor of traffic light features.
        Returns:
            torch.Tensor: The encoded scene features.
        """
        # Reshape inputs to merge batch and agent dimensions
        B, A, T_hist, _, D = agent_histories.shape
        S_r = road_graphs.shape[3]
        S_i = agent_interactions.shape[3]
        S_tls = traffic_lights.shape[3]
        
        # Agent history encoding
        agent_histories = agent_histories.view(B * A, T_hist, D)
        agent_positional_encodings = None if agent_positional_encodings is None\
                                            else agent_positional_encodings.view(B * A, T_hist, D)
        agent_encoding = self.agent_hist_encoder(agent_histories, agent_positional_encodings) # B*A x T_hist x D

        # Road graph encoding
        road_graphs = road_graphs.view(B * A, S_r, D)
        road_positional_encodings = None if road_positional_encodings is None\
                                            else road_positional_encodings.view(B * A, S_r, D)
        road_encoding = self.road_encoder(road_graphs, road_positional_encodings) # B*A x S_r x D

        # Agent interaction encoding
        agent_interactions = agent_interactions.view(B * A, S_i * T_hist, D)
        agent_interaction_positional_encodings = None if agent_interaction_positional_encodings is None \
                                                        else agent_interaction_positional_encodings.view(B * A, S_i * T_hist, D)
        agent_int_encoding = self.agent_int_encoder(
            agent_interactions,
            agent_interaction_positional_encodings
        ) # B*A x (S_i * T_hist) x D

        # Traffic light encoding
        traffic_lights = traffic_lights.view(B * A, S_tls * T_hist, D)
        traffic_light_positional_encodings = None if traffic_light_positional_encodings is None \
                                                    else traffic_light_positional_encodings.view(B * A, S_tls * T_hist, D)
        traffic_light_encoding = self.traffic_light_encoder(
            traffic_lights,
            traffic_light_positional_encodings
        ) # B*A x (S_tls * T_hist) x D
        
        return torch.cat(
            [agent_encoding, road_encoding, agent_int_encoding, traffic_light_encoding], dim=1
        ) # (B * A) x () x D

class EarlyFusionSceneEncoder(torch.nn.Module):
    pass

class HierarchicalFusionSceneEncoder(torch.nn.Module):
    pass

class SceneEncoder(torch.nn.Module):
    """
    Scene Encoder module for processing scene context information.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
        fusion: Literal["late", "early", "hierarchical"] = "late",
        num_latents: int = 16,
        attention_type: Literal["multi_axis", "latent", "factorized"] = "multi_axis"
    ):
        """
        Initializes the SceneEncoder.
        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int): The dimension of the feedforward network model.
            num_layers (int): The number of encoder layers.
            dropout (float): The dropout value.
            fusion (Literal): The fusion method to use ("late", "early", "hierarchical").
            attention_type (Literal): The type of attention mechanism to use.
        """
        super().__init__()
        if fusion not in ["late", "early", "hierarchical"]:
            raise ValueError(f"Invalid fusion method: {fusion}")
        if fusion == "late":
            self.encoder = LateFusionSceneEncoder(
                d_model, nhead, dim_feedforward, num_layers, dropout, num_latents, attention_type
            )
        elif fusion == "early":
            self.encoder = EarlyFusionSceneEncoder(
                d_model, nhead, dim_feedforward, num_layers, dropout, num_latents, attention_type
            )
        else:  # hierarchical
            self.encoder = HierarchicalFusionSceneEncoder(
                d_model, nhead, dim_feedforward, num_layers, dropout, num_latents, attention_type
            )

    def forward(
        self,
        agent_histories: torch.Tensor, # B x A x T_hist x 1 x D
        agent_interactions: torch.Tensor, # B x A x T_hist x S_i x D
        road_graphs: torch.Tensor, # B x A x 1 x S_r x D
        traffic_lights: torch.Tensor, # B x A x T_hist x S_tls x D
        agent_positional_encodings: torch.Tensor | None =None, # B x A x T_hist x 1 x D
        agent_interaction_positional_encodings: torch.Tensor | None =None, # B x A x 
                                                                            # T_hist x S_i x D
        road_positional_encodings: torch.Tensor | None =None, # B x A x 1 x S_r x D
        traffic_light_positional_encodings: torch.Tensor | None =None # B x A x T_hist x S_tls x D

    ) -> torch.Tensor: # B x A x sequence_length x D
        """
        Forward pass for the SceneEncoder.
        Args:
            agent_histories (torch.Tensor): Tensor of agent historical trajectories.
            agent_interactions (torch.Tensor): Tensor of agent interaction features.
            road_graphs (torch.Tensor): Tensor of road graph features.
            traffic_lights (torch.Tensor): Tensor of traffic light features.
        Returns:
            torch.Tensor: The encoded scene features.
        """       
        return self.encoder(
            agent_histories,
            agent_interactions,
            road_graphs,
            traffic_lights,
            agent_positional_encodings, # B x A x T_hist x 1 x D
            agent_interaction_positional_encodings, # B x A x T_hist x S_i x D
            road_positional_encodings, # B x A x 1 x S_r x D
            traffic_light_positional_encodings # B x A x T_hist x S_tls x D
        )

class EncoderLayer(torch.nn.Module):
    """
    Single encoder layer consisting of self-attention and feed-forward network.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the EncoderLayer.
        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super().__init__()
        self.self_attn = get_multihead_attention(d_model, nhead)
        self.conv1 = torch.nn.Conv1d(d_model, dim_feedforward, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(dim_feedforward, d_model, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
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
        Forward pass for the EncoderLayer.
        Args:
            src (torch.Tensor): The input tensor of shape (batch_size, seq_length, d_model).
            positional_encoding (torch.Tensor | None): Positional encoding tensor.
            key_mask (torch.Tensor | None): Mask tensor for attention.
        Returns:
            torch.Tensor: The output tensor after processing.
        """
        # query: B x L_Q x E
        # key: B x L_K x E
        # value: B x L_V x E, L_K = L_V
        # positional_encoding: B x L_Q x E (optional)
        if positional_encoding is not None:
            key_with_pos = key + positional_encoding
            value_with_pos = value + positional_encoding
            attention_output = self.self_attn(query, key_with_pos, value_with_pos, key_mask)[0]
        else:
            attention_output = self.self_attn(query, key, value, key_mask)[0]
        query = query + self.dropout(attention_output)
        query = self.norm1(query)
        # Feedforward with 1x1 conv: (N, S, E) -> (N, E, S) for Conv1d
        query_conv = query.transpose(1, 2)  # N x E x S
        query1 = self.conv2(self.dropout(torch.relu(self.conv1(query_conv))))
        query1 = query1.transpose(1, 2)  # N x S x E
        query = query + self.dropout(query1)
        return self.norm2(query)

class LatentEncoderLayer(torch.nn.Module):
    """
    Latent Encoder Layer with specialized attention mechanism.
    Instead of use input as query, learn a set of latent queries.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_latents: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the LatentEncoderLayer.
        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int): The dimension of the feedforward network model.
            num_latents (int): The number of latent queries to learn.
            dropout (float): The dropout value.
        """
        super().__init__()
        self.latent_queries = torch.nn.Parameter(torch.randn(num_latents, d_model))

        # Initialize encoder as usual
        self.self_attn = get_multihead_attention(d_model, nhead)
        self.conv1 = torch.nn.Conv1d(d_model, dim_feedforward, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(dim_feedforward, d_model, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.apply(init_weights)
        
    def forward(
        self,
        key: torch.Tensor, # B x L_K x D
        value: torch.Tensor, # B x L_V x D
        positional_encoding: torch.Tensor | None =None,
        key_mask: torch.Tensor | None =None,
    ) -> torch.Tensor:
        """
        Forward pass for the LatentEncoderLayer.
        Args:
            key (torch.Tensor): The key tensor of shape (batch_size, seq_length, d_model).
            value (torch.Tensor): The value tensor of shape (batch_size, seq_length, d_model).
            positional_encoding (torch.Tensor | None): Positional encoding tensor.
            key_mask (torch.Tensor | None): Mask tensor for attention.
        Returns:
            torch.Tensor: The output tensor after processing.
        """
        batch_size = key.size(0)
        # Expand latent queries to batch size
        query = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)  # B x num_latents x D
        if positional_encoding is not None:
            key_with_pos = key + positional_encoding
            value_with_pos = value + positional_encoding
            attention_output = self.self_attn(query, key_with_pos, value_with_pos, key_mask)[0]
        else:
            attention_output = self.self_attn(query, key, value, key_mask)[0]
        query = query + self.dropout(attention_output)
        query = self.norm1(query)
        # Feedforward with 1x1 conv: (N, S, E) -> (N, E, S) for Conv1d
        query_conv = query.transpose(1, 2)  # N x E x S
        query1 = self.conv2(self.dropout(torch.relu(self.conv1(query_conv))))
        query1 = query1.transpose(1, 2)  # N x S x E
        query = query + self.dropout(query1)
        return self.norm2(query)

class Encoder(torch.nn.Module):
    """
    Encoder module consisting of multiple EncoderLayer modules.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the Encoder.
        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int): The dimension of the feedforward network model.
            num_layers (int): The number of encoder layers.
            dropout (float): The dropout value.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        src: torch.Tensor, # B x S x D
        positional_encoding: torch.Tensor | None =None, # B x S x D
    ):
        """
        Forward pass for the Encoder.
        Args:
            src (torch.Tensor): The input tensor of shape (batch_size, seq_length, d_model).
            positional_encoding (torch.Tensor | None): Positional encoding tensor.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        output = src
        for layer in self.layers:
            output = layer(output, output, output, positional_encoding)
        return output

class LatentEncoder(torch.nn.Module):
    """
    Latent Encoder module consisting of multiple LatentEncoderLayer modules.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        num_latents: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the LatentEncoder.
        Args:
            d_model (int): The number of expected features in the input.
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int): The dimension of the feedforward network model.
            num_layers (int): The number of encoder layers.
            num_latents (int): The number of latent queries to learn.
            dropout (float): The dropout value.
        """
        super().__init__()
        # First layer is latent encoder layer, others are normal encoder layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            LatentEncoderLayer(d_model, nhead, dim_feedforward, num_latents, dropout)
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            )

    def forward(
        self,
        src: torch.Tensor, # B x S x D
        positional_encoding: torch.Tensor | None =None, # B x S x D
    ):
        """
        Forward pass for the LatentEncoder.
        Args:
            src (torch.Tensor): The input tensor of shape (batch_size, seq_length, d_model).
            positional_encoding (torch.Tensor | None): Positional encoding tensor.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        # Feed through first latent encoder layers
        output = self.layers[0](src, src, positional_encoding=positional_encoding)

        # Feed through remaining encoder num_layers
        for layer in self.layers[1:]:
            output = layer(output, src, src, positional_encoding)
        return output

def build_encoder(
    d_model: int,
    nhead: int,
    dim_feedforward: int,
    num_layers: int,
    dropout: float = 0.1,
    num_latents: int = 16,
    attention_type: Literal["multi_axis", "latent", "factorized"] = "multi_axis"
) -> torch.nn.Module:
    """
    Builds an encoder consisting of multiple EncoderLayer modules.

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        num_layers (int): The number of encoder layers.
        dropout (float): The dropout value.
        attention_type (Literal): The type of attention mechanism to use.

    Returns:
        torch.nn.Module: The constructed encoder module.
    """
    if attention_type == "latent":
        return LatentEncoder(
            d_model,
            nhead,
            dim_feedforward,
            num_layers,
            num_latents,
            dropout
        )
    return Encoder(
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        dropout
    )

