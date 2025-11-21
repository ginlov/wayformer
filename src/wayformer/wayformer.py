import torch

from typing import Type
from src.wayformer.encoders import SceneEncoder, build_encoder 
from src.wayformer.decoders import TrajectoryDecoder
from src.wayformer.factories import build_positional_embedding
from src.wayformer.config import DatasetConfig

class Wayformer(torch.nn.Module):
    """
    The Wayformer model for trajectory prediction.
    Combines a SceneEncoder and a TrajectoryDecoder.
    1. The SceneEncoder processes the input scene data to produce a latent representation.
    2. The TrajectoryDecoder takes this latent representation and generates predicted 
    trajectories for agents in the scene.
    3. Projection layers and positional encoders are used to adapt input features to 
    the model's expected dimensions.
    4. The model is designed to handle various input modalities, including agent history,
    interactions, road information, and traffic light states.
    """
    def __init__(
        self,
        encoder: SceneEncoder,
        decoder: TrajectoryDecoder,
        gmm_likelihood_projection: torch.nn.Module,
        gmm_param_projection: torch.nn.Module,
        agent_projection: torch.nn.Module,
        agent_inter_projection: torch.nn.Module,
        road_projection: torch.nn.Module,
        traffic_light_projection: torch.nn.Module,
        agent_pos_encoder: torch.nn.Parameter | None = None,
        agent_inter_pos_encoder: torch.nn.Parameter | None = None,
        road_pos_encoder: torch.nn.Parameter | None = None,
        trafic_light_pos_encoder: torch.nn.Parameter | None = None,
    ):
        """
        Initializes the Wayformer model with the given components.

        Args:
            encoder (SceneEncoder): The scene encoder module.
            decoder (TrajectoryDecoder): The trajectory decoder module.
            agent_projection (torch.nn.Module): Projection layer for agent features.
            agent_inter_projection (torch.nn.Module): Projection layer for agent interaction features.
            road_projection (torch.nn.Module): Projection layer for road features.
            traffic_light_projection (torch.nn.Module): Projection layer for traffic light features.
            agent_pos_encoder (torch.nn.Module, optional): Positional encoder for agent features.
            agent_inter_pos_encoder (torch.nn.Module, optional): Positional encoder for agent interaction features.
            road_pos_encoder (torch.nn.Module, optional): Positional encoder for road features.
            trafic_light_pos_encoder (torch.nn.Module, optional): Positional encoder for traffic light features.
        """
        super().__init__()
        self.encoder: SceneEncoder = encoder
        self.decoder: TrajectoryDecoder = decoder
        self.agent_projection = agent_projection
        self.agent_inter_projection = agent_inter_projection
        self.road_projection = road_projection
        self.traffic_light_projection = traffic_light_projection
        self.agent_pos_encoder = agent_pos_encoder
        self.agent_inter_pos_encoder = agent_inter_pos_encoder
        self.road_pos_encoder = road_pos_encoder
        self.trafic_light_pos_encoder = trafic_light_pos_encoder

        # Project GMM likelihoods
        self.gmm_likelihood_projection = gmm_likelihood_projection
        self.gmm_param_projection = gmm_param_projection

    def forward(
        self,
        agent_hist: torch.Tensor, # [B, A, T, 1, D_agent_hist]
        agent_inter: torch.Tensor , # [B, A, T, S_i, D_agent_inter]
        road: torch.Tensor, # [B, A, 1, S_road, D_road]
        traffic_light: torch.Tensor, # [B, A, T, S_traffic_light, D_traffic_light]
    ):
        # Project inputs to model dimensions
        agent_hist = self.agent_projection(agent_hist)  # [B, A, T, 1, D_model]
        agent_inter = self.agent_inter_projection(agent_inter) # [B, A, T, S_i, D_model]
        road = self.road_projection(road)  # [B, A, 1, S_road, D_model]
        traffic_light = self.traffic_light_projection(traffic_light)  # [B, A, T, S_traffic_light, D_model]

        # Encode inputs
        B, A = agent_hist.shape[:2]
        scene_encoding = self.encoder(
            agent_hist,
            agent_inter,
            road,
            traffic_light,
            self.agent_pos_encoder.unsqueeze(0).unsqueeze(0).expand(B, A, -1, -1, -1) \
                        if self.agent_pos_encoder is not None else None,
            self.agent_inter_pos_encoder.unsqueeze(0).unsqueeze(0).expand(B, A, -1, -1, -1) \
                        if self.agent_inter_pos_encoder is not None else None,
            self.road_pos_encoder.unsqueeze(0).unsqueeze(0).expand(B, A, -1, -1, -1) \
                        if self.road_pos_encoder is not None else None,
            self.trafic_light_pos_encoder.unsqueeze(0).unsqueeze(0).expand(B, A, -1, -1, -1) \
                        if self.trafic_light_pos_encoder is not None else None,
        ) # [B*A,(), D_model]

        # Decode to get trajectory predictions
        out = self.decoder(scene_encoding) # [B*A, num_modes, D_model]
        num_modes, d_model = out.shape[1], out.shape[2]
        out = out.view(B, A, num_modes, d_model) # [B, A, num_modes, D_model]

        # Project to GMM parameters
        likelihoods = self.gmm_likelihood_projection(out) # [B, A, num_modes, 1]
        gmm_params = self.gmm_param_projection(out) # [B, A, num_modes, 4]
        return (gmm_params, likelihoods)

def build_wayformer(
    d_model,
    nhead,
    dim_feedforward,
    num_layers,
    dropout,
    fusion,
    num_latents,
    attention_type,
    num_modes,
    datasetconfig: Type[DatasetConfig]
):
    encoder = SceneEncoder(d_model, nhead, dim_feedforward, num_layers, dropout, fusion, num_latents, attention_type)

    decoder = TrajectoryDecoder(num_modes, d_model, nhead, dim_feedforward, num_layers, dropout)

    # Build projection layers
    agent_projection = torch.nn.Linear(datasetconfig.agent_hist_dim, d_model)
    agent_inter_projection = torch.nn.Linear(datasetconfig.agent_int_dim, d_model)
    road_projection = torch.nn.Linear(datasetconfig.road_dim, d_model)
    traffic_light_projection = torch.nn.Linear(datasetconfig.traffic_light_dim, d_model)

    # Build positional encoders
    agent_pos_encoder = build_positional_embedding(datasetconfig.hist_timesteps, 1, d_model)
    agent_inter_pos_encoder = build_positional_embedding(datasetconfig.hist_timesteps, datasetconfig.num_near_agents ,d_model)
    road_pos_encoder = build_positional_embedding(1, datasetconfig.num_road_segments, d_model)
    trafic_light_pos_encoder = build_positional_embedding(datasetconfig.hist_timesteps, datasetconfig.num_traffic_lights, d_model)

    # gmm projections
    gmm_likelihood_projection = torch.nn.Linear(d_model, 1)
    gmm_param_projection = torch.nn.Linear(d_model, 4)

    model = Wayformer(
        encoder,
        decoder,
        gmm_likelihood_projection,
        gmm_param_projection,
        agent_projection,
        agent_inter_projection,
        road_projection,
        traffic_light_projection,
        agent_pos_encoder,
        agent_inter_pos_encoder,
        road_pos_encoder,
        trafic_light_pos_encoder
    )
    return model
