from src.wayformer.wayformer import Wayformer, build_wayformer
from typing import List

import torch

class PPOWayformer(torch.nn.Module):
    def __init__(
        self,
        wayformer: Wayformer,
        value_network: torch.nn.Module,
    ):
        super().__init__()
        self.wayformer = wayformer
        self.value_network = value_network

    def forward(
        self,
        agent_hist: torch.Tensor, # [A, T, 1, D_agent_hist]
        agent_inter: torch.Tensor , # [A, T, S_i, D_agent_inter]
        road: torch.Tensor, # [A, 1, S_road, D_road]
        traffic_light: torch.Tensor, # [A, T, S_traffic_light, D_traffic_light]
        agent_masks: torch.Tensor | None = None, # [A, T, 1]
        agent_inter_masks: torch.Tensor | None = None, # [A, T, S_i]
        road_masks: torch.Tensor | None = None, # [A, 1, S_road]
        traffic_light_masks: torch.Tensor | None = None, # [A, T, S_traffic_light]
    ):
        wayformer_output = self.wayformer(
            agent_hist,
            agent_inter,
            road,
            traffic_light,
            agent_masks,
            agent_inter_masks,
            road_masks,
            traffic_light_masks,
            axillary_outputs=True,
        )
        value = self.value_network(wayformer_output[2])
        return (wayformer_output[0], wayformer_output[1], value)

def build_ppo_wayformer(
    d_model,
    nhead,
    dim_feedforward,
    num_layers,
    dropout,
    fusion,
    num_latents,
    attention_type,
    num_decoder_layers,
    num_modes,
    datasetconfig,
    hidden_dim_value_network: List[int],
):
    wayformer = build_wayformer(
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
        fusion,
        num_latents,
        attention_type,
        num_decoder_layers,
        num_modes,
        datasetconfig,
    )
    value_network_layers = []
    input_dim = d_model
    for hdim in hidden_dim_value_network:
        value_network_layers.append(torch.nn.Linear(input_dim, hdim))
        value_network_layers.append(torch.nn.ReLU())
        input_dim = hdim
    value_network_layers.append(torch.nn.Linear(input_dim, 1))
    value_network = torch.nn.Sequential(*value_network_layers)

    ppo_wayformer = PPOWayformer(
        wayformer=wayformer,
        value_network=value_network,
    )
    return ppo_wayformer
