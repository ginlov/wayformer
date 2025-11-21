import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
print(parent)
sys.path.append(parent)

import torch
import pytest

from src.wayformer.wayformer import Wayformer, build_wayformer

class DummyEncoder(torch.nn.Module):
    def forward(self, *args, **kwargs):
        # Return a dummy encoding of shape [B*A, D_model]
        agent_hist = args[0]
        B, A = agent_hist.shape[:2]
        D_model = agent_hist.shape[-1]
        return torch.randn(B * A, D_model)

class DummyDecoder(torch.nn.Module):
    def __init__(self, num_modes, d_model):
        super().__init__()
        self.num_modes = num_modes
        self.d_model = d_model

    def forward(self, x):
        # x: [B*A, D_model]
        B_A = x.shape[0]
        return torch.randn(B_A, self.num_modes, self.d_model)

class DummyConfig:
    agent_hist_dim = 8
    agent_int_dim = 6
    road_dim = 5
    traffic_light_dim = 4
    hist_timesteps = 3
    num_near_agents = 2
    num_road_segments = 2
    num_traffic_lights = 2

@pytest.fixture
def wayformer_model():
    d_model = 16
    num_modes = 3
    encoder = DummyEncoder()
    decoder = DummyDecoder(num_modes, d_model)
    agent_projection = torch.nn.Linear(DummyConfig.agent_hist_dim, d_model)
    agent_inter_projection = torch.nn.Linear(DummyConfig.agent_int_dim, d_model)
    road_projection = torch.nn.Linear(DummyConfig.road_dim, d_model)
    traffic_light_projection = torch.nn.Linear(DummyConfig.traffic_light_dim, d_model)
    agent_pos_encoder = torch.nn.Parameter(torch.randn(DummyConfig.hist_timesteps, 1, d_model))
    agent_inter_pos_encoder = torch.nn.Parameter(torch.randn(DummyConfig.hist_timesteps, DummyConfig.num_near_agents, d_model))
    road_pos_encoder = torch.nn.Parameter(torch.randn(1, DummyConfig.num_road_segments, d_model))
    trafic_light_pos_encoder = torch.nn.Parameter(torch.randn(DummyConfig.hist_timesteps, DummyConfig.num_traffic_lights, d_model))
    gmm_likelihood_projection = torch.nn.Linear(d_model, 1)
    gmm_param_projection = torch.nn.Linear(d_model, 4)
    return Wayformer(
        encoder, decoder, gmm_likelihood_projection, gmm_param_projection,
        agent_projection, agent_inter_projection, road_projection, traffic_light_projection,
        agent_pos_encoder, agent_inter_pos_encoder, road_pos_encoder, trafic_light_pos_encoder
    )

def test_wayformer_forward_shapes(wayformer_model):
    B, A, T, S_i, S_road, S_traffic_light = 2, 3, 3, 2, 2, 2
    D_agent_hist = DummyConfig.agent_hist_dim
    D_agent_inter = DummyConfig.agent_int_dim
    D_road = DummyConfig.road_dim
    D_traffic_light = DummyConfig.traffic_light_dim

    agent_hist = torch.randn(B, A, T, 1, D_agent_hist)
    agent_inter = torch.randn(B, A, T, S_i, D_agent_inter)
    road = torch.randn(B, A, 1, S_road, D_road)
    traffic_light = torch.randn(B, A, T, S_traffic_light, D_traffic_light)

    gmm_params, likelihoods = wayformer_model(agent_hist, agent_inter, road, traffic_light)
    assert gmm_params.shape[:3] == (B, A, wayformer_model.decoder.num_modes)
    assert likelihoods.shape[:3] == (B, A, wayformer_model.decoder.num_modes)
    assert gmm_params.shape[-1] == 4
    assert likelihoods.shape[-1] == 1

