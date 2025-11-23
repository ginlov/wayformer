import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grandparent = os.path.dirname(parent)
print(grandparent)
sys.path.append(grandparent)

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
    future_timesteps = 5
    num_near_agents = 2
    num_road_segments = 2
    num_traffic_lights = 2

@pytest.fixture
def wayformer_model():
    return build_wayformer(
        d_model=16, 
        nhead=2,
        dim_feedforward=32,
        num_layers=2,
        dropout=0.1,
        fusion='late',
        num_latents=4,
        attention_type='latent',
        num_modes=3,
        datasetconfig=DummyConfig

    )

def test_wayformer_forward_shapes(wayformer_model):
    A, T, S_i, S_road, S_traffic_light = 3, 3, 2, 2, 2
    D_agent_hist = DummyConfig.agent_hist_dim
    D_agent_inter = DummyConfig.agent_int_dim
    D_road = DummyConfig.road_dim
    D_traffic_light = DummyConfig.traffic_light_dim

    agent_hist = torch.randn(A, T, 1, D_agent_hist)
    agent_inter = torch.randn(A, T, S_i, D_agent_inter)
    road = torch.randn(A, 1, S_road, D_road)
    traffic_light = torch.randn(A, T, S_traffic_light, D_traffic_light)

    gmm_params, likelihoods = wayformer_model(agent_hist, agent_inter, road, traffic_light)
    assert gmm_params.shape[:2] == (A, 3)
    assert likelihoods.shape[:2] == (A, 3)
    assert gmm_params.shape[-1] == 4
    assert likelihoods.shape[-1] == 1

