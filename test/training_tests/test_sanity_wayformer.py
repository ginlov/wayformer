import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grandparent = os.path.dirname(parent)
sys.path.append(grandparent)

import torch
import pytest
from src.wayformer.wayformer import build_wayformer
from src.wayformer.config import DatasetConfig

import torch
import pytest
from src.wayformer.wayformer import build_wayformer
from src.wayformer.config import DatasetConfig

class DummyConfig:
    agent_hist_dim = 8
    agent_int_dim = 4
    road_dim = 6
    traffic_light_dim = 3
    hist_timesteps = 2
    future_timesteps = 5
    num_near_agents = 2
    num_road_segments = 2
    num_traffic_lights = 2

@pytest.mark.parametrize("device", ["cpu"])
def test_wayformer_training_step(device):
    torch.manual_seed(0)
    A, T, S_i, S_road, S_tl = 2, DummyConfig.hist_timesteps,\
                DummyConfig.num_near_agents, DummyConfig.num_road_segments, DummyConfig.num_traffic_lights
    D_agent_hist = DummyConfig.agent_hist_dim
    D_agent_inter = DummyConfig.agent_int_dim
    D_road = DummyConfig.road_dim
    D_tl = DummyConfig.traffic_light_dim
    num_modes = 2

    model = build_wayformer(
        d_model=16,
        nhead=2,
        dim_feedforward=32,
        num_layers=1,
        dropout=0.1,
        fusion="late",
        num_latents=2,
        attention_type="latent",
        num_decoder_layers=1,
        num_modes=num_modes,
        num_likelihoods_proj_layers=1,
        datasetconfig=DummyConfig
    ).to(device)

    # Dummy data
    agent_hist = torch.randn(A, T+1, 1, D_agent_hist, device=device)
    agent_inter = torch.randn(A, T+1, S_i, D_agent_inter, device=device)
    road = torch.randn(A, 1, S_road, D_road, device=device)
    traffic_light = torch.randn(A, T+1, S_tl, D_tl, device=device)
    target = torch.randn(A, num_modes, DummyConfig.future_timesteps, 4, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    losses = []
    for _ in range(200):
        optimizer.zero_grad()
        gmm_params, _ = model(agent_hist, agent_inter, road, traffic_light)
        loss = torch.nn.functional.mse_loss(gmm_params, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Loss: {loss.item()}")
    losses = torch.tensor(losses)
    # Check that loss decreases
    assert torch.mean(losses[-10:]) < torch.mean(losses[:10])
