import os
import sys
import pytest

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grandparent = os.path.dirname(parent)
print(grandparent)
sys.path.append(grandparent)

import torch
from src.wayformer.encoders import Encoder, LatentEncoder, LateFusionSceneEncoder, SceneEncoder

###############################################
# TEST ENCODER                                #
###############################################

def test_encoder_output_shape():
    """
    Test that Encoder returns the correct output shape.
    """
    batch_size, seq_len, d_model = 2, 10, 16
    nhead, dim_feedforward, num_layers = 4, 32, 3
    encoder = Encoder(d_model, nhead, dim_feedforward, num_layers)
    src = torch.randn(batch_size, seq_len, d_model)
    out = encoder(src)
    assert out.shape == (batch_size, seq_len, d_model)

def test_encoder_with_positional_encoding():
    """
    Test Encoder with positional encoding.
    """
    batch_size, seq_len, d_model = 2, 12, 16
    nhead, dim_feedforward, num_layers = 4, 32, 2
    encoder = Encoder(d_model, nhead, dim_feedforward, num_layers)
    src = torch.randn(batch_size, seq_len, d_model)
    pos = torch.randn(batch_size, seq_len, d_model)
    out = encoder(src, positional_encoding=pos)
    assert out.shape == (batch_size, seq_len, d_model)

###############################################
# TEST LATENT ENCODER                         #
###############################################

def test_latentencoder_output_shape():
    """
    Test that LatentEncoder returns the correct output shape.
    """
    batch_size, seq_len, d_model = 2, 10, 16
    nhead, dim_feedforward, num_layers, num_latents = 4, 32, 3, 6
    encoder = LatentEncoder(d_model, nhead, dim_feedforward, num_layers, num_latents)
    src = torch.randn(batch_size, seq_len, d_model)
    out = encoder(src)
    assert out.shape == (batch_size, num_latents, d_model)

def test_latentencoder_with_positional_encoding():
    """
    Test LatentEncoder with positional encoding.
    """
    batch_size, seq_len, d_model = 2, 8, 16
    nhead, dim_feedforward, num_layers, num_latents = 4, 32, 2, 5
    encoder = LatentEncoder(d_model, nhead, dim_feedforward, num_layers, num_latents)
    src = torch.randn(batch_size, seq_len, d_model)
    pos = torch.randn(batch_size, seq_len, d_model)
    out = encoder(src, positional_encoding=pos)
    assert out.shape == (batch_size, num_latents, d_model)

###############################################
# TEST LATE FUSION SCENE ENCODER              #
###############################################

def test_latefusion_scene_encoder_output_shape():
    """
    Test that LateFusionSceneEncoder returns the correct output shape.
    """
    batch_size, num_agents, t_hist, s_i, s_r, s_tls, d_model = 2, 3, 4, 5, 6, 7, 8
    nhead, dim_feedforward, num_layers, num_latents = 2, 16, 2, 4
    encoder = LateFusionSceneEncoder(
        d_model, nhead, dim_feedforward, num_layers, 0.1, num_latents, "multi_axis"
    )
    agent_histories = torch.randn(batch_size, num_agents, t_hist, 1, d_model)
    agent_interactions = torch.randn(batch_size, num_agents, t_hist, s_i, d_model)
    road_graphs = torch.randn(batch_size, num_agents, 1, s_r, d_model)
    traffic_lights = torch.randn(batch_size, num_agents, t_hist, s_tls, d_model)
    out = encoder(agent_histories, agent_interactions, road_graphs, traffic_lights)
    assert out.shape[0] == batch_size * num_agents
    assert out.shape[-1] == d_model

def test_latefusion_scene_encoder_with_positional_encodings():
    """
    Test LateFusionSceneEncoder with positional encodings.
    """
    batch_size, num_agents, t_hist, s_i, s_r, s_tls, d_model = 2, 2, 3, 4, 5, 6, 8
    nhead, dim_feedforward, num_layers, num_latents = 2, 16, 2, 4
    encoder = LateFusionSceneEncoder(
        d_model, nhead, dim_feedforward, num_layers, 0.1, num_latents, "multi_axis"
    )
    agent_histories = torch.randn(batch_size, num_agents, t_hist, 1, d_model)
    agent_interactions = torch.randn(batch_size, num_agents, t_hist, s_i, d_model)
    road_graphs = torch.randn(batch_size, num_agents, 1, s_r, d_model)
    traffic_lights = torch.randn(batch_size, num_agents, t_hist, s_tls, d_model)
    agent_pos_enc = torch.randn(batch_size, num_agents, t_hist, 1, d_model)
    agent_int_pos_enc = torch.randn(batch_size, num_agents, t_hist, s_i, d_model)
    road_pos_enc = torch.randn(batch_size, num_agents, 1, s_r, d_model)
    traffic_light_pos_enc = torch.randn(batch_size, num_agents, t_hist, s_tls, d_model)
    out = encoder(
        agent_histories, agent_interactions, road_graphs, traffic_lights,
        agent_pos_enc, agent_int_pos_enc, road_pos_enc, traffic_light_pos_enc
    )
    assert out.shape[0] == batch_size * num_agents
    assert out.shape[-1] == d_model

###############################################
# TEST SCENE ENCODER                         #
###############################################

def test_scene_encoder_latefusion_output_shape():
    """
    Test SceneEncoder with late fusion returns correct output shape.
    """
    batch_size, num_agents, t_hist, s_i, s_r, s_tls, d_model = 2, 3, 4, 5, 6, 7, 8
    nhead, dim_feedforward, num_layers, num_latents = 2, 16, 2, 4
    encoder = SceneEncoder(
        d_model, nhead, dim_feedforward, num_layers, 0.1,
        fusion="late", num_latents=num_latents, attention_type="multi_axis"
    )
    agent_histories = torch.randn(batch_size, num_agents, t_hist, 1, d_model)
    agent_interactions = torch.randn(batch_size, num_agents, t_hist, s_i, d_model)
    road_graphs = torch.randn(batch_size, num_agents, 1, s_r, d_model)
    traffic_lights = torch.randn(batch_size, num_agents, t_hist, s_tls, d_model)
    out = encoder(agent_histories, agent_interactions, road_graphs, traffic_lights)
    expected_seq_len = agent_histories.shape[2] + road_graphs.shape[3] + agent_interactions.shape[3] + traffic_lights.shape[3]
    assert out.shape[0] == batch_size * num_agents
    assert out.shape[-1] == d_model

def test_scene_encoder_invalid_fusion():
    """
    Test SceneEncoder raises ValueError for invalid fusion type.
    """
    d_model, nhead, dim_feedforward, num_layers = 8, 2, 16, 2
    with pytest.raises(ValueError):
        SceneEncoder(
            d_model, nhead, dim_feedforward, num_layers, 0.1,
            fusion="invalid", num_latents=4, attention_type="multi_axis"
        )

