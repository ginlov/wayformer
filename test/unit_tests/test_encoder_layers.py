import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grandparent = os.path.dirname(parent)
print(grandparent)
sys.path.append(grandparent)

import torch
import pytest
from src.wayformer.encoders import EncoderLayer, LatentEncoderLayer

###############################################
# TEST ENCODER LAYER                          #
###############################################
def test_encoderlayer_output_shape():
    """
    Test that EncoderLayer returns the correct output shape.
    """
    batch_size, lq, lk, lv, d_model = 2, 8, 8, 8, 16
    nhead, dim_feedforward = 4, 32
    layer = EncoderLayer(d_model, nhead, dim_feedforward)
    query = torch.randn(batch_size, lq, d_model)
    key = torch.randn(batch_size, lk, d_model)
    value = torch.randn(batch_size, lv, d_model)
    out = layer(query, key, value)
    assert out.shape == (batch_size, lq, d_model)

def test_encoderlayer_with_positional_encoding():
    """
    Test EncoderLayer with positional encoding.
    """
    batch_size, lq, lk, lv, d_model = 2, 8, 12, 12, 16
    nhead, dim_feedforward = 4, 32
    layer = EncoderLayer(d_model, nhead, dim_feedforward)
    query = torch.randn(batch_size, lq, d_model)
    key = torch.randn(batch_size, lk, d_model)
    value = torch.randn(batch_size, lv, d_model)
    pos = torch.randn(batch_size, lk, d_model)
    out = layer(query, key, value, positional_encoding=pos)
    assert out.shape == (batch_size, lq, d_model)

def test_encoderlayer_gradients():
    """
    Test that gradients flow through EncoderLayer.
    """
    batch_size, lq, lk, lv, d_model = 2, 8, 8, 8, 16
    nhead, dim_feedforward = 4, 32
    layer = EncoderLayer(d_model, nhead, dim_feedforward)
    query = torch.randn(batch_size, lq, d_model, requires_grad=True)
    key = torch.randn(batch_size, lk, d_model)
    value = torch.randn(batch_size, lv, d_model)
    out = layer(query, key, value)
    loss = out.sum()
    loss.backward()
    assert query.grad is not None
    assert query.grad.shape == query.shape

def test_encoderlayer_invalid_input_shape():
    """
    Test that EncoderLayer raises an error for invalid input shape.
    """
    layer = EncoderLayer(16, 4, 32)
    query = torch.randn(2, 8, 15)  # Wrong d_model
    key = torch.randn(2, 8, 16)
    value = torch.randn(2, 8, 16)
    with pytest.raises(AssertionError):
        layer(query, key, value)

###############################################
# TEST LATENT ENCODER LAYER                   #
###############################################

def test_latent_encoder_layer_output_shape():
    """
    Test that LatentEncoderLayer returns the correct output shape.
    """
    batch_size, seq_len, d_model = 2, 10, 16
    nhead, dim_feedforward, num_latents = 4, 32, 6
    layer = LatentEncoderLayer(d_model, nhead, dim_feedforward, num_latents)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    out = layer(key, value)
    assert out.shape == (batch_size, num_latents, d_model)

def test_latent_encoder_layer_with_positional_encoding():
    """
    Test LatentEncoderLayer with positional encoding.
    """
    batch_size, seq_len, d_model = 2, 12, 16
    nhead, dim_feedforward, num_latents = 4, 32, 8
    layer = LatentEncoderLayer(d_model, nhead, dim_feedforward, num_latents)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    pos = torch.randn(batch_size, seq_len, d_model)
    out = layer(key, value, positional_encoding=pos)
    assert out.shape == (batch_size, num_latents, d_model)

def test_latent_encoder_layer_gradients():
    """
    Test that gradients flow through LatentEncoderLayer.
    """
    batch_size, seq_len, d_model = 2, 8, 16
    nhead, dim_feedforward, num_latents = 4, 32, 5
    layer = LatentEncoderLayer(d_model, nhead, dim_feedforward, num_latents)
    key = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    value = torch.randn(batch_size, seq_len, d_model)
    out = layer(key, value)
    loss = out.sum()
    loss.backward()
    assert key.grad is not None
    assert key.grad.shape == key.shape

