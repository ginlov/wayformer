import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grandparent = os.path.dirname(parent)
sys.path.append(grandparent)

import torch
from src.wayformer.decoders import DecoderLayer, TrajectoryDecoder

###############################################
# TEST DECODER LAYER                          #
###############################################

def test_decoder_layer_forward():
    batch_size = 2
    seq_len_q = 4
    seq_len_kv = 6
    d_model = 8
    nhead = 2
    dim_feedforward = 16

    layer = DecoderLayer(d_model, nhead, dim_feedforward)
    query = torch.randn(batch_size, seq_len_q, d_model)
    key = torch.randn(batch_size, seq_len_kv, d_model)
    value = torch.randn(batch_size, seq_len_kv, d_model)
    positional_encoding = torch.randn(batch_size, seq_len_kv, d_model)
    key_mask = torch.ones(batch_size, seq_len_kv, dtype=torch.bool)

    # Test with positional encoding
    out = layer(query, key, value, positional_encoding, key_mask)
    assert out.shape == (batch_size, seq_len_q, d_model)

    # Test without positional encoding
    out2 = layer(query, key, value)
    assert out2.shape == (batch_size, seq_len_q, d_model)

###############################################
# TEST TRAJECTORY DECODER                     #
###############################################

def test_trajectory_decoder_forward():
    batch_size = 3
    num_modes = 5
    seq_len_mem = 7
    d_model = 8
    nhead = 2
    dim_feedforward = 16
    num_layers = 2

    decoder = TrajectoryDecoder(num_modes, d_model, nhead, dim_feedforward, num_layers)
    memory = torch.randn(batch_size, seq_len_mem, d_model)
    positional_encoding = torch.randn(batch_size, seq_len_mem, d_model)

    # Test with positional encoding
    out = decoder(memory, positional_encoding)
    assert out.shape == (batch_size, num_modes, d_model)

    # Test without positional encoding
    out2 = decoder(memory)
    assert out2.shape == (batch_size, num_modes, d_model)

