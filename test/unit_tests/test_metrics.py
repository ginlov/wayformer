import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grandparent = os.path.dirname(parent)
sys.path.append(grandparent)

import pytest
import torch
from src.wayformer.metrics import WaymoMetrics

@pytest.fixture
def wayformer_metrics():
    return WaymoMetrics()

@pytest.fixture
def dummy_databatch():
    n_other_agents = 3
    label_pos = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])  # Shape: (1, 3, 2), linear motion
    label_mask = torch.ones((1, 4), dtype=torch.bool)  # Shape: (1, 4)
    agent_future_width = torch.tensor([2.0])  # Shape: (1,)
    other_agents_future_pos = torch.tensor([[
        [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],  # Agent 1
        [[-5.0, -5.0], [-6.0, -6.0], [-7.0, -7.0], [-8.0, -8.0]],  # Agent 2
        [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0]]   # Agent 3
    ]])  # Shape: (1, 3, 4, 2)
    other_agents_future_mask = torch.ones((1, n_other_agents, 4), dtype=torch.bool)  # Shape: (1, 3, 4)
    other_agents_future_width = torch.tensor([[2.0, 2.0, 2.0]])  # Shape: (1, 3)
    predictions = (
        torch.tensor([[
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],  # Mode 1
            [[0.0, 0.0], [1.5, 1.5], [3.0, 3.0], [4.5, 4.5]]   # Mode 2
        ]]),  # Shape: (1, 2, 4, 2)
        torch.tensor([[0.7, 0.3]])  # Shape: (1, 2)
    )

    return ({
        'label_pos': label_pos,
        'label_mask': label_mask,
        'agent_future_width': agent_future_width,
        'other_agents_future_pos': other_agents_future_pos,
        'other_agents_future_mask': other_agents_future_mask,
        'other_agents_future_width': other_agents_future_width
    }, predictions)

def test_compute_metrics(wayformer_metrics, dummy_databatch):
    databatch, predictions = dummy_databatch
    A, ts, num_modes = predictions[0].shape[0], predictions[0].shape[2], predictions[0].shape[1]

    metrics = wayformer_metrics(databatch, predictions)

    assert 'min_average_distance_error' in metrics
    assert 'min_fde' in metrics
    assert 'brier_min_fde' in metrics
    assert 'overlap' in metrics
    for i in range(ts):
        assert f'miss_rate_t{str(i+1).zfill(2)}' in metrics
        assert f'min_distance_error_t{str(i+1).zfill(2)}' in metrics

    # Since one of the predicted modes perfectly matches the ground truth,
    # min_ade and min_fde should be zero, and miss_rate should be zero.
    assert metrics['min_average_distance_error'] == 0.0
    assert metrics['min_fde'] == 0.0
    assert metrics['overlap'] == 0.0
    for i in range(ts):
        assert metrics[f'miss_rate_t{str(i+1).zfill(2)}'] == 0
        assert metrics[f'min_distance_error_t{str(i+1).zfill(2)}'] == 0.0


def test_overlap_metrics(wayformer_metrics, dummy_databatch):
    databatch, predictions = dummy_databatch
    A, ts, num_modes = predictions[0].shape[0], predictions[0].shape[2], predictions[0].shape[1]

    # Modify predictions to create overlap scenario
    overlapping_prediction = torch.tensor([[
        [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],  # Mode 1 (overlap with agent 1)
        [[0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]   # Mode 2
    ]])  # Shape: (1, 2, 4, 2)
    overlapping_probabilities = torch.tensor([[0.7, 0.3]])  # Shape: (1, 2)
    overlapping_predictions = (overlapping_prediction, overlapping_probabilities)

    metrics = wayformer_metrics(databatch, overlapping_predictions)

    assert 'overlap' in metrics
    assert metrics['overlap'] > 0.0  # Expect some overlap due to the first mode
