import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grandparent = os.path.dirname(parent)
sys.path.append(grandparent)

import pytest
import torch
import copy
from src.wayformer.loss import WayformerLoss

@pytest.fixture
def loss_function():
    return WayformerLoss()

def test_wayformer_loss_wo_mask(loss_function):
    A, ts, num_modes = 3, 60, 5

    targets = torch.randn(A, ts, 2)
    target_mask = torch.ones(A, ts)

    predictions = (
        torch.randn(A, num_modes, ts, 4),
        torch.softmax(torch.randn(A, num_modes), dim=-1)
    )

    loss = loss_function(targets, target_mask, predictions)

    assert 'loss/loss' in loss
    assert 'loss/classification_loss' in loss
    assert 'loss/regression_loss' in loss
    assert loss['loss/loss'].item() > 0
    assert loss['loss/classification_loss'].item() > 0
    assert loss['loss/regression_loss'].item() > 0

def test_wayformer_loss_with_mask(loss_function):
    A, ts, num_modes = 2, 50, 4

    targets = torch.randn(A, ts, 2)
    target_mask = torch.zeros(A, ts)
    target_mask[0, :25] = 1.0
    target_mask[1, 10:40] = 1.0

    predictions = (
        torch.randn(A, num_modes, ts, 4),
        torch.softmax(torch.randn(A, num_modes), dim=-1)
    )

    loss = loss_function(targets, target_mask, predictions)

    assert 'loss/loss' in loss
    assert 'loss/classification_loss' in loss
    assert 'loss/regression_loss' in loss
    assert loss['loss/loss'].item() > 0
    assert loss['loss/classification_loss'].item() > 0
    assert loss['loss/regression_loss'].item() > 0

def test_wayformer_perfect_match(loss_function):
    # Perfect prediction case
    A, ts, num_modes = 1, 100, 6

    targets = torch.randn(A, ts, 2)
    target_mask = torch.ones(A, ts)
    predictions = (
        torch.zeros(A, num_modes, ts, 4).fill_(-100),  # Perfect predictions at zero
        torch.zeros(A, num_modes).fill_(1.0 / num_modes)  #
    )

    predictions[0][:, :, :, :2] = targets.unsqueeze(1).expand(-1, num_modes, -1, -1)

    loss = loss_function(targets, target_mask, predictions)

    expected_classification_loss = -torch.log(torch.tensor(1.0 / num_modes) + 1e-9)

    assert torch.isclose(loss['loss/classification_loss'], expected_classification_loss, atol=1e-4), \
        f"Expected classification loss {expected_classification_loss}, got {loss['loss/classification_loss']}"
    assert torch.isclose(torch.exp(loss['loss/regression_loss']), torch.tensor(0.0), atol=1e-4), \
        f"Expected regression loss 0.0, got {torch.exp(loss['loss/regression_loss'])}"


def test_relative_classification_value_loss(loss_function):
    A, ts, num_modes = 2, 80, 3

    targets = torch.randn(A, ts, 2) * 10.0  # Larger scale targets
    target_mask = torch.ones(A, ts)

    predictions = (
        torch.randn(A, num_modes, ts, 4) * 10.0,
        torch.softmax(torch.randn(A, num_modes), dim=-1)
    )

    better_predictions = copy.deepcopy(predictions)
    # Better classification predictions
    # Find best mode and increase its probability base on closeness to target
    for a in range(A):
        dists = torch.norm(better_predictions[0][a, :, :, :2] - targets[a].unsqueeze(0), dim=-1).mean(dim=-1)
        best_mode = torch.argmin(dists)
        # increase the proabability of the best mode and decrease others accordingly
        for i in range(num_modes):
            if i == best_mode:
                better_predictions[1][a, i] += 0.25
            else:
                better_predictions[1][a, i] -= 0.25 / (num_modes - 1)

    loss = loss_function(targets, target_mask, predictions)
    better_loss = loss_function(targets, target_mask, better_predictions)

    assert better_loss['loss/classification_loss'].item() < loss['loss/classification_loss'].item(), \
        f"Better classification predictions should yield lower classification loss, got {better_loss['loss/classification_loss'].item()} >= {loss['loss/classification_loss'].item()}"
    assert better_loss['loss/regression_loss'].item() == loss['loss/regression_loss'].item(), \
        f"Regression loss should not increase with better classification predictions, got {better_loss['loss/regression_loss'].item()} != {loss['loss/regression_loss'].item()}"

def test_relative_regression_value_loss(loss_function):
    A, ts, num_modes = 2, 80, 3

    targets = torch.randn(A, ts, 2) * 10.0  # Larger scale targets
    target_mask = torch.ones(A, ts)

    predictions = (
        torch.randn(A, num_modes, ts, 4) * 10.0,
        torch.softmax(torch.ones(A, num_modes), dim=-1)
    )

    better_predictions = copy.deepcopy(predictions)
    # Improve regression predictions for the best mode
    for a in range(A):
        dists = torch.norm(better_predictions[0][a, :, :, :2] - targets[a].unsqueeze(0), dim=-1).mean(dim=-1)
        best_mode = torch.argmin(dists)
        # Move best mode predictions closer to targets
        better_predictions[0][a, best_mode, :, :2] = targets[a].unsqueeze(0)

    loss = loss_function(targets, target_mask, predictions)
    better_loss = loss_function(targets, target_mask, better_predictions)

    assert better_loss['loss/regression_loss'].item() < loss['loss/regression_loss'].item(), \
        f"Better regression predictions should yield lower regression loss, got {better_loss['loss/regression_loss'].item()} >= {loss['loss/regression_loss'].item()}"
    assert better_loss['loss/classification_loss'].item() == loss['loss/classification_loss'].item(), \
        f"Classification loss should not increase with better regression predictions, got {better_loss['loss/classification_loss'].item()} != {loss['loss/classification_loss'].item()}"
