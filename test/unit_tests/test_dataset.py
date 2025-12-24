import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
grandparent = os.path.dirname(parent)
sys.path.append(grandparent)

from src.data.dataset import WaymoDataset
from time import time
import random

###############################################
# TEST WAYMO DATASET                          #
###############################################
def test_waymo_dataset():
    # Use training set to test
    base_folder = os.path.abspath("/home/leo/data/scenario_format_waymo/training/")

    start_time = time()
    dataset = WaymoDataset(base_folder)
    end_time = time()
    print(f"Loaded WaymoDataset with {len(dataset)} samples in {end_time - start_time:.2f} seconds.")
    assert len(dataset) > 0, "Dataset should not be empty"

    index_to_sample = dataset.tracks
    track = random.choice(index_to_sample)

    time_start = time()
    sample = dataset[track]
    time_end = time()
    print(f"Retrieved sample for track {track} in {time_end - time_start:.2f} seconds.")
    assert list(sample.keys()) == [
        "agent_features",
        "agent_mask",
        'agent_interaction_features',
        'agent_interaction_mask',
        'road_features',
        'road_mask',
        'traffic_light_features',
        'traffic_light_mask',
        'label_pos',
        'label_heading',
        'label_mask',
        'idx',
        'agent_future_width',
        'other_agents_future_pos',
        'other_agents_future_width',
        'other_agents_future_mask'
    ]
