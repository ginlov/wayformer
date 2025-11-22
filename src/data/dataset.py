import os 
import pickle
import numpy as np
import random

from typing import List, Tuple

from dataclasses import dataclass
from src.wayformer.config import DatasetConfig
from tqdm import tqdm

@dataclass(frozen=True)
class FeatureConfig:
    agent_features = ['position', 'length', 'width', 'height', 'heading', 'velocity'] 
    road_features = ['polyline', 'type', 'speed_limit_mph']
    traffic_light_features = ['state']

tf_w2i = {
    'LANE_STATE_ARROW_CAUTION': 0,
    'LANE_STATE_ARROW_GO': 1,
    'LANE_STATE_ARROW_STOP': 2,
    'LANE_STATE_CAUTION': 3,
    'LANE_STATE_FLASHING_CAUTION': 4,
    'LANE_STATE_FLASHING_STOP': 5,
    'LANE_STATE_GO': 6,
    'LANE_STATE_STOP': 7,
    'LANE_STATE_UNKNOWN': 8,
}

class WaymoDataset():
    def __init__(
        self,
        base_folder: str,
        metadata_file: str = 'dataset_summary.pkl',
        mapping_file: str = 'dataset_mapping.pkl',
        cache_size: int = 1000,
    ):
        self.base_folder = base_folder
        self.metadata_path = os.path.join(base_folder, metadata_file)
        self.mapping_path = os.path.join(base_folder, mapping_file)
        assert os.path.exists(self.metadata_path), f"Metadata path {self.metadata_path} does not exist."
        assert os.path.exists(self.mapping_path), f"Summary path {self.mapping_path} does not exist."

        # Read metadata
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        with open(self.mapping_path, 'rb') as f:
            self.mapping = pickle.load(f)

        self.index_to_key = {i: k for i, k in enumerate(self.mapping.keys())}
        
        # Build tracks list
        self.tracks = []
        for i, key in tqdm(enumerate(self.metadata), desc="Building track index"):
            scene_meta = self.metadata[key]
            sdc_index = scene_meta['sdc_track_index']
            self.tracks.append((i, sdc_index))
            for item in scene_meta['tracks_to_predict'].values():
                self.tracks.append((i, item['track_index']))

        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(list(self.mapping.keys()))

    def __getitem__(self, index: Tuple[int, int]):
        if index in self.cache:
            return self.cache[index]

        key = self.index_to_key[index[0]]
        data_path = os.path.join(self.base_folder, self.mapping[key], key)

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        train_data = data_sample(data, index[1])

        if len(self.cache) < self.cache_size:
            self.cache[index] = train_data
        else: # replace oldest cacha entry
            oldest_index = next(iter(self.cache))
            del self.cache[oldest_index]
            self.cache[index] = train_data

        return train_data

def rotate_pos(
    new_origin: List[float], # [2]
    new_x_direction: float,
    pos: np.ndarray # [N, 2]
):
    """
    For convienience, all positions are represented as [x, y] 2D coordinates and
    z is ignored.
    Heading is in radians, is the radius from x-axis to the heading direction.
    """
    # Translate positions to new origin
    rel_pos = pos - np.array(new_origin)
    # Rotation matrix for -new_x_direction
    c, s = np.cos(-new_x_direction), np.sin(-new_x_direction)
    rot_mat = np.array([[c, -s], [s, c]])
    # Apply rotation to all positions
    rotated = rel_pos @ rot_mat.T
    return rotated

def rotate_heading(
    new_x_direction,
    heading
):
    """
    Rotate heading angles by -new_x_direction.
    Heading is in radians, is the radius from x-axis to the heading direction.
    """
    rotated = heading - new_x_direction
    # Normalize to [-pi, pi]
    rotated = (rotated + np.pi) % (2 * np.pi) - np.pi
    return rotated

def data_sample(scene, track_index: int):
    current_ts = 10

    tracks = list(scene['tracks'].values())

    # Agent features
    track_to_train = tracks[track_index]['state']
    current_track_pos = track_to_train['position'][current_ts][:2].copy() # [2]
    current_track_heading = track_to_train['heading'][current_ts].copy() # scalar
    features = np.concatenate(
        [
            rotate_pos(current_track_pos, current_track_heading,
                       track_to_train['position'][:, :2]), # [length, 2]
            track_to_train['length'][:, None], # [length, 1]
            track_to_train['width'][:, None], # [length, 1]
            track_to_train['height'][:, None], # [length, 1]
            rotate_heading(current_track_heading,
                           track_to_train['heading'])[:, None], # [length, 1]
            rotate_pos(current_track_pos, current_track_heading,
                       track_to_train['velocity']), # [length, 2]
        ],
        axis=1
    ) # [length, 8]
    agent_mask = track_to_train['valid'] # [length]
    features = features[:current_ts + 1, :] # [T, 8]
    agent_mask = agent_mask[:current_ts + 1] # [T]

    label_pos = rotate_pos(
        current_track_pos, current_track_heading,
        track_to_train['position'][current_ts + 1:, :2]
    ) # [feature_ts, 2]
    label_heading = rotate_heading(
        current_track_heading,
        track_to_train['heading'][current_ts + 1:]
    ) # [feature_ts]

    # Agent interaction features
    other_agents_pos = np.array([item['state']['position'][current_ts][:2] \
        for i, item in enumerate(tracks) if i != track_index]) # [N, 2]

    # Compute distance to selected track
    dists = np.linalg.norm(other_agents_pos - current_track_pos, axis=1) # [N]
    near_indices = np.argsort(dists)

    # Get DataConfig.num_near_agents closest agents
    num_near_agents = DatasetConfig.num_near_agents
    if len(dists) >= num_near_agents:
        selected_indices = near_indices[:num_near_agents]
        agent_interaction_mask = np.ones((num_near_agents,), dtype=np.bool)
    else:
        # Pad with zeros if not enough agents
        selected_indices = near_indices
        agent_interaction_mask = np.array([True] * len(dists) + [False] * pad_size, dtype=np.bool)
    
    agent_interaction_features = []
    interaction_mask = []
    for idx in selected_indices:
        other_track = tracks[idx]['state']
        other_features = np.concatenate(
            [
                rotate_pos(current_track_pos, current_track_heading,
                           other_track['position'][:current_ts+1, :2]), # [length, 2]
                other_track['length'][:current_ts+1, None], # [length, 1]
                other_track['width'][:current_ts+1, None], # [length, 1]
                other_track['height'][:current_ts+1, None], # [length, 1]
                rotate_heading(current_track_heading,
                               other_track['heading'])[:current_ts+1, None], # [length, 1]
                rotate_pos(current_track_pos, current_track_heading,
                           other_track['velocity'][:current_ts+1]), # [length, 2]
            ],
            axis=1
        ) # [length, 8]
        mask = other_track['valid'][:current_ts + 1] # [length]
        interaction_mask.append(mask)
        agent_interaction_features.append(other_features)

    agent_interaction_features = np.stack(agent_interaction_features, axis=0).transpose(1, 0, 2)
    # [length, num_near_agents, 8]
    interaction_mask = np.stack(interaction_mask, axis=0).transpose(1, 0)
    # [length, num_near_agents]

    # Padding if needed
    pad_size = num_near_agents - len(selected_indices)
    if pad_size > 0:
        agent_interaction_features = np.pad(
            agent_interaction_features,
            ((0, 0), (0, pad_size), (0, 0)),
            mode='constant',
            constant_values=0
        ) # [length, num_near_agents, 8]
        interaction_mask = np.pad(
            interaction_mask,
            ((0, 0), (0, pad_size)),
            mode='constant',
            constant_values=False
        ) # [length, num_near_agents]

    # Road features
    # Find nearest polyline and polygon segments
    # Get list of road segments
    poly_segments = []
    segment_types = []
    speed_limits = []
    for value in scene['map_features'].values():
        if 'speed_limit_mph' in value:
            spl = value['speed_limit_mph']
        else:
            spl = 0.0

        if 'polyline' in value:
            # One segment is the connection of two points
            points = value['polyline']
            for i in range(len(points) - 1):
                poly_segments.append(points[i:i+2, :2]) # [2, 2]
                segment_types.append(0)
                speed_limits.append(spl)
        elif 'polygon' in value:
            # One segment is the connection of two points
            # But polygon is closed, so we connect last point to first points
            points = value['polygon']
            for i in range(len(points)):
                poly_segments.append(points[i:i+2, :2] if i < len(points) - 1 else
                                     np.array([points[i], points[0]])[:, :2]) # [2, 2]
                segment_types.append(1)
                speed_limits.append(spl)
        else:
            # Static object like traffic sign
            poly_segments.append(value['position'][None, :2]) # [1, 2]
            segment_types.append(2)
            speed_limits.append(spl)

    # Get nearest segments
    # Find midpoints of segments
    segment_midpoints = np.array([
        np.mean(segment, axis=0) for segment in poly_segments
    ]) # [M, 2]
    dists_to_segments = np.linalg.norm(segment_midpoints - current_track_pos, axis=1) # [M]
    nearest_indices = np.argsort(dists_to_segments)
    if len(dists_to_segments) >= DatasetConfig.num_road_segments:
        selected_indices = nearest_indices[:DatasetConfig.num_road_segments]
    else:
        selected_indices = nearest_indices

    # Flatten segments or duplicate points for static objects
    poly_segments = np.array([poly_segments[i].flatten() if segment_types[i] != 2 \
                             else np.tile(poly_segments[i].flatten(), 2) for i in selected_indices])
    segment_types = [segment_types[i] for i in selected_indices]
    speed_limits = [speed_limits[i] for i in selected_indices]
    
    # Road features construction
    poly_segments = rotate_pos(
        current_track_pos, current_track_heading,
        poly_segments.reshape(-1, 2)
    ).reshape(-1, 4) # [num_segments, 4]
    road_features = np.concatenate(
        [
            poly_segments, # [num_segments, 4]
            np.array(segment_types)[:, None], # [num_segments, 1]
            np.array(speed_limits)[:, None], # [num_segments, 1]
        ],
        axis=1
    ) # [1, num_segments, 6]

    # Padding if needed
    pad_size = DatasetConfig.num_road_segments - len(selected_indices)
    if pad_size > 0:
        road_features = np.pad(
            road_features,
            ((0, pad_size), (0, 0)),
            mode='constant',
            constant_values=0
        ) # [num_segments, 6]
        road_mask = np.array([True] * len(selected_indices) + [False] * pad_size, dtype=np.bool)
    else:
        road_mask = np.array([True] * DatasetConfig.num_road_segments, dtype=np.bool)

    road_features = road_features[None, :] # [1, num_segments, 6]
    road_mask = road_mask[None, :] # [1, num_segments]

    # Traffic light features
    traffic_lights = list(scene['dynamic_map_states'].values())
    tl_pos = np.array([item['stop_point'][:2] for item in traffic_lights]) # [K, 2]

    if len(tl_pos) == 0:
        # No traffic lights in the scene
        # [11, num_tls, 3]
        traffic_light_features = np.zeros((11, DatasetConfig.num_traffic_lights, 3), dtype=np.float32)
        traffic_light_mask = np.array([[False] * DatasetConfig.num_traffic_lights]*11, dtype=np.bool)
    else:
        dists_to_tls = np.linalg.norm(tl_pos - current_track_pos, axis=1) # [K]
        nearest_indices = np.argsort(dists_to_tls)
        if len(dists_to_tls) >= DatasetConfig.num_traffic_lights:
            selected_indices = nearest_indices[:DatasetConfig.num_traffic_lights]
        else:
            selected_indices = nearest_indices

        tl_states = []
        traffic_light_mask = []
        for idx in selected_indices:
            item = traffic_lights[idx]
            state = [tf_w2i[_tls] if _tls is not None else -1 for _tls in item['state']['object_state'][:current_ts+1]]
            tl_states.append(state)
            traffic_light_mask.append([True if s != -1 else False for s in state])

        traffic_light_features = np.array(tl_states) # [num_tls, 11]
        tls_masks = np.array(traffic_light_mask).transpose(1, 0) # [11, num_tls]

        # Rotate positions
        tl_positions = rotate_pos(
            current_track_pos, current_track_heading,
            tl_pos[selected_indices]
        ) # [num_tls, 2]

        traffic_light_features = np.concatenate(
            [
                np.repeat(tl_positions[None, :, :], 11, axis=0), # [T, num_tls, 2]
                traffic_light_features.transpose(1, 0)[:, :, None], # [T, num_tls, 1]
            ],
            axis=2
        ) # [T, num_tls, 3]

        # Padding if needed
        pad_size = DatasetConfig.num_traffic_lights - len(selected_indices)
        if pad_size > 0:
            traffic_light_features = np.pad(
                traffic_light_features,
                ((0, 0), (0, pad_size), (0, 0)),
                mode='constant',
                constant_values=0
            ) # [11, num_tls, 3]
            traffic_light_mask = np.pad(
                tls_masks,
                ((0, 0), (0, pad_size)),
                mode='constant',
                constant_values=False
            ) # [11, num_tls]
    return {
        'agent_features': features, # [11, 8]
        'agent_mask': agent_mask, # [11]
        'agent_interaction_features': agent_interaction_features, # [11, num_near_agents, 8]
        'agent_interaction_mask': interaction_mask, # [11, num_near_agents]
        'road_features': road_features, # [1, num_road_segments, 6]
        'road_mask': road_mask, # [1, num_road_segments]
        'traffic_light_features': traffic_light_features, # [11, num_traffic_lights, 3]
        'traffic_light_mask': traffic_light_mask, # [11, num_traffic_lights]
        'label_pos': label_pos, # [80, 2]
        'label_heading': label_heading, # [80]
    }
