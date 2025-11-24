import numpy as np
import torch
import time
import cv2

from typing import List
from src.wayformer.config import DatasetConfig
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection

from cvrunner.utils.logger import get_cv_logger

logger = get_cv_logger()

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

def rotate_velo(
    new_x_direction: float,
    velo: np.ndarray  # [N, 2]
):
    """
    Rotate velocity vectors by -new_x_direction.
    """
    c, s = np.cos(-new_x_direction), np.sin(-new_x_direction)
    rot_mat = np.array([[c, -s], [s, c]])
    rotated = velo @ rot_mat.T
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
            rotate_velo(current_track_heading,
                       track_to_train['velocity']), # [length, 2]
        ],
        axis=1
    ) # [length, 8]
    agent_mask = track_to_train['valid'] # [length]
    features = features[:current_ts + 1, :] # [T, 8]
    agent_mask = agent_mask[:current_ts + 1] # [T]

    label_pos = rotate_pos(
        current_track_pos, current_track_heading,
        track_to_train['position'][current_ts + 1:current_ts + 1 + DatasetConfig.future_timesteps, :2]
    ) # [feature_ts, 2]
    label_heading = rotate_heading(
        current_track_heading,
        track_to_train['heading'][current_ts + 1:current_ts + 1 + DatasetConfig.future_timesteps]
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
    else:
        # Pad with zeros if not enough agents
        selected_indices = near_indices
    
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
                rotate_velo(current_track_heading,
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
    dists_to_segments = np.linalg.norm(segment_midpoints - current_track_pos - np.array([15.0, 0.0]), axis=1) # [M]
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

    road_features = road_features[None, :].astype(np.float32) # [1, num_segments, 6]
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
        traffic_light_mask= np.array(traffic_light_mask).transpose(1, 0) # [11, num_tls]

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
                traffic_light_mask,
                ((0, 0), (0, pad_size)),
                mode='constant',
                constant_values=False
            ) # [11, num_tls]

    assert features.shape == (11, 8)
    assert agent_mask.shape == (11,)
    assert agent_interaction_features.shape == (11, DatasetConfig.num_near_agents, 8)
    assert interaction_mask.shape == (11, DatasetConfig.num_near_agents)
    assert road_features.shape == (1, DatasetConfig.num_road_segments, 6)
    assert road_features.dtype == np.float32
    assert road_mask.shape == (1, DatasetConfig.num_road_segments)
    assert traffic_light_features.shape == (11, DatasetConfig.num_traffic_lights, 3)
    assert traffic_light_mask.shape == (11, DatasetConfig.num_traffic_lights)
    assert label_pos.shape == (DatasetConfig.future_timesteps, 2)
    assert label_heading.shape == (DatasetConfig.future_timesteps,)
    return {
        'agent_features': features[:, None, :], # [11, 1, 8]
        'agent_mask': agent_mask[:, None], # [11, 1]
        'agent_interaction_features': agent_interaction_features, # [11, num_near_agents, 8]
        'agent_interaction_mask': interaction_mask, # [11, num_near_agents]
        'road_features': road_features, # [1, num_road_segments, 6]
        'road_mask': road_mask, # [1, num_road_segments]
        'traffic_light_features': traffic_light_features.astype(np.float32), # [11, num_traffic_lights, 3]
        'traffic_light_mask': traffic_light_mask, # [11, num_traffic_lights]
        'label_pos': label_pos, # [fut_ts, 2]
        'label_heading': label_heading, # [fut_ts]
        'idx': f"{scene['id']}_{track_index}"
    }

def collate_fn(batch):
    return {
        key: torch.tensor(np.array([item[key] for item in batch])) \
        if key != 'idx' else [item[key] for item in batch]for key in batch[0].keys()
    }

def visualize_one(
    road_lines,
    hist_traj,
    future_traj,
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)  # smaller/faster
    canvas = FigureCanvasAgg(fig)

    # Draw road lines with LineCollection
    lines = np.array(road_lines).reshape(-1, 2, 2)
    lc = LineCollection(lines, colors='gray', linewidths=1)
    ax.add_collection(lc)

    # Draw historical trajectory
    ax.plot(hist_traj[:, 0], hist_traj[:, 1], color='red', label='History')

    # Draw future trajectories (combine all modes for speed, one label)
    for mode in range(future_traj.shape[0]):
        ax.plot(future_traj[mode, :, 0], future_traj[mode, :, 1], color='blue')
    ax.plot([], [], color='blue', label='Prediction')  # single legend entry

    ax.legend()
    ax.set_aspect('equal')
    ax.set_title('Trajectory Visualization')

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img = buf[..., :3].reshape(buf.shape[0], buf.shape[1], 3).copy()
    w, h = fig.get_size_inches() * fig.dpi
    img = img.reshape(int(h), int(w), 3)
    plt.close(fig)
    return img

def visualize_one_cv2(road_lines, hist_traj, future_traj, gt_traj, img_size=800, margin=20):
    # Gather all points to determine bounds
    points = []
    for line in road_lines:
        points.append(line[:2])
        points.append(line[2:4])
    points.extend(hist_traj)
    for mode in range(future_traj.shape[0]):
        points.extend(future_traj[mode])
    points = np.array(points)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)

    # Compute scale and offset to fit all points with margin
    scale = (img_size - 2 * margin) / np.max(max_xy - min_xy)
    offset = min_xy - margin / scale

    def to_img_coords(xy):
        # Scale and flip y-axis for image coordinates
        xy = (xy - offset) * scale
        xy[..., 1] = img_size - xy[..., 1]  # y-axis: bottom to top
        return xy

    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    colors = [
        (0, 0, 255),      # Red
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (128, 0, 255),    # Purple
        (255, 128, 0),    # Orange
    
        (255, 0, 128),    # Pink-Red
        (0, 128, 255),    # Gold-ish
        (128, 128, 255),  # Lavender
        (255, 128, 128),  # Salmon
        (128, 255, 0),    # Lime Yellow (not green)
    ]

    # Draw road lines
    for line in road_lines:
        pt1 = tuple(np.round(to_img_coords(np.array(line[:2]))).astype(int))
        pt2 = tuple(np.round(to_img_coords(np.array(line[2:4]))).astype(int))
        cv2.line(img, pt1, pt2, color=(128, 128, 128), thickness=1)

    # Draw historical trajectory (red)
    hist_pts = np.round(to_img_coords(hist_traj)).astype(int)
    cv2.polylines(img, [hist_pts], isClosed=False, color=(0, 0, 255), thickness=1)

    # Draw ground-truth trajectory (green)
    gt_pts = np.round(to_img_coords(gt_traj)).astype(int)
    cv2.polylines(img, [np.concatenate([hist_pts[-1].reshape(1, 2), gt_pts], axis=0)], isClosed=False, color=(0,255,0), thickness=1)

    # Draw future trajectories (blue)
    for mode in range(future_traj.shape[0]):
        fut_pts = np.round(to_img_coords(future_traj[mode])).astype(int)
        cv2.polylines(img, [np.concatenate([hist_pts[-1].reshape(1, 2), fut_pts], axis=0)], isClosed=False, color=colors[mode%10], thickness=1)
    return img

@torch.no_grad()
def visualize_scene(
    data_batch,
    predictions,
    gt_traj # [A, ts, 2]
):
    road_lines = data_batch['road_features'][:, 0, :, :4].cpu().numpy()
    hist_traj = data_batch['agent_features'][:, :, 0, :2].cpu().numpy()
    gt_traj = gt_traj.cpu().numpy()

    traj_preds, mode_probs = predictions

    future_traj = traj_preds[:, :, :, :2].cpu().numpy() # [A, num_modes, ts, 2]

    out_imgs = []
    for i in range(road_lines.shape[0]):
        img = visualize_one_cv2(
            road_lines[i],
            hist_traj[i],
            future_traj[i],
            gt_traj[i]
        )
        out_imgs.append(img)
    return out_imgs
