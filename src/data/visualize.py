import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
import cv2

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
    gt_traj,# [A, ts, 2]
    gt_mask # [A, ts]
):
    road_lines = data_batch['road_features'][:, 0, :, :4].cpu().numpy()
    hist_traj = list(data_batch['agent_features'][:, :, 0, :2].cpu().numpy())
    hist_mask = data_batch['agent_mask'][:, :, 0].cpu().numpy()
    for i in range(len(hist_traj)):
        hist_traj[i] = hist_traj[i][hist_mask[i]]
    gt_traj = list(gt_traj.cpu().numpy())
    gt_mask = gt_mask.cpu().numpy()
    for i in range(len(gt_traj)):
        gt_traj[i] = gt_traj[i][gt_mask[i]]

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

