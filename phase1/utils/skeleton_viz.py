"""
skeleton_viz.py
---------------
OpenCV overlay utilities for real-time skeleton visualization.

Draws:
  - Skeleton edges (bone connections)
  - Keypoint circles (color-coded by confidence)
  - Track IDs and bboxes
  - Pair proximity indicator
  - Inter-person line (dashed when in proximity)
  - FPS counter
"""

from __future__ import annotations
import cv2
import numpy as np
from phase1.core.pose_extractor import SKELETON_EDGES
from phase1.core.person_tracker import Track
from phase1.core.pair_manager import InteractionPair


# Colors (BGR)
COLOR_SKELETON   = (200, 200, 200)
COLOR_CONF_HIGH  = (80, 220, 80)    # green — high confidence keypoint
COLOR_CONF_MED   = (80, 180, 220)   # yellow — medium confidence
COLOR_CONF_LOW   = (80, 80, 180)    # red — low confidence
COLOR_BBOX_A     = (255, 140, 0)    # orange — person A
COLOR_BBOX_B     = (0, 200, 255)    # cyan — person B
COLOR_PAIR_LINE  = (180, 0, 255)    # purple — pair connection
COLOR_TEXT_BG    = (0, 0, 0)


def draw_skeleton(
    frame: np.ndarray,
    keypoints_px: np.ndarray,
    color_edges: tuple = COLOR_SKELETON,
    min_conf: float = 0.3,
) -> None:
    """
    Draw skeleton edges and keypoints on frame in-place.

    Args:
        frame: BGR image
        keypoints_px: (17, 3) array in PIXEL coordinates (not normalized)
        color_edges: BGR color for skeleton edges
        min_conf: skip edges/joints below this confidence
    """
    h, w = frame.shape[:2]

    # Draw edges (bones)
    for idx_a, idx_b in SKELETON_EDGES:
        conf_a = keypoints_px[idx_a, 2]
        conf_b = keypoints_px[idx_b, 2]
        if conf_a < min_conf or conf_b < min_conf:
            continue
        pt_a = (int(keypoints_px[idx_a, 0]), int(keypoints_px[idx_a, 1]))
        pt_b = (int(keypoints_px[idx_b, 0]), int(keypoints_px[idx_b, 1]))
        # Bounds check
        if not (_in_frame(pt_a, w, h) and _in_frame(pt_b, w, h)):
            continue
        cv2.line(frame, pt_a, pt_b, color_edges, 1, cv2.LINE_AA)

    # Draw keypoints
    for i in range(17):
        conf = keypoints_px[i, 2]
        if conf < min_conf:
            continue
        px = int(keypoints_px[i, 0])
        py = int(keypoints_px[i, 1])
        if not _in_frame((px, py), w, h):
            continue
        if conf >= 0.7:
            color = COLOR_CONF_HIGH
        elif conf >= 0.5:
            color = COLOR_CONF_MED
        else:
            color = COLOR_CONF_LOW
        cv2.circle(frame, (px, py), 3, color, -1, cv2.LINE_AA)


def draw_track(
    frame: np.ndarray,
    track: Track,
    color: tuple,
    label: str | None = None,
) -> None:
    """Draw bbox and track ID for a single person."""
    x1, y1, x2, y2 = [int(v) for v in track.bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    text = label or f"ID {track.track_id}"
    draw_label(frame, text, (x1, y1 - 6), color)

    draw_skeleton(frame, track.keypoints, color_edges=color)


def draw_interaction_pair(
    frame: np.ndarray,
    pair: InteractionPair,
    track_a: Track,
    track_b: Track,
) -> None:
    """
    Draw both people in the pair with their IDs and an inter-person line.
    """
    draw_track(frame, track_a, COLOR_BBOX_A, label=f"A (ID {pair.track_id_a})")
    draw_track(frame, track_b, COLOR_BBOX_B, label=f"B (ID {pair.track_id_b})")

    # Draw dashed line between hip centers
    hip_a = pair.norm_a.hip_center_px.astype(int)
    hip_b = pair.norm_b.hip_center_px.astype(int)
    _draw_dashed_line(frame, tuple(hip_a), tuple(hip_b), COLOR_PAIR_LINE, thickness=1)

    # Proximity frames indicator
    if pair.proximity_frames > 0:
        label = f"proximity: {pair.proximity_frames}f"
        mid = ((hip_a + hip_b) / 2).astype(int)
        draw_label(frame, label, tuple(mid), COLOR_PAIR_LINE)

    # Facing angle info (useful for debugging)
    fa = pair.pair_features.facing_angle_a
    fb = pair.pair_features.facing_angle_b
    dist = pair.pair_features.distance_norm
    info = f"dist={dist:.1f}T  angleA={np.degrees(fa):.0f}  angleB={np.degrees(fb):.0f}"
    draw_label(frame, info, (10, frame.shape[0] - 40), (220, 220, 220))


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Draw FPS counter in top-right corner."""
    text = f"{fps:.1f} fps"
    tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0], None
    x = frame.shape[1] - 90
    draw_label(frame, text, (x, 22), (180, 255, 180))


def draw_label(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple = (255, 255, 255),
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw text with a dark background for readability."""
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2), COLOR_TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_no_detection(frame: np.ndarray) -> None:
    """Draw a "waiting for detection" message."""
    h, w = frame.shape[:2]
    draw_label(frame, "No persons detected", (10, 30), (100, 100, 255))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _in_frame(pt: tuple[int, int], w: int, h: int) -> bool:
    return 0 <= pt[0] < w and 0 <= pt[1] < h


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple,
    thickness: int = 1,
    dash_len: int = 10,
    gap_len: int = 8,
) -> None:
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    total = np.hypot(x2 - x1, y2 - y1)
    if total < 1:
        return
    dx, dy = (x2 - x1) / total, (y2 - y1) / total
    pos = 0.0
    drawing = True
    while pos < total:
        seg_len = dash_len if drawing else gap_len
        end = min(pos + seg_len, total)
        if drawing:
            p1 = (int(x1 + dx * pos), int(y1 + dy * pos))
            p2 = (int(x1 + dx * end), int(y1 + dy * end))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        pos = end
        drawing = not drawing
