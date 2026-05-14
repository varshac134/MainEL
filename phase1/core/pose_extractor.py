"""
pose_extractor.py
-----------------
YOLOv8-Pose wrapper.

Responsibilities:
  - Load the YOLOv8-Pose model (nano/small/medium selectable)
  - Run inference on a single BGR frame
  - Return raw detections: bboxes, keypoints (17, 3), confidence scores

Does NOT track, normalize, or buffer — those are separate concerns.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO


# COCO 17 keypoint names — index = joint id
KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Skeleton connectivity for visualization
SKELETON_EDGES = [
    (0, 1), (0, 2),           # nose → eyes
    (1, 3), (2, 4),           # eyes → ears
    (5, 6),                   # shoulders
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 11), (6, 12),         # shoulders → hips
    (11, 12),                 # hips
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]

# Index shortcuts used for normalization
IDX_LEFT_HIP = 11
IDX_RIGHT_HIP = 12
IDX_LEFT_SHOULDER = 5
IDX_RIGHT_SHOULDER = 6


@dataclass
class RawDetection:
    """Single person detection from YOLOv8-Pose — pixel coordinates, unnormalized."""
    bbox: np.ndarray          # shape (4,) — [x1, y1, x2, y2] in pixels
    keypoints: np.ndarray     # shape (17, 3) — [x, y, confidence] in pixels
    det_confidence: float     # person detection confidence


class PoseExtractor:
    """
    Thin wrapper around YOLOv8-Pose.

    Usage:
        extractor = PoseExtractor(model_size="n")   # n=nano, s=small, m=medium
        detections = extractor.run(frame)           # frame: BGR np.ndarray
    """

    MODEL_SIZES = {
        "n": "yolov8n-pose.pt",
        "s": "yolov8s-pose.pt",
        "m": "yolov8m-pose.pt",
        "l": "yolov8l-pose.pt",
    }

    def __init__(
        self,
        model_size: str = "n",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ):
        """
        Args:
            model_size: "n" (nano, fastest) | "s" | "m" | "l" (most accurate)
            conf_threshold: minimum person detection confidence to keep
            iou_threshold: NMS IoU threshold
            device: "auto" | "cpu" | "cuda" | "mps"
        """
        model_name = self.MODEL_SIZES.get(model_size)
        if model_name is None:
            raise ValueError(f"model_size must be one of {list(self.MODEL_SIZES.keys())}")

        # Ultralytics auto-downloads on first use
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Warm up the model (avoids first-frame latency spike)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, device=self.device)

    def run(self, frame: np.ndarray) -> list[RawDetection]:
        """
        Run pose estimation on a single BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            List of RawDetection — one per detected person, sorted by bbox area (largest first).
        """
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
        )

        detections: list[RawDetection] = []

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()       # (N, 4)
            confs = result.boxes.conf.cpu().numpy()       # (N,)
            kpts  = result.keypoints.data.cpu().numpy()   # (N, 17, 3)

            for i in range(len(boxes)):
                det = RawDetection(
                    bbox=boxes[i],
                    keypoints=kpts[i],
                    det_confidence=float(confs[i]),
                )
                detections.append(det)

        # Sort by bbox area descending (largest/closest person first)
        detections.sort(
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            reverse=True,
        )

        return detections
