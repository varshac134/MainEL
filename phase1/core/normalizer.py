"""
normalizer.py
-------------
Converts raw pixel-coordinate keypoints into normalized pose representations.

Two-step normalization (no hardcoded body measurements):
  1. TRANSLATE: center on hip midpoint → removes absolute position
  2. SCALE: divide by torso length → removes body-size variation

After this, (0,0) is always the hip center, and "1 unit" is always one torso length.
This means a tall person and a short person doing the same action look identical to
the model.

Also computes inter-person features (relative geometry between person A and B).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from phase1.core.pose_extractor import IDX_LEFT_HIP, IDX_RIGHT_HIP, IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER


@dataclass
class NormalizedPose:
    """Normalized pose for a single person."""
    track_id: int
    keypoints_norm: np.ndarray    # shape (17, 3) — (x, y, conf), normalized
    torso_length: float           # pixel torso length before normalization (useful for scale)
    hip_center_px: np.ndarray     # shape (2,) — pixel coords of hip midpoint
    shoulder_center_px: np.ndarray # shape (2,) — pixel coords of shoulder midpoint
    facing_direction: np.ndarray  # shape (2,) — unit vector, shoulder→shoulder direction


@dataclass
class PairFeatures:
    """
    Spatial relationship features between two people.
    All values are in normalized units (torso lengths) or radians.
    No hardcoded thresholds — just raw geometry for the model to learn from.
    """
    id_a: int
    id_b: int

    # Distance between hip centers (in torso-length units of person A)
    distance_norm: float

    # Angle between the vector A→B and person A's facing direction (radians)
    # Near 0 = A facing B, near π = A facing away from B
    facing_angle_a: float

    # Same for person B
    facing_angle_b: float

    # Relative keypoints: for each of B's 17 joints, position relative to A's hip center
    # Shape (17, 2) — gives the model full spatial context of B relative to A
    relative_keypoints_b: np.ndarray

    # Whether the bounding boxes overlap (rough body contact indicator)
    bbox_overlap_ratio: float


MIN_TORSO_LENGTH = 20.0   # pixels — below this the person is too small/far to be useful
FALLBACK_TORSO = 100.0    # used when torso joints are not detected (low confidence)


class Normalizer:
    """
    Normalizes raw keypoints and computes inter-person features.

    Usage:
        norm = Normalizer(min_keypoint_conf=0.3)
        normalized = norm.normalize_pose(track)
        pair_feats = norm.compute_pair_features(norm_a, norm_b, bbox_a, bbox_b)
    """

    def __init__(self, min_keypoint_conf: float = 0.3):
        """
        Args:
            min_keypoint_conf: keypoints below this confidence are zeroed out
                               (treated as not visible) rather than using noisy coords
        """
        self.min_keypoint_conf = min_keypoint_conf

    def normalize_pose(self, track_id: int, keypoints_px: np.ndarray) -> NormalizedPose:
        """
        Normalize a single person's keypoints.

        Args:
            track_id: stable integer ID from PersonTracker
            keypoints_px: (17, 3) array of [x, y, confidence] in pixel coords

        Returns:
            NormalizedPose with keypoints centered and scaled to torso units.
        """
        kpts = keypoints_px.copy()

        # Zero out low-confidence keypoints (don't use noisy coords)
        low_conf_mask = kpts[:, 2] < self.min_keypoint_conf
        kpts[low_conf_mask, :2] = 0.0

        # Compute hip center (translation anchor)
        hip_center = self._joint_midpoint(kpts, IDX_LEFT_HIP, IDX_RIGHT_HIP)

        # Compute shoulder center (for torso length and facing direction)
        shoulder_center = self._joint_midpoint(kpts, IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER)

        # Torso length = distance hip_center → shoulder_center
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        if torso_length < MIN_TORSO_LENGTH:
            torso_length = FALLBACK_TORSO

        # Facing direction: left→right shoulder vector (perpendicular to forward)
        left_shoulder = kpts[IDX_LEFT_SHOULDER, :2]
        right_shoulder = kpts[IDX_RIGHT_SHOULDER, :2]
        if kpts[IDX_LEFT_SHOULDER, 2] > self.min_keypoint_conf and \
           kpts[IDX_RIGHT_SHOULDER, 2] > self.min_keypoint_conf:
            facing_vec = right_shoulder - left_shoulder
            facing_norm = np.linalg.norm(facing_vec)
            if facing_norm > 1e-6:
                facing_direction = facing_vec / facing_norm
            else:
                facing_direction = np.array([1.0, 0.0])
        else:
            facing_direction = np.array([1.0, 0.0])

        # Normalize: translate to hip center, scale by torso length
        kpts_norm = kpts.copy()
        # Only translate/scale the (x, y) — leave confidence as-is
        kpts_norm[:, :2] = (kpts[:, :2] - hip_center) / torso_length
        # Keep low-conf joints zeroed (don't scale-up noise)
        kpts_norm[low_conf_mask, :2] = 0.0

        return NormalizedPose(
            track_id=track_id,
            keypoints_norm=kpts_norm,
            torso_length=torso_length,
            hip_center_px=hip_center,
            shoulder_center_px=shoulder_center,
            facing_direction=facing_direction,
        )

    def compute_pair_features(
        self,
        norm_a: NormalizedPose,
        norm_b: NormalizedPose,
        bbox_a: np.ndarray,
        bbox_b: np.ndarray,
    ) -> PairFeatures:
        """
        Compute inter-person spatial features between two normalized poses.

        Args:
            norm_a, norm_b: NormalizedPose for each person
            bbox_a, bbox_b: [x1, y1, x2, y2] pixel bboxes

        Returns:
            PairFeatures capturing the spatial relationship between A and B.
        """
        # Vector from A's hip center to B's hip center
        vec_a_to_b = norm_b.hip_center_px - norm_a.hip_center_px
        dist_px = np.linalg.norm(vec_a_to_b)
        distance_norm = dist_px / norm_a.torso_length

        # Facing angles
        if dist_px > 1e-6:
            unit_a_to_b = vec_a_to_b / dist_px
        else:
            unit_a_to_b = np.array([0.0, 0.0])

        facing_angle_a = self._angle_between(norm_a.facing_direction, unit_a_to_b)
        facing_angle_b = self._angle_between(norm_b.facing_direction, -unit_a_to_b)

        # B's keypoints relative to A's hip center, scaled by A's torso
        relative_kpts_b = (norm_b.keypoints_norm[:, :2] * norm_b.torso_length
                           + norm_b.hip_center_px - norm_a.hip_center_px) / norm_a.torso_length

        # BBox overlap ratio
        bbox_overlap_ratio = self._bbox_overlap_ratio(bbox_a, bbox_b)

        return PairFeatures(
            id_a=norm_a.track_id,
            id_b=norm_b.track_id,
            distance_norm=distance_norm,
            facing_angle_a=facing_angle_a,
            facing_angle_b=facing_angle_b,
            relative_keypoints_b=relative_kpts_b,
            bbox_overlap_ratio=bbox_overlap_ratio,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _joint_midpoint(kpts: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
        """Midpoint between two joints. Falls back to the available one if the other is missing."""
        conf_a = kpts[idx_a, 2]
        conf_b = kpts[idx_b, 2]
        if conf_a > 0 and conf_b > 0:
            return (kpts[idx_a, :2] + kpts[idx_b, :2]) / 2.0
        elif conf_a > 0:
            return kpts[idx_a, :2].copy()
        elif conf_b > 0:
            return kpts[idx_b, :2].copy()
        else:
            # Both invisible — use bbox-derived fallback (caller's responsibility)
            return np.array([0.0, 0.0])

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Angle between two 2D unit vectors in radians [0, π]."""
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return float(np.arccos(dot))

    @staticmethod
    def _bbox_overlap_ratio(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        """Intersection area as ratio of the smaller bbox area."""
        ix1 = max(bbox_a[0], bbox_b[0])
        iy1 = max(bbox_a[1], bbox_b[1])
        ix2 = min(bbox_a[2], bbox_b[2])
        iy2 = min(bbox_a[3], bbox_b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        smaller = min(area_a, area_b)
        return float(inter / (smaller + 1e-6))
