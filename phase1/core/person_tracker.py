"""
person_tracker.py
-----------------
Lightweight IoU-based tracker for maintaining stable person IDs across frames.

This is a simplified ByteTrack-inspired implementation — no dependencies on
external tracking libraries (which have complex installs). Uses IoU matching
with a short Kalman-free dead-zone buffer (tracks survive N missed frames).

For Phase 1, this is sufficient. If you need better ID stability in crowded
scenes (3+ people), you can swap in full ByteTrack or BoTrack later — the
interface is identical.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from phase1.core.pose_extractor import RawDetection


@dataclass
class Track:
    """A single tracked person with history."""
    track_id: int
    bbox: np.ndarray           # current bbox [x1, y1, x2, y2]
    keypoints: np.ndarray      # current keypoints (17, 3)
    det_confidence: float
    age: int = 0               # frames since first seen
    missed_frames: int = 0     # consecutive frames without a match
    is_active: bool = True

    # Short bbox history for velocity estimation (last 5 centroids)
    centroid_history: list[np.ndarray] = field(default_factory=list)

    @property
    def centroid(self) -> np.ndarray:
        """Center of current bbox."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])

    @property
    def velocity(self) -> np.ndarray:
        """Approximate pixel/frame velocity from centroid history."""
        if len(self.centroid_history) < 2:
            return np.zeros(2)
        return self.centroid_history[-1] - self.centroid_history[-2]


class PersonTracker:
    """
    Assigns stable IDs to persons across frames using IoU matching.

    Usage:
        tracker = PersonTracker(max_missed=10, min_iou=0.3)
        tracks = tracker.update(detections)   # call each frame
    """

    def __init__(
        self,
        max_missed: int = 10,
        min_iou: float = 0.3,
        max_tracks: int = 10,
    ):
        """
        Args:
            max_missed: frames a track survives without a detection match
            min_iou: minimum IoU to consider a detection a match for a track
            max_tracks: safety cap on simultaneous tracks (fight scenes: 2-4)
        """
        self.max_missed = max_missed
        self.min_iou = min_iou
        self.max_tracks = max_tracks
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list[RawDetection]) -> list[Track]:
        """
        Match detections to existing tracks and return active tracks.

        Args:
            detections: list of RawDetection from PoseExtractor.run()

        Returns:
            List of active Track objects, each with a stable track_id.
        """
        active_tracks = [t for t in self._tracks.values() if t.is_active]

        if not detections:
            # No detections this frame — age all tracks
            for t in active_tracks:
                t.missed_frames += 1
                if t.missed_frames > self.max_missed:
                    t.is_active = False
            return [t for t in active_tracks if t.is_active]

        if not active_tracks:
            # No existing tracks — create one per detection
            for det in detections[:self.max_tracks]:
                self._spawn_track(det)
            return list(self._tracks.values())

        # Build IoU cost matrix: rows = tracks, cols = detections
        iou_matrix = self._build_iou_matrix(active_tracks, detections)

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        # Greedy matching: best IoU pairs first
        if iou_matrix.size > 0:
            flat_order = np.argsort(-iou_matrix, axis=None)
            for flat_idx in flat_order:
                row = flat_idx // iou_matrix.shape[1]
                col = flat_idx %  iou_matrix.shape[1]
                if iou_matrix[row, col] < self.min_iou:
                    break
                if row in matched_track_ids or col in matched_det_indices:
                    continue
                # Match: update track
                track = active_tracks[row]
                det = detections[col]
                track.bbox = det.bbox
                track.keypoints = det.keypoints
                track.det_confidence = det.det_confidence
                track.missed_frames = 0
                track.age += 1
                track.centroid_history.append(track.centroid)
                if len(track.centroid_history) > 5:
                    track.centroid_history.pop(0)
                matched_track_ids.add(row)
                matched_det_indices.add(col)

        # Unmatched tracks — increment missed counter
        for i, t in enumerate(active_tracks):
            if i not in matched_track_ids:
                t.missed_frames += 1
                if t.missed_frames > self.max_missed:
                    t.is_active = False

        # Unmatched detections — spawn new tracks (up to max_tracks)
        current_count = sum(1 for t in self._tracks.values() if t.is_active)
        for j, det in enumerate(detections):
            if j not in matched_det_indices and current_count < self.max_tracks:
                self._spawn_track(det)
                current_count += 1

        return [t for t in self._tracks.values() if t.is_active]

    def reset(self):
        """Clear all tracks (call between video clips)."""
        self._tracks.clear()
        self._next_id = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_track(self, det: RawDetection) -> Track:
        t = Track(
            track_id=self._next_id,
            bbox=det.bbox.copy(),
            keypoints=det.keypoints.copy(),
            det_confidence=det.det_confidence,
        )
        t.centroid_history.append(t.centroid)
        self._tracks[self._next_id] = t
        self._next_id += 1
        return t

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """IoU between two boxes [x1, y1, x2, y2]."""
        ix1 = max(box_a[0], box_b[0])
        iy1 = max(box_a[1], box_b[1])
        ix2 = min(box_a[2], box_b[2])
        iy2 = min(box_a[3], box_b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _build_iou_matrix(
        self,
        tracks: list[Track],
        detections: list[RawDetection],
    ) -> np.ndarray:
        mat = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                mat[i, j] = self._iou(t.bbox, d.bbox)
        return mat
