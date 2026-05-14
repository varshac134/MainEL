"""
pair_manager.py
---------------
Selects the primary two-person interaction pair from all active tracks.

In a fight detection scenario, we care about the two people who are:
  a) closest to each other, AND
  b) have been near each other for some time (not just passing by)

This class maintains pair stability — it won't flip A/B assignments
every frame just because detection order changed.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from phase1.core.person_tracker import Track
from phase1.core.normalizer import Normalizer, NormalizedPose, PairFeatures


@dataclass
class InteractionPair:
    """The primary two-person interaction pair for a given frame."""
    norm_a: NormalizedPose
    norm_b: NormalizedPose
    pair_features: PairFeatures
    track_id_a: int
    track_id_b: int
    proximity_frames: int      # consecutive frames this pair has been close


class PairManager:
    """
    Selects and maintains the primary interaction pair across frames.

    Usage:
        manager = PairManager(normalizer)
        pair = manager.update(active_tracks)   # returns InteractionPair or None
    """

    def __init__(
        self,
        normalizer: Normalizer,
        proximity_threshold_torso: float = 3.0,
        min_proximity_frames: int = 3,
    ):
        """
        Args:
            normalizer: Normalizer instance
            proximity_threshold_torso: distance (in torso units) below which
                two people are considered in proximity
            min_proximity_frames: minimum frames of proximity before emitting a pair
                (prevents spurious pairings from crossing pedestrians)
        """
        self.normalizer = normalizer
        self.proximity_threshold = proximity_threshold_torso
        self.min_proximity_frames = min_proximity_frames

        self._current_pair_ids: tuple[int, int] | None = None
        self._proximity_counter: dict[frozenset, int] = {}

    def update(self, tracks: list[Track]) -> InteractionPair | None:
        """
        Select the primary interaction pair from active tracks.

        Returns None if fewer than 2 people are visible or no pair is
        close enough to be considered an interaction.
        """
        if len(tracks) < 2:
            return None

        # Normalize all visible people
        normalized: list[tuple[Track, NormalizedPose]] = []
        for track in tracks:
            norm = self.normalizer.normalize_pose(track.track_id, track.keypoints)
            normalized.append((track, norm))

        # Find the closest pair
        best_pair: tuple[int, int] | None = None
        best_dist = float("inf")

        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                track_i, norm_i = normalized[i]
                track_j, norm_j = normalized[j]

                dist_px = np.linalg.norm(norm_i.hip_center_px - norm_j.hip_center_px)
                dist_norm = dist_px / max(norm_i.torso_length, norm_j.torso_length)

                if dist_norm < best_dist:
                    best_dist = dist_norm
                    best_pair = (track_i.track_id, track_j.track_id)

        if best_pair is None:
            return None

        pair_key = frozenset(best_pair)

        # Track how long this pair has been in proximity
        if best_dist <= self.proximity_threshold:
            self._proximity_counter[pair_key] = self._proximity_counter.get(pair_key, 0) + 1
        else:
            self._proximity_counter[pair_key] = max(
                0, self._proximity_counter.get(pair_key, 0) - 1
            )

        # Decay counters for other pairs (they moved apart)
        for k in list(self._proximity_counter.keys()):
            if k != pair_key:
                self._proximity_counter[k] = max(0, self._proximity_counter[k] - 2)

        prox_frames = self._proximity_counter.get(pair_key, 0)

        # Maintain stable A/B assignment (don't flip every frame)
        if self._current_pair_ids is not None:
            current_key = frozenset(self._current_pair_ids)
            if current_key == pair_key:
                # Same pair — keep existing A/B order
                id_a, id_b = self._current_pair_ids
            else:
                # New pair — assign in track_id order
                id_a, id_b = sorted(best_pair)
                self._current_pair_ids = (id_a, id_b)
        else:
            id_a, id_b = sorted(best_pair)
            self._current_pair_ids = (id_a, id_b)

        # Look up normalized poses by track_id
        norm_map = {t.track_id: n for t, n in normalized}
        if id_a not in norm_map or id_b not in norm_map:
            return None

        norm_a = norm_map[id_a]
        norm_b = norm_map[id_b]

        # Get bboxes
        track_map = {t.track_id: t for t, _ in normalized}
        bbox_a = track_map[id_a].bbox
        bbox_b = track_map[id_b].bbox

        pair_features = self.normalizer.compute_pair_features(norm_a, norm_b, bbox_a, bbox_b)

        return InteractionPair(
            norm_a=norm_a,
            norm_b=norm_b,
            pair_features=pair_features,
            track_id_a=id_a,
            track_id_b=id_b,
            proximity_frames=prox_frames,
        )
