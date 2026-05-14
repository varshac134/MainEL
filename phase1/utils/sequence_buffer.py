"""
sequence_buffer.py
------------------
Sliding window buffer that accumulates pose frames and emits
fixed-length sequences for model inference.

Output tensor shape: (T, 2, 17, C)
  T = window length (default 30 frames = ~1 second at 30fps)
  2 = person A, person B
  17 = COCO keypoints
  C = feature channels per keypoint

Feature channels (C = 6):
  [0] x  — normalized x coordinate
  [1] y  — normalized y coordinate
  [2] c  — keypoint confidence
  [3] vx — x velocity (Δx from previous frame)
  [4] vy — y velocity (Δy from previous frame)
  [5] s  — keypoint speed (magnitude of velocity)

This tensor is the direct input to the ST-GCN in Phase 2.
"""

from __future__ import annotations
from collections import deque
import numpy as np
from phase1.core.pair_manager import InteractionPair


# Number of feature channels per keypoint
N_CHANNELS = 6
N_KEYPOINTS = 17


class SequenceBuffer:
    """
    Maintains a sliding window of interaction pairs and emits
    (T, 2, 17, C) tensors when the buffer is full.

    Window size is 45 frames (~1.5s at 30fps).
    The extra length lets the model see the full approach→contact→reaction arc,
    which is critical for distinguishing aggressive buildup from friendly greeting.

    Usage:
        buf = SequenceBuffer(window_size=45, step=1)
        tensor = buf.push(pair)     # returns tensor when full, else None
        buf.reset()                 # call between clips
    """

    def __init__(self, window_size: int = 45, step: int = 1):
        """
        Args:
            window_size: T — number of frames per sequence
            step: emit a new sequence every `step` frames (1 = every frame, sliding window)
        """
        self.window_size = window_size
        self.step = step

        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        # Store previous frame's keypoints for velocity computation
        self._prev_kpts: dict[int, np.ndarray] = {}   # track_id → (17, 2)
        self._frame_count = 0

    def push(self, pair: InteractionPair) -> np.ndarray | None:
        """
        Add one frame to the buffer.

        Args:
            pair: InteractionPair from PairManager.update()

        Returns:
            np.ndarray of shape (T, 2, 17, C) when buffer is full,
            None otherwise.
        """
        frame_tensor = self._encode_pair(pair)
        self._buffer.append(frame_tensor)
        self._frame_count += 1

        if len(self._buffer) == self.window_size and \
           (self._frame_count % self.step) == 0:
            return np.stack(list(self._buffer), axis=0)   # (T, 2, 17, C)

        return None

    def reset(self):
        """Clear the buffer — call between video clips or when person IDs change."""
        self._buffer.clear()
        self._prev_kpts.clear()
        self._frame_count = 0

    def get_partial(self) -> np.ndarray | None:
        """
        Return whatever is in the buffer (padded with zeros if < window_size).
        Useful for end-of-video processing.
        """
        if len(self._buffer) == 0:
            return None
        frames = list(self._buffer)
        while len(frames) < self.window_size:
            frames.insert(0, np.zeros_like(frames[0]))
        return np.stack(frames, axis=0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode_pair(self, pair: InteractionPair) -> np.ndarray:
        """
        Encode one frame as a (2, 17, C) tensor.
        """
        result = np.zeros((2, N_KEYPOINTS, N_CHANNELS), dtype=np.float32)

        for person_idx, (norm, track_id) in enumerate([
            (pair.norm_a, pair.track_id_a),
            (pair.norm_b, pair.track_id_b),
        ]):
            kpts = norm.keypoints_norm   # (17, 3) — [x, y, conf]

            # Channels 0-2: x, y, confidence
            result[person_idx, :, 0] = kpts[:, 0]
            result[person_idx, :, 1] = kpts[:, 1]
            result[person_idx, :, 2] = kpts[:, 2]

            # Channels 3-5: velocity (computed from previous frame)
            prev = self._prev_kpts.get(track_id)
            if prev is not None:
                vx = kpts[:, 0] - prev[:, 0]
                vy = kpts[:, 1] - prev[:, 1]
                result[person_idx, :, 3] = vx
                result[person_idx, :, 4] = vy
                result[person_idx, :, 5] = np.sqrt(vx**2 + vy**2)

            # Store for next frame's velocity computation
            self._prev_kpts[track_id] = kpts.copy()

        return result
