"""
realtime_inference.py
---------------------
Plugs the trained ST-GCN into the Phase 1 webcam loop.

Full pipeline:
  webcam → YOLOv8-Pose → tracker → normalizer → pair_manager
        → sequence_buffer → feature_constructor → ST-GCN → EMA smoother
        → OpenCV overlay

Run:
  python phase2/inference/realtime_inference.py --model checkpoints/best_model.pt
"""

from __future__ import annotations
import argparse
import time
import sys
import os
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "phase1_updated (2)"))

from phase1.core.pose_extractor import PoseExtractor
from phase1.core.person_tracker import PersonTracker
from phase1.core.normalizer import Normalizer
from phase1.core.pair_manager import PairManager
from phase1.utils.sequence_buffer import SequenceBuffer
from phase1.utils.skeleton_viz import (
    draw_interaction_pair, draw_track, draw_fps,
    draw_no_detection, draw_label,
)
from phase2.data.feature_constructor import FeatureConstructor
from phase2.data.enhanced_features import EnhancedFeatureConstructor
from phase2.model.stgcn import LightSTGCN

CLASSES = ["neutral", "friendly", "aggressive", "fight"]
CLASS_COLORS = {
    "neutral":    (180, 180, 180),   # gray
    "friendly":   (80,  200,  80),   # green
    "aggressive": (30,  140, 255),   # orange
    "fight":      (50,   50, 220),   # red
}
ALERT_CLASSES = {"aggressive", "fight"}


class EMASmoother:
    """
    Exponential moving average over classifier output probabilities.
    Prevents flickering — prediction must stay high for ~0.5s to trigger.
    """

    def __init__(self, alpha: float = 0.3, n_classes: int = 4):
        self.alpha = alpha
        self.state = np.ones(n_classes, dtype=np.float32) / n_classes

    def update(self, probs: np.ndarray) -> np.ndarray:
        self.state = self.alpha * probs + (1 - self.alpha) * self.state
        return self.state

    def reset(self):
        self.state = np.ones(len(self.state), dtype=np.float32) / len(self.state)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to best_model.pt checkpoint")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--pose_model", default="n", choices=["n", "s", "m", "l"])
    p.add_argument("--conf_threshold", type=float, default=0.9,
                   help="Minimum smoothed probability to display an alert label (aggressive/fight)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p.parse_args()


def load_model(checkpoint_path: str) -> LightSTGCN:
    fc = FeatureConstructor()
    model = LightSTGCN(in_channels=fc.n_channels)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[model] Loaded from {checkpoint_path} (epoch {ckpt.get('epoch', '?')}, "
          f"val_acc={ckpt.get('val_acc', 0):.3f})")
    return model


def main():
    args = parse_args()

    print("[Phase 2] Loading components...")
    model       = load_model(args.model)
    extractor   = PoseExtractor(model_size=args.pose_model)
    tracker     = PersonTracker(max_missed=10)
    normalizer  = Normalizer()
    pair_mgr    = PairManager(normalizer)
    seq_buffer  = SequenceBuffer(window_size=30, step=1)
    # Use enhanced features for better motion understanding
    fc          = EnhancedFeatureConstructor(apply_smoothing=True)
    smoother    = EMASmoother(alpha=0.3)
    
    print(f"[features] Using ENHANCED features (velocity, acceleration, direction angles)")
    print(f"[features] Preprocessing: Low-conf filtering + temporal smoothing")

    def open_camera(source):
        """Try several OpenCV back‑ends to open a webcam or video file.
        Order of attempts:
          1️⃣ DirectShow (CAP_DSHOW) with integer index
          2️⃣ Media Foundation (CAP_MSMF) with integer index
          3️⃣ Default backend with integer index
          4️⃣ Treat source as a file path (any backend)
          5️⃣ Scan indices 0‑4 as a last resort
        Returns a cv2.VideoCapture object or None.
        """
        # Helper to try a specific backend flag
        def try_backend(idx, flag=None):
            if flag is not None:
                cap = cv2.VideoCapture(idx, flag)
            else:
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                name = "DirectShow" if flag == cv2.CAP_DSHOW else (
                    "MediaFoundation" if flag == cv2.CAP_MSMF else "default")
                print(f"[INFO] Opened camera {idx} using {name} backend.")
                return cap
            cap.release()
            return None

        # 1️⃣ DirectShow with integer index
        try:
            idx = int(source)
            cap = try_backend(idx, cv2.CAP_DSHOW)
            if cap:
                return cap
        except (ValueError, TypeError):
            pass

        # 2️⃣ Media Foundation (MSMF) with integer index (sometimes works when DSHOW fails)
        try:
            idx = int(source)
            cap = try_backend(idx, cv2.CAP_MSMF)
            if cap:
                return cap
        except (ValueError, TypeError):
            pass

        # 3️⃣ Default backend with integer index
        try:
            idx = int(source)
            cap = try_backend(idx)
            if cap:
                return cap
        except (ValueError, TypeError):
            pass

        # 4️⃣ Treat source as a file path (could be video file or URL)
        cap = cv2.VideoCapture(str(source))
        if cap.isOpened():
            print(f"[INFO] Opened video file or stream '{source}'.")
            return cap
        cap.release()

        # 5️⃣ Scan a few common indices as a last resort
        for i in range(5):
            cap = try_backend(i)
            if cap:
                return cap

        # Nothing worked
        print("[WARN] No camera or video source found. Falling back to dummy black frames.")
        return None

    cap = open_camera(args.camera)
    if cap is None:
        # Create a dummy frame generator that yields a black image of the requested size
        dummy_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        def dummy_gen():
            while True:
                yield True, dummy_frame.copy()
        cap = dummy_gen()  # will be used as an iterator later
        use_dummy = True
    else:
        use_dummy = False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not use_dummy:
        print("[INFO] Warming up camera...")
        for _ in range(10):
            cap.read()
            time.sleep(0.01)

    print("[Phase 2] Running. Press 'q' to quit.")

    fps_hist = []
    current_label = "neutral"
    current_probs = np.ones(4) / 4
    last_seq_time = 0.0

    while True:
        t0 = time.perf_counter()
        # Read frame depending on source type
        if use_dummy:
            # cap is a generator yielding (ret, frame)
            ret, frame = next(cap)
        else:
            ret, frame = cap.read()
        if not ret:
            break

        # Phase 1: pose + tracking + pairing
        detections = extractor.run(frame)
        tracks     = tracker.update(detections)
        pair       = pair_mgr.update(tracks)

        sequence = None
        if pair is not None:
            sequence = seq_buffer.push(pair)

        # Phase 2: inference on new sequence
        if sequence is not None:
            features = fc.construct(sequence)                          # (T, 2, 17, C) with enhanced motion
            x = torch.from_numpy(
                features.transpose(3, 0, 1, 2)[np.newaxis].astype(np.float32)
            )                                                      # (1, C, T, 2, 17)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()

            smoothed = smoother.update(probs)
            current_probs = smoothed
            
            raw_label = CLASSES[smoothed.argmax()]
            top_conf = float(smoothed.max())
            
            # ------------------------------------------------------------------
            # Heuristics & Confidence Filtering
            # ------------------------------------------------------------------
            # Calculate distance and motion from the final few frames
            # features shape: (T, 2, 17, C). C=0,1 are X,Y coordinates.
            recent_A = features[-10:, 0, :, 0:2]  # Person A last 10 frames
            recent_B = features[-10:, 1, :, 0:2]  # Person B last 10 frames
            
            # Inter-person distance (Torso centers: shoulders 5,6 and hips 11,12)
            center_A = np.mean(recent_A[:, [5, 6, 11, 12], :], axis=(0, 1))
            center_B = np.mean(recent_B[:, [5, 6, 11, 12], :], axis=(0, 1))
            inter_person_dist = np.linalg.norm(center_A - center_B)
            
            # Maximum wrist velocity (Wrists: 9, 10)
            wrist_vel_A = np.max(np.linalg.norm(np.diff(recent_A[:, [9, 10], :], axis=0), axis=-1)) if recent_A.shape[0] > 1 else 0.0
            wrist_vel_B = np.max(np.linalg.norm(np.diff(recent_B[:, [9, 10], :], axis=0), axis=-1)) if recent_B.shape[0] > 1 else 0.0
            max_wrist_vel = max(wrist_vel_A, wrist_vel_B)

            # Apply Strict Rules based on definitions:
            # Fight: requires closeness (collision) and high chaotic motion
            if raw_label == "fight":
                if inter_person_dist > 1.5 or max_wrist_vel < 0.05:
                    raw_label = "aggressive"
            
            # Aggressive: requires higher velocity (sudden arm motion) or specific threatening gestures
            # Calculate distance change over the last 10 frames (Approaching rapidly)
            start_center_A = np.mean(recent_A[0, [5, 6, 11, 12], :], axis=0)
            start_center_B = np.mean(recent_B[0, [5, 6, 11, 12], :], axis=0)
            start_dist = np.linalg.norm(start_center_A - start_center_B)
            approach_speed = start_dist - inter_person_dist  # positive means they are getting closer

            shoulder_y_A = np.mean(recent_A[-1, [5, 6], 1])
            shoulder_y_B = np.mean(recent_B[-1, [5, 6], 1])

            is_raised_hands = False
            is_approaching = False

            # Check for Raised Hands (at least one person has both wrists above their shoulder)
            if (recent_A[-1, 9, 1] < shoulder_y_A and recent_A[-1, 10, 1] < shoulder_y_A) or \
               (recent_B[-1, 9, 1] < shoulder_y_B and recent_B[-1, 10, 1] < shoulder_y_B):
                is_raised_hands = True

            # Check for rapid approach
            if approach_speed > 0.15 and max_wrist_vel > 0.03:
                is_approaching = True

            if raw_label == "aggressive":
                if max_wrist_vel < 0.02 and inter_person_dist > 1.0 and not is_raised_hands:
                    raw_label = "neutral"
                elif is_raised_hands:
                    gesture_sublabel = " (Raised Hands)"
                elif is_approaching:
                    gesture_sublabel = " (Approaching)"
                elif max_wrist_vel > 0.08:
                    gesture_sublabel = " (Sudden Motion)"
            elif raw_label == "neutral" and (is_raised_hands or is_approaching):
                raw_label = "aggressive"
                gesture_sublabel = " (Raised Hands)" if is_raised_hands else " (Approaching)"
            
            # Friendly: requires low velocity and closeness
            if raw_label == "friendly":
                if max_wrist_vel > 0.08:
                    raw_label = "aggressive"
                elif inter_person_dist > 2.0:
                    raw_label = "neutral"

            # ------------------------------------------------------------------
            # High-Five and Handshake Explicit Detection
            # ------------------------------------------------------------------
            gesture_sublabel = ""
            
            # Compare wrist distances across persons (9=L_Wrist, 10=R_Wrist)
            w_A_L, w_A_R = recent_A[-1, 9], recent_A[-1, 10]
            w_B_L, w_B_R = recent_B[-1, 9], recent_B[-1, 10]
            
            dist_LL = np.linalg.norm(w_A_L - w_B_L)
            dist_RR = np.linalg.norm(w_A_R - w_B_R)
            dist_LR = np.linalg.norm(w_A_L - w_B_R)
            dist_RL = np.linalg.norm(w_A_R - w_B_L)
            
            dists = [dist_LL, dist_RR, dist_LR, dist_RL]
            min_wrist_dist = min(dists)
            
            # If wrists are touching/extremely close
            if min_wrist_dist < 0.35:
                # Average shoulder Y (smaller Y is higher up)
                shoulder_y_A = np.mean(recent_A[-1, [5, 6], 1])
                shoulder_y_B = np.mean(recent_B[-1, [5, 6], 1])
                avg_shoulder_y = (shoulder_y_A + shoulder_y_B) / 2
                
                # Average hip Y
                hip_y_A = np.mean(recent_A[-1, [11, 12], 1])
                hip_y_B = np.mean(recent_B[-1, [11, 12], 1])
                avg_hip_y = (hip_y_A + hip_y_B) / 2
                
                interacting_idx = np.argmin(dists)
                if interacting_idx == 0: interact_y = (w_A_L[1] + w_B_L[1]) / 2
                elif interacting_idx == 1: interact_y = (w_A_R[1] + w_B_R[1]) / 2
                elif interacting_idx == 2: interact_y = (w_A_L[1] + w_B_R[1]) / 2
                else: interact_y = (w_A_R[1] + w_B_L[1]) / 2
                
                # If interaction is above shoulders, it's a high-five
                if interact_y < avg_shoulder_y:
                    raw_label = "friendly"
                    gesture_sublabel = " (High-Five)"
                # If interaction is between shoulders and hips, it's a handshake
                elif interact_y >= avg_shoulder_y and interact_y < avg_hip_y:
                    raw_label = "friendly"
                    gesture_sublabel = " (Handshake)"

            # Apply Confidence Filtering:
            # If the model predicts aggressive/fight but confidence is low, 
            # we downgrade it to 'neutral' to avoid false positives.
            if raw_label in ALERT_CLASSES and top_conf < args.conf_threshold:
                current_label = "neutral"
            else:
                current_label = raw_label
                
            last_seq_time = time.perf_counter()

        # Fade label if no sequence for >2s (person left frame)
        if time.perf_counter() - last_seq_time > 2.0:
            smoother.reset()
            current_label = "neutral"
            current_probs = np.ones(4) / 4

        # ---- Visualization ----
        vis = frame.copy()

        # Draw skeletons / pair
        if pair is not None:
            track_map = {t.track_id: t for t in tracks}
            t_a = track_map.get(pair.track_id_a)
            t_b = track_map.get(pair.track_id_b)
            if t_a and t_b:
                draw_interaction_pair(vis, pair, t_a, t_b)
        else:
            for t in tracks:
                draw_track(vis, t, (180, 180, 180))
            if not tracks:
                draw_no_detection(vis)

        # Main prediction label
        color = CLASS_COLORS[current_label]
        conf = float(current_probs[CLASSES.index(current_label)])

        # Alert flash: red border for aggressive/fight
        # We only get here for ALERT_CLASSES if confidence was already >= conf_threshold due to filtering above
        if current_label in ALERT_CLASSES:
            intensity = int(min(conf * 255, 200))
            cv2.rectangle(vis, (0, 0), (args.width - 1, args.height - 1),
                          (0, 0, intensity), 4)

        # Prediction banner
        display_label = current_label.upper() + gesture_sublabel if 'gesture_sublabel' in locals() and current_label in ["friendly", "aggressive"] else current_label.upper()
        banner_text = f"{display_label}  {conf:.0%}"
        banner_h = 44
        cv2.rectangle(vis, (0, 0), (args.width, banner_h), (0, 0, 0), -1)
        draw_label(vis, banner_text, (10, banner_h - 10), color, font_scale=0.9, thickness=2)

        # Probability bars for all classes
        bar_x = args.width - 180
        for i, (cls_name, cls_color) in enumerate(CLASS_COLORS.items()):
            bar_y = 10 + i * 22
            prob = float(current_probs[i])
            bar_w = int(prob * 160)
            cv2.rectangle(vis, (bar_x, bar_y), (bar_x + 160, bar_y + 14), (40, 40, 40), -1)
            cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), cls_color, -1)
            draw_label(vis, f"{cls_name[:10]:10s} {prob:.2f}",
                       (bar_x - 165, bar_y + 12), (200, 200, 200), font_scale=0.38)

        # FPS
        fps_hist.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        if len(fps_hist) > 30:
            fps_hist.pop(0)
        draw_fps(vis, float(np.mean(fps_hist)))

        cv2.imshow("Fight Detection — Phase 2", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
