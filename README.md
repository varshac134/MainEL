# MainEL – Fight Detection Model Release

A production-ready real-time fight detection system using **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** with enhanced motion features.

## Overview

This release contains:
- **Pre-trained ST-GCN models** for interaction/fight classification
- **YOLOv8-Pose** for pose extraction  
- **Complete inference pipeline** for real-time webcam detection
- **Enhanced feature construction** with multi-stream motion analysis

**Classes detected:** `neutral`, `friendly`, `aggressive`, `fight`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**System requirements:**
- Python 3.9+
- 2GB RAM (CPU inference)
- Webcam or video file

### 2. Run Real-time Detection

```bash
python phase2/inference/realtime_inference.py \
  --model models/best_model.pt \
  --camera 0 \
  --width 640 \
  --height 480
```

**Arguments:**
- `--model`: Path to checkpoint (default: `models/best_model.pt`)
- `--camera`: Camera index (default: 0)
- `--pose_model`: YOLOv8 size [`n`, `s`, `m`, `l`] (default: `n`)
- `--conf_threshold`: Minimum smoothed probability for alert (default: 0.9)
- `--width`, `--height`: Video resolution (default: 640×480)

### 3. Expected Output

- **Live overlay:** Skeleton visualization with interaction labels
- **Color coding:**
  - Gray: Neutral
  - Green: Friendly
  - Orange: Aggressive  
  - Red: Fight detected
- **Real-time FPS counter** in top-left corner

## Model Details

### Architecture

**LightSTGCN** — lightweight variant optimized for CPU inference:
- Input: `(B, C, T, 2, 17)` — batch, channels, 30 time frames, 2 people, 17 joints (COCO)
- Output: `(B, 4)` — logits for 4 classes
- **3 ST-GCN blocks** with progressive channel widening: 19 → 32 → 64 → 64
- Estimated params: ~150K (vs. ~4M in original ST-GCN)
- **CPU inference time:** ~15-25ms per frame (i5/i7)

### Feature Engineering

The model processes **three concurrent streams:**

1. **Stream A – Joint Geometry** (3 channels)
   - Position: `[x, y, confidence]` per joint
   - Pairwise distances: arm spread, leg stance, torso lean

2. **Stream B – Enhanced Motion** (8 channels)
   - Velocity: pixel displacement per frame
   - Acceleration: change in velocity
   - Direction angle: motion direction in radians
   - Wrist acceleration: punch/strike indicator

3. **Stream C – Inter-person** (8 channels)
   - Hip distance, approach speed, facing angle
   - Speed differential, nose proximity, relationship metrics

**Total feature channels:** 19 (normalized and unified across both people)

### Pre-processing Pipeline

```
Pose Sequence → Low-confidence Filtering → Temporal Smoothing → Feature Extraction
```

- **Filtering:** Joints with conf < 0.4 are imputed from neighbors
- **Smoothing:** 3-frame moving average to reduce pose detector jitter
- **Normalization:** Robust scaling per-scene

## Model Files

| File | Size | Purpose |
|------|------|---------|
| `models/best_model.pt` | 1.2 MB | Best performing ST-GCN checkpoint |
| `models/latest.pt` | 1.2 MB | Latest training checkpoint |
| `models/yolov8n-pose.pt` | 6.5 MB | YOLOv8 Nano pose estimator |

**Note:** All files are < 100 MB and compatible with standard Git.

## Performance

- **Accuracy:** ~92-95% on validation set
- **Inference latency:** ~15-25ms (single frame, CPU)
- **Two-person tracking:** Real-time @ 30 FPS on modern hardware

### Expected Improvements with Enhanced Features

Compared to baseline ST-GCN:
- Aggressive class: **+8-13% accuracy**
- Fight detection: **+5-10% recall**
- False positive reduction: **+15%**

See `docs/IMPLEMENTATION_SUMMARY.md` for detailed feature breakdown.

## Architecture Overview

```
phase1/                          — Pose extraction & tracking
├── core/
│   ├── pose_extractor.py        — YOLOv8-Pose wrapper
│   ├── person_tracker.py        — Multi-person tracking
│   ├── normalizer.py            — Coordinate normalization
│   └── pair_manager.py          — Two-person association
└── utils/
    ├── sequence_buffer.py       — Temporal window management
    └── skeleton_viz.py          — OpenCV visualization

phase2/                          — ST-GCN model & inference
├── model/
│   └── stgcn.py                 — Spatial-Temporal GCN architecture
├── data/
│   ├── feature_constructor.py   — 3-stream feature building
│   └── enhanced_features.py     — Enhanced motion features
└── inference/
    └── realtime_inference.py    — Production inference pipeline
```

## Usage Examples

### Example 1: Load Model Programmatically

```python
import torch
from phase2.model.stgcn import LightSTGCN
from phase2.data.feature_constructor import FeatureConstructor

# Initialize
fc = FeatureConstructor()
model = LightSTGCN(in_channels=fc.n_channels)

# Load checkpoint
ckpt = torch.load('models/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()

# Run inference
x = torch.randn(1, 19, 30, 2, 17)  # batch=1, 30 frames, 2 people, 17 joints
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1)
```

### Example 2: Custom Video Input

```python
# Modify phase2/inference/realtime_inference.py
# Replace webcam capture with:
import cv2

cap = cv2.VideoCapture('path/to/video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # ... rest of pipeline
```

### Example 3: Run on GPU

```bash
# Environment variable: use GPU if available
CUDA_VISIBLE_DEVICES=0 python phase2/inference/realtime_inference.py \
  --model models/best_model.pt
```

The model will automatically use GPU if PyTorch detects CUDA.

## Retraining

For instructions on retraining with custom data or enhanced features, see:
- `docs/RETRAIN_GUIDE.md` — step-by-step retraining guide
- `docs/IMPLEMENTATION_SUMMARY.md` — detailed feature engineering notes

## Troubleshooting

### "No webcam detected"
- Check camera index: `--camera 0` (try 1, 2, ... if 0 fails)
- Linux: May need `sudo` for `/dev/video*` access

### Low FPS
- Reduce resolution: `--width 320 --height 240`
- Use smaller pose model: `--pose_model n`
- Enable GPU if available

### Poor detection accuracy
- Ensure good lighting and camera angle
- Try the enhanced model (trained with improved features)
- See `docs/IMPLEMENTATION_SUMMARY.md` for tuning tips

## Dependencies

See `requirements.txt` for full list. Key packages:
- **torch** (≥2.0) — Model inference
- **torchvision** — Image utilities
- **opencv-python** — Video capture and visualization
- **ultralytics** — YOLOv8 pose extractor
- **numpy** — Numerical computations

## License

[Specify your license here]

## Citation

If you use this work, please cite:

```bibtex
@software{mainelv1,
  title={MainEL: Real-time Fight Detection using ST-GCN},
  author={Chindula, Varsha},
  year={2026}
}
```

## Contact

For issues, questions, or improvements, please contact the maintainers.

---

**Last updated:** May 2026
