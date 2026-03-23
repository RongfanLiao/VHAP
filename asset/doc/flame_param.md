# `flame_param.npz` — FLAME Parameter File

This file stores FLAME 3D morphable head model parameters exported by `FLAMEParamDatasetWriter` (see [export_flame_params.py](../../../../../../vhap/export_flame_params.py)). It contains 224 frames of tracked parameters for a single subject.

All values are `float32` unless noted otherwise. Rotations are in **axis-angle** representation.

## Per-Frame Dynamic Parameters

These are indexed by frame (224 frames in this clip).

| Key | Shape | Description |
|---|---|---|
| `translation` | `(224, 3)` | Global head translation (x, y, z) per frame, **zero-centred** (mean subtracted). |
| `rotation` | `(224, 3)` | Global head rotation (axis-angle, 3 DoF) per frame. |
| `neck_pose` | `(224, 3)` | Neck joint rotation (axis-angle, 3 DoF) per frame. |
| `jaw_pose` | `(224, 3)` | Jaw joint rotation (axis-angle, 3 DoF) per frame. The x-component controls mouth opening. |
| `eyes_pose` | `(224, 6)` | Eye gaze rotations — 3 DoF per eye (left eye + right eye). |
| `expr` | `(224, 100)` | Expression blend-shape coefficients (100-dim FLAME expression space) per frame. |

## Static Parameters

| Key | Shape | Description |
|---|---|---|
| `shape` | `(300,)` | FLAME identity shape coefficients (300-dim PCA space). Shared across all frames — this defines the subject's face geometry. |

## Canonical (Neutral-Pose) Parameters

These define a **reference neutral pose** used by downstream renderers (e.g. LAM). All values are single-frame (`(1, ...)`).

| Key | Shape | Value | Description |
|---|---|---|---|
| `canonical_translation` | `(1, 3)` | `[0, 0, 0]` | Zero translation — face at origin. |
| `canonical_rotation` | `(1, 3)` | `[0, 0, 0]` | Zero rotation — face looking straight ahead. |
| `canonical_neck_pose` | `(1, 3)` | `[0, 0, 0]` | Zero neck rotation. |
| `canonical_jaw_pose` | `(1, 3)` | `[0.3, 0, 0]` | Slightly open mouth (0.3 rad ≈ 17°). Prevents inner-mouth artifacts in rendering. (dtype: `float64`) |
| `canonical_eyes_pose` | `(1, 6)` | `[0, 0, 0, 0, 0, 0]` | Eyes looking straight ahead. |
| `canonical_expr` | `(1, 100)` | all zeros | Neutral expression (no blend-shape activation). |

## How It Is Generated

The pipeline in [video_to_flame_param.py](../../../../../../tools/video_to_flame_param.py) runs three steps:

1. **Preprocess** — Extract video frames at target FPS, run foreground matting, detect 2D face landmarks.
2. **Track** — Multi-stage FLAME optimisation fits the 3D model to the 2D landmarks and RGB images across all frames.
3. **Export** — `FLAMEParamDatasetWriter` selects per-frame parameters by timestep index and writes this `.npz` file.

## Usage

```python
import numpy as np

data = np.load("flame_param.npz")

# Per-frame parameters
translation = data["translation"]   # (N, 3)
rotation    = data["rotation"]      # (N, 3)
expression  = data["expr"]          # (N, 100)

# Identity (shared across frames)
shape = data["shape"]               # (300,)

# Canonical pose (for rendering neutral face)
canonical_jaw = data["canonical_jaw_pose"]  # (1, 3)
```
