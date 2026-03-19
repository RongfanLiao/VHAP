# Export Format

The VHAP pipeline exports tracked FLAME parameters as a lightweight dataset
for downstream consumers (e.g., LAM inference). This document defines the
output contract.

## Output Structure

```
<export_dir>/
├── transforms.json       # per-frame camera intrinsics & extrinsics
├── flame_param.npz       # consolidated FLAME parameters (all frames)
└── foreground_image.png  # reference image (first frame, white background)
```

## `transforms.json`

```jsonc
{
  // shared intrinsics (last frame's values, for compatibility)
  "cx": 307.0,
  "cy": 287.0,
  "fl_x": 1224.0,
  "fl_y": 1224.0,
  "h": 574,
  "w": 614,
  "camera_angle_x": 0.49,
  "camera_angle_y": 0.46,

  "timestep_indices": [0, 1, 2, ...],
  "camera_indices": [0],

  "frames": [
    {
      "timestep_index": 0,              // sequential index after extraction
      "timestep_index_original": 0,     // original frame index in source video
      "timestep_id": "000000",          // string ID matching image filename
      "camera_index": 0,
      "camera_id": "00",
      "cx": 307.0,                      // principal point x
      "cy": 287.0,                      // principal point y
      "fl_x": 1224.0,                   // focal length x (pixels)
      "fl_y": 1224.0,                   // focal length y (pixels)
      "h": 574,                         // image height
      "w": 614,                         // image width
      "camera_angle_x": 0.49,           // horizontal FoV (radians)
      "camera_angle_y": 0.46,           // vertical FoV (radians)
      "transform_matrix": [[...], ...]  // 4x4 camera-to-world, OpenGL convention
    },
    ...
  ]
}
```

### Camera Convention

- Extrinsics are **camera-to-world** (c2w) matrices in **OpenGL** convention
  (Y-up, -Z forward).
- All cameras are relocated so the FLAME mesh sequence is centred at the
  origin (mean translation subtracted).

## `flame_param.npz`

Loaded via `np.load("flame_param.npz")`. All arrays use `float32` unless noted.

### Per-frame parameters

| Key | Shape | Description |
|-----|-------|-------------|
| `translation` | `(N, 3)` | Global translation per frame |
| `rotation` | `(N, 3)` | Global rotation (axis-angle) per frame |
| `neck_pose` | `(N, 3)` | Neck pose (axis-angle) per frame |
| `jaw_pose` | `(N, 3)` | Jaw pose (axis-angle) per frame |
| `eyes_pose` | `(N, 6)` | Eye gaze (left + right, axis-angle) per frame |
| `expr` | `(N, 100)` | Expression coefficients per frame |

Where `N` = number of exported frames.

### Canonical (identity) parameters

| Key | Shape | Description |
|-----|-------|-------------|
| `shape` | `(300,)` | Identity shape coefficients (shared across all frames) |
| `canonical_translation` | `(1, 3)` | Zero translation |
| `canonical_rotation` | `(1, 3)` | Zero rotation |
| `canonical_neck_pose` | `(1, 3)` | Zero neck pose |
| `canonical_jaw_pose` | `(1, 3)` | `[0.3, 0, 0]` — slightly open mouth |
| `canonical_eyes_pose` | `(1, 6)` | Zero gaze |
| `canonical_expr` | `(1, 100)` | Zero expression |

### Notes

- Translation is already relocated (mean subtracted) to centre the mesh at origin.
- `canonical_jaw_pose` uses a fixed slight opening (`0.3 rad ≈ 17°`) as the
  neutral pose for rendering.
- Static offsets, dynamic offsets, texture, and lighting are **not** exported
  (not needed by downstream LAM inference).

## `foreground_image.png`

- RGB image of the first frame (`timestep_index == 0`).
- Composited on a **white** background using the alpha map.
- Resolution matches the source video frame size.
- Used as the reference/source image for LAM inference.
