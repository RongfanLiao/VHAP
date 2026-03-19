# VHAP vs GaussianAvatars

## Overview

| | **VHAP** (this repo) | **GaussianAvatars** |
|---|---|---|
| **Role** | Preprocessing / tracking | Avatar training / rendering |
| **Input** | Raw video | VHAP-exported data (images, FLAME params, masks) |
| **Output** | FLAME parameters, camera transforms, masks | Photorealistic 3D Gaussian head avatar |
| **Core technique** | Photometric optimization to fit FLAME mesh to video frames | 3D Gaussian Splatting rigged to FLAME mesh |
| **Rendering** | Differentiable mesh rasterization (nvdiffrast) for optimization | 3DGS splatting for real-time rendering |

## Pipeline

```
Video  ──VHAP──>  FLAME params + cameras + images  ──GaussianAvatars──>  Animatable 3DGS avatar
```

## What each repo does

### VHAP (Versatile Head Alignment with Adaptive Appearance Priors)

VHAP is a **FLAME-based head tracking and preprocessing tool**. It processes video sequences to extract precise FLAME parameters (shape, expression, pose, texture) for individual heads across frames.

Key capabilities:
- Detects facial landmarks using STAR or face-alignment
- Optimizes FLAME parameters through a multi-stage process (landmark-only, then texture-based refinement)
- Uses differentiable mesh rasterization (nvdiffrast) for photometric optimization
- Supports both monocular and multi-view video sequences
- Exports tracked FLAME parameters and camera transforms as lightweight datasets

VHAP does **NOT** generate 3DGS avatars. It is the upstream data preparation tool that answers: **"where is the face and how does it move?"**

### GaussianAvatars

GaussianAvatars creates **photorealistic head avatars with rigged 3D Gaussians**. It binds thousands of 3D Gaussians to the FLAME mesh faces, then optimizes their appearance from multi-view images.

Key capabilities:
- Photorealistic head avatar creation using 3D Gaussian Splatting
- Gaussians rigged to FLAME model for controllable animation
- Dynamic expression and pose control via FLAME parameters
- Novel view synthesis from arbitrary camera viewpoints
- Real-time rendering via an interactive viewer

GaussianAvatars answers: **"what does the face look like in full detail?"**

## Data flow

VHAP exports the following data that GaussianAvatars consumes:

```
export_folder/
├── transforms.json       # per-frame camera intrinsics & extrinsics
└── flame_param.npz       # FLAME params (shape, expr, rotation, ...)
                          # optionally contains tex_extra, lights,
                          # and canonical_* parameters
```

GaussianAvatars expects VHAP-preprocessed data in its transforms JSON format with per-frame entries containing:
- `timestep_index`: frame identifier
- `camera_index` / `camera_id`: view identifier
- `transform_matrix`: camera pose (4x4)
- `file_path`: image location
- `flame_param_path`: FLAME parameters (.npz)
- `fg_mask_path`: foreground mask

## Rendering comparison

The `render_flame_video.py` script in VHAP only visualizes the **tracked FLAME mesh** (a coarse parametric model with optional painted texture). This is useful for verifying tracking quality but is not photorealistic.

GaussianAvatars goes further by training ~600K iterations to optimize Gaussian positions, colors, scales, rotations, and opacities, achieving photorealistic quality with real-time rendering.
