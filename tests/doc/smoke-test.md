# Smoke Tests

End-to-end smoke tests for both the video_to_flame_param pipeline and the vgen
(tools/flame_param_to_video.py) inference pipeline.

## Prerequisites

- NVIDIA GPU with CUDA support (nvdiffrast has no CPU fallback)
- System `ffmpeg` binary installed
- Dev dependencies installed:

```bash
pip install -e ".[dev]"
```

- For vgen inference tests (stages 7-8): LAM model checkpoint at
  `model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors`

## Run

```bash
# Full suite (video_to_flame_param + vgen):
pytest tests/ -v -s

# video_to_flame_param pipeline only:
pytest tests/test_smoke.py -v -s

# vgen pipeline only (requires prior video_to_flame_param run):
pytest tests/test_smoke_vgen.py -v -s

# vgen data-loading tests only (no LAM model needed):
pytest tests/test_smoke_vgen.py -v -s -k "not lam_inference and not output_video"
```

Expected runtime: ~2–5 minutes for video_to_flame_param (stages 1-4), plus additional
time for LAM inference (stages 7-8) depending on GPU.

## Test Flow

### video_to_flame_param Pipeline (`test_smoke.py`)

| # | Test | Stage | What it does | Validates |
|---|------|-------|-------------|-----------|
| 1 | `test_preprocess` | Preprocess | Extracts frames via `video2frames()`, runs `style_matte()` | `images/` has `.jpg` frames; `alpha_maps/` has matching count |
| 2 | `test_detect_landmarks` | Preprocess | Runs `detect_landmarks(cfg)` | `landmark2d/` dir exists with `.npz` files |
| 3 | `test_track` | Track | Runs `GlobalTracker(cfg).optimize()` with debug config (init_steps=10, seq_steps=5, n_epochs=2) | `tracked_flame_params*.npz` and `config.yml` exist in output dir |
| 4 | `test_export` | Export | Runs `FLAMEParamDatasetWriter.write()` | `transforms.json` has `"frames"` key; `flame_param.npz` has `shape`, `expr`, `rotation`, `translation`; `foreground_image.png` exists |

### vgen Inference Pipeline (`test_smoke_vgen.py`)

| # | Test | Stage | What it does | Validates |
|---|------|-------|-------------|-----------|
| 5 | `test_preprocess_image` | Data loading | Runs `preprocess_image()` on avatar's `foreground_image.png` | Output tensor shape `[1, 3, H, W]`, dtype float32, H/W divisible by 14 |
| 6 | `test_motion_loader` | Data loading | Runs `VhapMotionLoader.prepare()` on avatar dir | `render_c2ws` shape `[1, N, 4, 4]`, `flame_params` dict has all expected keys |
| 7 | `test_lam_inference` | Inference | Builds LAM model, runs `run_inference()` end-to-end | Output video created (skipped if no checkpoint/CUDA) |
| 8 | `test_output_video` | Validation | Checks output video file | File exists with reasonable size |

## Test Input

- **Video**: `asset/watch_movie.mp4` (460 KB, 614x574, 60 fps)
- Extracted at `target_fps=30`

## Test Output Structure

All outputs go to a pytest-managed temp directory:

```
<tmp>/
├── watch_movie/
│   ├── images/           # extracted .jpg frames
│   ├── alpha_maps/       # foreground masks from style_matte
│   └── landmark2d/       # detected facial landmarks
├── track_output/
│   └── <timestamp>/
│       ├── config.yml
│       └── tracked_flame_params_2.npz
├── export_output/            # avatar dir consumed by vgen tests
│   ├── transforms.json       # per-frame camera intrinsics & extrinsics
│   ├── flame_param.npz       # consolidated FLAME parameters
│   └── foreground_image.png  # reference image (first frame)
└── smoke_output.mp4          # vgen inference output video
```

## Key Files

- `tests/conftest.py` — shared fixtures (sample video path, temp dirs, cross-module state)
- `tests/test_smoke.py` — video_to_flame_param pipeline tests (stages 1-4)
- `tests/test_smoke_vgen.py` — vgen inference tests (stages 5-8)
- `pyproject.toml` — pytest config under `[tool.pytest.ini_options]`

## Notes

- Tests are ordered via `pytest-order` since each stage depends on the previous.
- The vgen tests chain from the video_to_flame_param test's export output via a session-scoped state file.
- Stages 5-6 (data loading) run on CPU and don't require the LAM checkpoint.
- Stages 7-8 are automatically skipped if the LAM checkpoint or CUDA is unavailable.
- Debug config uses minimal optimization steps to keep the test fast while still exercising the full code path.
