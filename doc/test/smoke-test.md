# Smoke Test

End-to-end smoke test for the VHAP tracking pipeline. Validates the full
preprocess → track → export flow in debug mode on a short sample video.

## Prerequisites

- NVIDIA GPU with CUDA support (nvdiffrast has no CPU fallback)
- System `ffmpeg` binary installed
- Dev dependencies installed:

```bash
pip install -e ".[dev]"
```

## Run

```bash
pytest tests/test_smoke.py -v -s
```

Expected runtime: ~2–5 minutes in debug mode (minimal steps/epochs).

## Test Flow

| # | Test | Stage | What it does | Validates |
|---|------|-------|-------------|-----------|
| 1 | `test_preprocess` | Preprocess | Extracts frames via `video2frames()`, runs `style_matte()` | `images/` has `.jpg` frames; `alpha_maps/` has matching count |
| 2 | `test_detect_landmarks` | Preprocess | Runs `detect_landmarks(cfg)` | `landmark2d/` dir exists with `.npz` files |
| 3 | `test_track` | Track | Runs `GlobalTracker(cfg).optimize()` with debug config (init_steps=10, seq_steps=5, n_epochs=2) | `tracked_flame_params*.npz` and `config.yml` exist in output dir |
| 4 | `test_export` | Export | Runs `FLAMEParamDatasetWriter.write()` | `transforms.json` has `"frames"` key; `flame_param.npz` has `shape`, `expr`, `rotation`, `translation`; `foreground_image.png` exists |

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
└── export_output/
    ├── transforms.json       # per-frame camera intrinsics & extrinsics
    ├── flame_param.npz       # consolidated FLAME parameters
    └── foreground_image.png  # reference image (first frame)
```

## Key Files

- `tests/conftest.py` — shared fixtures (sample video path, temp working dir)
- `tests/test_smoke.py` — the 4 ordered test functions
- `pyproject.toml` — pytest config under `[tool.pytest.ini_options]`

## Notes

- Tests are ordered via `pytest-order` since each stage depends on the previous.
- Does **not** test the inference pipeline (`infer_vhap.py`), which requires LAM model weights.
- Debug config uses minimal optimization steps to keep the test fast while still exercising the full code path.
