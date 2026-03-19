# VHAP


VHAP is a head-tracking and avatar-data preparation project built around FLAME, differentiable rasterization, and photometric alignment. In this workspace, the repository supports two practical workflows in the same environment:

1. Turn a raw video into tracked FLAME parameters and an exportable lightweight dataset.
2. Use the exported avatar package to drive the integrated LAM inference stack and generate a video.

The result is a single repo that covers preprocessing, foreground matting, landmark detection, FLAME optimization, dataset export, and downstream talking-head generation.

## What This Project Does

- Tracks a head sequence from monocular video using FLAME and photometric optimization.
- Exports compact avatar-facing / NeRF-style assets such as `transforms.json`, `flame_param.npz`, masks, and a foreground reference image.
- Integrates a `vgen/` inference stack so exported VHAP assets can be rendered into a final video with `tools/flame_param_to_video.py`.
- Provides lower-level tracking utilities, NeRSemble entrypoints, and interactive FLAME viewers.

## Main Entry Points

| Path | Purpose |
| --- | --- |
| `tools/video_to_flame_param.py` | End-to-end monocular pipeline: extract frames, matte foreground, track FLAME, export lightweight assets. |
| `tools/flame_param_to_video.py` | Run the integrated LAM inference path on an exported avatar directory and write an output video. |
| `vhap/flame_editor.py` | Inspect FLAME masks and regions. |
| `vhap/flame_viewer.py` | Visualize tracked FLAME sequences. |

## Repository Layout

- `vhap/`: tracking, rendering, export, FLAME models, viewers, and utilities.
- `vgen/`: integrated LAM model and inference code.
- `configs/`: inference and experiment configuration files.
- `doc/`: workflow-specific notes for monocular and NeRSemble setups.
- `model_zoo/`: checkpoints and third-party assets expected by the pipeline.
- `asset/`: FLAME-related assets and media used by the project.

## Supported Workflows

### 1. Monocular Video to Exported Avatar Package

Given a raw video such as `data/0408_right.mp4`, the pipeline can:

1. Extract frames.
2. Generate foreground mattes.
3. Detect landmarks.
4. Optimize FLAME parameters.
5. Export a compact folder that downstream tools can consume.

### 2. Exported Avatar Package to Generated Video

Given an exported folder such as `export/data/0408_right/`, the LAM integration can:

1. Load the foreground image and motion data.
2. Build the LAM model from `vgen/`.
3. Render a video to `output/videos/<avatar>.mp4`.

### 3. Multi-view / NeRSemble Tracking

The repository also keeps the original VHAP tracking stack for NeRSemble and NeRSemble V2. Use the dedicated docs and entrypoints for those datasets.

## Installation

The project is intended for Linux with an NVIDIA GPU. Python 3.10 and CUDA 12.1 are the safest baseline for the current workspace state.

```shell
git clone git@github.com:ShenhanQian/VHAP.git
cd VHAP

conda create --name vhap -y python=3.10
conda activate vhap

# CUDA toolkit and native build tools used by nvdiffrast / PyTorch extensions.
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64" 2>/dev/null || true
conda env config vars set CUDA_HOME=$CONDA_PREFIX

# Install a CUDA-matching PyTorch build.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the project and its pinned Python dependencies.
pip install -e .
```

Notes:

- `nvdiffrast` is built from source on first use. The current renderer bootstrap infers the active env from `sys.executable` and fills `CUDA_HOME`, `CUDACXX`, and related search paths automatically, but the CUDA toolkit still needs to be installed inside the env.
- The integrated LAM path is version-sensitive. Keep the pinned runtime from `pyproject.toml`, especially `accelerate==1.13.0`, `diffusers==0.32.2`, and `transformers==4.41.2`. Newer versions may run without crashing but still change the rendered video.
- The same environment has been validated to run both `tools/video_to_flame_param.py` and `tools/flame_param_to_video.py`.

## Required Assets

### FLAME

Download FLAME assets from the [official FLAME website](https://flame.is.tue.mpg.de/download.php) and place them here:

- `asset/flame/flame2023.pkl`
- `asset/flame/FLAME_masks.pkl`

Optional:

- FLAME 2020 can be used via `asset/flame/generic_model.pkl`, but you will need to update the FLAME model path in the code accordingly.

### Model Weights

If you use the default commands and paths in this repo, keep these assets available:

- StyleMatte checkpoint under `model_zoo/matting/stylematte_synth.pt`
- LAM weights under `model_zoo/lam_models/releases/lam/lam-20k/step_045500/`

## Quick Start

### 1. Preprocess, Track, and Export a Monocular Video

For a real run, disable debug mode so the tracker uses the full optimization schedule:

```shell
python tools/video_to_flame_param.py \
  --input data/0408_right.mp4 \
  --no-debug \
  --epoch 30
```

By default this writes:

- Tracking outputs to `output/data/0408_right/<timestamp>/`
- Exported lightweight assets to `export/data/0408_right/`

Typical exported files include:

```text
export/data/0408_right/
  flame_param.npz
  foreground_image.png
  transforms.json
```

Typical tracking outputs include:

```text
output/data/0408_right/<timestamp>/
  config.yml
  tracked_flame_params_*.npz
  *.log
```

For a fast smoke test, omit `--no-debug`. The current CLI defaults to a short debug profile intended for validation rather than final quality.

Useful options:

- `--output-folder`: override where tracking outputs are written.
- `--export-output-folder`: override where exported assets are written.
- `--matting-method`: choose `style_matte` or `robust_video_matting`.
- `--batch-size`: control tracking batch size.

### 2. Generate a Video from Exported VHAP Assets

```shell
python tools/flame_param_to_video.py -a export/data/0408_right
```

By default this writes:

```text
output/videos/0408_right.mp4
```

The avatar directory is expected to contain the exported VHAP assets, including:

- `foreground_image.png`
- `flame_param.npz`
- `transforms.json`
- optionally `0408_right.wav` or another `<avatar_name>.wav` file for audio muxing

Useful options:

- `--output`: custom output video path.
- `--model_name`: custom LAM checkpoint directory.
- `--infer_config`: custom inference config YAML.

## Utilities

Inspect FLAME masks and custom regions:

```shell
python vhap/flame_editor.py
```

Visualize a tracked sequence:

```shell
python vhap/flame_viewer.py \
  --param_path output/data/0408_right/<timestamp>/tracked_flame_params_30.npz
```

Both viewers support flat shading via `--no-shade-smooth`. The viewer can also render with a texture image through `--tex_path`.

## Troubleshooting

### `nvdiffrast` or CUDA build errors

- Make sure the active env contains `cuda-toolkit`, `ninja`, and `cmake`.
- If the linker cannot find `-lcudart`, make sure `lib64` points to the env `lib` directory.
- If the extension cache is stale, reinstall and clear the torch extension cache:

```shell
pip install nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling --force-reinstall
rm -rf ~/.cache/torch_extensions/*/nvdiffrast*
```

### `tools/flame_param_to_video.py` runs but the video looks wrong

This usually means the LAM runtime drifted away from the pinned versions. Re-check the currently pinned dependencies in `pyproject.toml`, especially:

- `accelerate==1.13.0`
- `diffusers==0.32.2`
- `transformers==4.41.2`

### Missing FLAME assets

If tracking fails early, verify that the FLAME model and mask files are present under `asset/flame/`.

## License

This project is released under [CC-BY-NC-SA-4.0](LICENSE).

The repository is derived from the multi-view head tracker used in [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars/tree/main/reference_tracker), which carries the following restriction:

