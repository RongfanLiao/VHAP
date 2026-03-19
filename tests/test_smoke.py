"""End-to-end smoke test for the video_to_flame_param pipeline.

Runs the same preprocess -> track -> export stages as
video_to_flame_param.py in debug mode on a short sample video.
Requires a CUDA GPU.

Usage::

    pytest tests/test_smoke.py -v -s
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from vhap.config.base import (
    BaseTrackingConfig,
    DataConfig,
    ExperimentConfig,
    LearningRateConfig,
    LogConfig,
    LossWeightConfig,
    ModelConfig,
    PipelineConfig,
    RenderConfig,
    StageLmkGlobalTrackingConfig,
    StageLmkInitAllConfig,
    StageLmkInitRigidConfig,
    StageLmkSequentialTrackingConfig,
    StageRgbGlobalTrackingConfig,
    StageRgbInitAllConfig,
    StageRgbInitOffsetConfig,
    StageRgbInitTextureConfig,
    StageRgbSequentialTrackingConfig,
)

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------
# Module-scoped state shared across ordered tests
# ---------------------------------------------------------------

class _State:
    """Mutable container to pass data between ordered tests."""
    root_folder: Path = None
    sequence: str = None
    image_dir: Path = None
    output_folder: Path = None
    export_folder: Path = None
    cfg: BaseTrackingConfig = None
    src_folder: Path = None  # resolved tracking output with timestamp


_st = _State()


def _build_debug_config(root_folder, sequence, output_folder):
    return BaseTrackingConfig(
        data=DataConfig(root_folder=root_folder, sequence=sequence, background_color="white"),
        model=ModelConfig(),
        render=RenderConfig(),
        log=LogConfig(),
        exp=ExperimentConfig(output_folder=output_folder),
        lr=LearningRateConfig(),
        w=LossWeightConfig(),
        pipeline=PipelineConfig(
            lmk_init_rigid=StageLmkInitRigidConfig(num_steps=10),
            lmk_init_all=StageLmkInitAllConfig(num_steps=10),
            lmk_sequential_tracking=StageLmkSequentialTrackingConfig(num_steps=5),
            lmk_global_tracking=StageLmkGlobalTrackingConfig(num_epochs=2),
            rgb_init_texture=StageRgbInitTextureConfig(num_steps=10),
            rgb_init_all=StageRgbInitAllConfig(num_steps=10),
            rgb_init_offset=StageRgbInitOffsetConfig(num_steps=10),
            rgb_sequential_tracking=StageRgbSequentialTrackingConfig(num_steps=5),
            rgb_global_tracking=StageRgbGlobalTrackingConfig(num_epochs=2),
        ),
        device="cuda",
        batch_size=64,
    )


# ---------------------------------------------------------------
# Tests — must run in order
# ---------------------------------------------------------------

@pytest.mark.order(1)
def test_preprocess(sample_video, work_dir):
    """Extract frames and run matting on the sample video."""
    from vhap.preprocess_video import style_matte, video2frames

    # Copy video into work_dir so intermediate dirs land there
    video_copy = work_dir / sample_video.name
    shutil.copy2(sample_video, video_copy)

    root_folder = video_copy.parent
    sequence = video_copy.stem
    image_dir = root_folder / sequence / "images"

    video2frames(video_copy, image_dir, target_fps=30)

    images = sorted(image_dir.glob("*.jpg"))
    assert len(images) > 0, "No frames extracted"

    style_matte(image_dir)

    alpha_dir = root_folder / sequence / "alpha_maps"
    alphas = sorted(alpha_dir.glob("*.jpg"))
    assert len(alphas) == len(images), (
        f"Alpha map count ({len(alphas)}) != frame count ({len(images)})"
    )

    # Store for subsequent tests
    _st.root_folder = root_folder
    _st.sequence = sequence
    _st.image_dir = image_dir
    _st.output_folder = work_dir / "track_output"
    _st.export_folder = work_dir / "export_output"


@pytest.mark.order(2)
def test_detect_landmarks(work_dir):
    """Detect facial landmarks on extracted frames."""
    from vhap.model.tracker import detect_landmarks

    assert _st.root_folder is not None, "test_preprocess must run first"

    cfg = _build_debug_config(_st.root_folder, _st.sequence, _st.output_folder)
    _st.cfg = cfg

    detect_landmarks(cfg)

    landmark_dir = _st.root_folder / _st.sequence / "landmark2d"
    assert landmark_dir.exists(), f"Landmark dir not found: {landmark_dir}"
    landmark_files = list(landmark_dir.rglob("*.npz"))
    assert len(landmark_files) > 0, "No landmark files generated"


@pytest.mark.order(3)
def test_track(work_dir):
    """Run FLAME tracking in debug mode."""
    from vhap.model.tracker import GlobalTracker

    assert _st.cfg is not None, "test_detect_landmarks must run first"

    tracker = GlobalTracker(_st.cfg)
    tracker.optimize()

    # Resolve timestamped output subfolder
    subdirs = sorted(_st.output_folder.iterdir())
    assert len(subdirs) > 0, "No tracking output subfolder created"
    _st.src_folder = subdirs[-1]

    config_yml = _st.src_folder / "config.yml"
    assert config_yml.exists(), f"config.yml not found in {_st.src_folder}"

    npz_files = list(_st.src_folder.glob("tracked_flame_params*.npz"))
    assert len(npz_files) > 0, "No tracked_flame_params*.npz files found"


@pytest.mark.order(4)
def test_export(work_dir, avatar_dir_state_file):
    """Export FLAME parameters from tracking output."""
    from vhap.export_as_nerf_dataset import load_config
    from vhap.export_flame_params import FLAMEParamDatasetWriter

    assert _st.src_folder is not None, "test_track must run first"

    _, cfg_saved = load_config(_st.output_folder)

    writer = FLAMEParamDatasetWriter(
        cfg_data=cfg_saved.data,
        cfg_model=cfg_saved.model,
        src_folder=_st.src_folder,
        tgt_folder=_st.export_folder,
        epoch=-1,
    )
    writer.write()

    # Verify transforms.json
    transforms_path = _st.export_folder / "transforms.json"
    assert transforms_path.exists(), "transforms.json not created"
    with open(transforms_path) as f:
        transforms = json.load(f)
    assert "frames" in transforms, "transforms.json missing 'frames' key"
    assert len(transforms["frames"]) > 0, "transforms.json has empty frames"

    # Verify flame_param.npz
    flame_path = _st.export_folder / "flame_param.npz"
    assert flame_path.exists(), "flame_param.npz not created"
    data = dict(np.load(flame_path))
    for key in ("shape", "expr", "rotation", "translation"):
        assert key in data, f"flame_param.npz missing key: {key}"

    # Verify foreground image
    fg_image = _st.export_folder / "foreground_image.png"
    assert fg_image.exists(), "foreground_image.png not created"

    # Communicate export folder to downstream vgen tests
    with open(avatar_dir_state_file, "w") as f:
        json.dump({"avatar_dir": str(_st.export_folder)}, f)

    # Clean up intermediate data (mirrors production pipeline cleanup)
    seq_dir = _st.root_folder / _st.sequence
    if seq_dir.exists():
        shutil.rmtree(seq_dir)
    if _st.output_folder.exists():
        shutil.rmtree(_st.output_folder)
