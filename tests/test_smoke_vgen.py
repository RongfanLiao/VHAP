"""Smoke test for the vgen (toos/flame_param_to_video.py) inference pipeline.

Runs preprocess_image -> VhapMotionLoader -> LAM inference on the
avatar directory produced by the video_to_flame_param smoke test.

Tier A (stages 5-6): Data loading only, CPU, no heavy model.
Tier B (stages 7-8): Full LAM inference, requires CUDA + checkpoint.

Usage::

    # Run both tracking + vgen tests in order:
    pytest tests/ -v -s

    # Run only vgen tests (requires avatar dir from prior tracking run):
    pytest tests/test_smoke_vgen.py -v -s
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.smoke

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAM_CHECKPOINT_DIR = (
    PROJECT_ROOT / "model_zoo" / "lam_models" / "releases" / "lam" / "lam-20k" / "step_045500"
)
LAM_CONFIG = PROJECT_ROOT / "configs" / "inference" / "lam-20k-8gpu.yaml"


# ---------------------------------------------------------------
# Module-scoped state shared across ordered tests
# ---------------------------------------------------------------

class _VgenState:
    """Mutable container to pass data between ordered vgen tests."""
    avatar_dir: Path = None
    image: torch.Tensor = None        # [1, 3, H, W]
    shape_param = None
    motion_seq: dict = None
    output_video: Path = None


_st = _VgenState()


def _resolve_avatar_dir(avatar_dir_state_file):
    """Read avatar dir from session state file, or skip."""
    if avatar_dir_state_file.exists():
        with open(avatar_dir_state_file) as f:
            data = json.load(f)
        avatar_dir = Path(data["avatar_dir"])
        if avatar_dir.exists():
            return avatar_dir
    pytest.skip(
        "Avatar directory not available. "
        "Run the full video_to_flame_param smoke test first: pytest tests/ -v -s"
    )


# ---------------------------------------------------------------
# Tests — must run in order (continuing from test_smoke.py order 1-4)
# ---------------------------------------------------------------

@pytest.mark.order(5)
def test_preprocess_image(avatar_dir_state_file):
    """Load and preprocess the foreground image from the avatar directory."""
    from vgen.runners.infer.head_utils import preprocess_image

    avatar_dir = _resolve_avatar_dir(avatar_dir_state_file)
    _st.avatar_dir = avatar_dir

    image_path = str(avatar_dir / "foreground_image.png")
    assert os.path.isfile(image_path), f"foreground_image.png not found: {image_path}"

    image, mask, intr, shape_param = preprocess_image(
        image_path, mask_path=None, intr=None, pad_ratio=0,
        bg_color=1., max_tgt_size=None, aspect_standard=1.0,
        enlarge_ratio=[1.0, 1.0], render_tgt_size=512, multiply=14,
        need_mask=False, get_shape_param=False,
    )

    assert image.dim() == 4, f"Expected 4D image tensor, got {image.dim()}D"
    assert image.shape[0] == 1, f"Expected batch=1, got {image.shape[0]}"
    assert image.shape[1] == 3, f"Expected 3 channels, got {image.shape[1]}"
    assert image.shape[2] % 14 == 0, f"Height {image.shape[2]} not divisible by 14"
    assert image.shape[3] % 14 == 0, f"Width {image.shape[3]} not divisible by 14"
    assert image.dtype == torch.float32

    _st.image = image
    _st.shape_param = shape_param


@pytest.mark.order(6)
def test_motion_loader():
    """Load motion sequence from the avatar directory using VhapMotionLoader."""
    from vgen.runners.infer.vhap_motion import VhapMotionLoader

    assert _st.avatar_dir is not None, "test_preprocess_image must run first"

    loader = VhapMotionLoader(str(_st.avatar_dir))
    motion_seq = loader.prepare(
        bg_color=1.,
        shape_param=_st.shape_param,
        vis_motion=False,
        render_image_res=512,
        test_sample=False,
    )

    # Validate structure
    for key in ("render_c2ws", "render_intrs", "render_bg_colors", "flame_params"):
        assert key in motion_seq, f"Missing key: {key}"

    c2ws = motion_seq["render_c2ws"]
    assert c2ws.dim() == 4, f"Expected [1, N, 4, 4], got shape {c2ws.shape}"
    assert c2ws.shape[0] == 1
    assert c2ws.shape[2] == 4 and c2ws.shape[3] == 4
    num_frames = c2ws.shape[1]
    assert num_frames > 0, "No frames in motion sequence"

    intrs = motion_seq["render_intrs"]
    assert intrs.shape == (1, num_frames, 4, 4)

    bg = motion_seq["render_bg_colors"]
    assert bg.shape == (1, num_frames, 3)

    fp = motion_seq["flame_params"]
    for key in ("betas", "expr", "rotation", "neck_pose", "jaw_pose",
                "eyes_pose", "translation"):
        assert key in fp, f"flame_params missing key: {key}"
    assert fp["expr"].shape[0] == 1
    assert fp["expr"].shape[1] == num_frames

    _st.motion_seq = motion_seq


@pytest.mark.order(7)
@pytest.mark.skipif(
    not (LAM_CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"LAM checkpoint not found at {LAM_CHECKPOINT_DIR}",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for LAM inference",
)
def test_lam_inference(tmp_path):
    """Run full LAM inference pipeline on the avatar directory."""
    assert _st.avatar_dir is not None, "test_preprocess_image must run first"
    assert _st.motion_seq is not None, "test_motion_loader must run first"

    from toos.flame_param_to_video import run_inference
    from vgen.inference import build_model, parse_configs

    # Save global state to restore later
    original_argv = sys.argv
    env_keys = ('APP_ENABLED', 'APP_MODEL_NAME', 'APP_INFER',
                'APP_TYPE', 'NUMBA_THREADING_LAYER')
    original_env = {k: os.environ.get(k) for k in env_keys}

    try:
        os.environ.update({
            'APP_ENABLED': '1',
            'APP_MODEL_NAME': str(LAM_CHECKPOINT_DIR),
            'APP_INFER': str(LAM_CONFIG),
            'APP_TYPE': 'infer.lam',
            'NUMBA_THREADING_LAYER': 'omp',
        })
        sys.argv = [sys.argv[0], f"model_name={LAM_CHECKPOINT_DIR}"]

        cfg, _ = parse_configs()
        sys.argv = [original_argv[0]]

        lam = build_model(cfg)
        lam.to("cuda")
        lam.eval()

        output_path = str(tmp_path / "smoke_output.mp4")
        run_inference(str(_st.avatar_dir), output_path, lam, cfg)

        _st.output_video = Path(output_path)
    finally:
        sys.argv = original_argv
        for k, v in original_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.order(8)
def test_output_video():
    """Validate the output video exists and has reasonable size."""
    if _st.output_video is None:
        pytest.skip("test_lam_inference was skipped or failed")

    assert _st.output_video.exists(), f"Output video not found: {_st.output_video}"
    size_bytes = _st.output_video.stat().st_size
    assert size_bytes > 1000, f"Output video suspiciously small: {size_bytes} bytes"
