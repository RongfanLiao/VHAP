"""Shared fixtures for VHAP smoke tests."""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_VIDEO = PROJECT_ROOT / "asset" / "watch_movie.mp4"


@pytest.fixture(scope="module")
def sample_video():
    assert SAMPLE_VIDEO.exists(), f"Test video not found: {SAMPLE_VIDEO}"
    return SAMPLE_VIDEO


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory):
    """Shared working directory for the entire smoke test module.

    Layout after tests run::

        work_dir/
        ├── watch_movie/          # preprocessed frames & landmarks
        │   ├── images/
        │   ├── alpha_maps/
        │   └── landmark2d/
        ├── track_output/         # tracking results
        │   └── <timestamp>/
        └── export_output/        # exported flame params
    """
    return tmp_path_factory.mktemp("smoke")
