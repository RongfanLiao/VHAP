"""End-to-end pipeline: convert a video into exported FLAME parameters.

Integrates the three data-extraction steps into a single script:

1. **Preprocess** — extract frames from a raw video file.
2. **Track** — run multi-stage FLAME tracking optimisation.
3. **Export** — save camera transforms and FLAME parameters as a lightweight dataset.

Usage::

    # Minimal (all defaults)
    # if not specify export_output_folder, 
    # it will be set to track_output_folder / "exported" by default.

    python tools/video_to_flame_param.py \\
        --input data/monocular/obama.mp4 \\
        --output-folder output/monocular/obama \\
    
    # alternatively, you can specify export_output_folder explicitly:
    python tools/video_to_flame_param.py \\
        --input data/monocular/obama.mp4 \\
        --output-folder output/monocular/obama \\
        --export-output-folder export/monocular/obama

    # With foreground matting and specific epoch
    python tools/video_to_flame_param.py \\
        --input data/monocular/obama.mp4 \\
        --output-folder output/monocular/obama \\
        --export-output-folder export/monocular/obama \\
        --matting-method robust_video_matting \\
        --epoch 20
"""

from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal, Optional

import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tyro
from tyro.conf import arg

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
from vhap.export_as_nerf_dataset import load_config
from vhap.util.log import get_logger

logger = get_logger(__name__, root=True)
from vhap.export_flame_params import FLAMEParamDatasetWriter
from vhap.model.tracker import GlobalTracker, detect_landmarks
from vhap.preprocess_video import (
    robust_video_matting,
    style_matte,
    video2frames,
)


# ------------------------------------------------------------------
# Step builders
# ------------------------------------------------------------------


def _build_tracking_config(
    root_folder: Path,
    sequence: str,
    output_folder: Path,
    device: str,
    batch_size: int,
    epoch: int = 30,
    debug: bool = False,
) -> BaseTrackingConfig:
    """Construct a :class:`BaseTrackingConfig` with sensible defaults.

    Args:
        root_folder: Parent directory containing the ``{sequence}/images/`` folder.
        sequence: Name of the sequence (video stem).
        output_folder: Where to write tracking outputs.
        device: ``'cuda'`` or ``'cpu'``.
        batch_size: Number of frames per batch.
        debug: If True, use minimal steps/epochs for quick testing.

    Returns:
        A fully-initialised tracking configuration.
    """
    if debug:
        init_steps = 10
        seq_steps = 5
        n_epochs = 2
    else:
        init_steps = 500
        seq_steps = 50
        n_epochs = epoch

    return BaseTrackingConfig(
        data=DataConfig(root_folder=root_folder, sequence=sequence, background_color="white"),
        model=ModelConfig(),
        render=RenderConfig(),
        log=LogConfig(),
        exp=ExperimentConfig(output_folder=output_folder),
        lr=LearningRateConfig(),
        w=LossWeightConfig(),
        pipeline=PipelineConfig(
            lmk_init_rigid=StageLmkInitRigidConfig(num_steps=init_steps),
            lmk_init_all=StageLmkInitAllConfig(num_steps=init_steps),
            lmk_sequential_tracking=StageLmkSequentialTrackingConfig(num_steps=seq_steps),
            lmk_global_tracking=StageLmkGlobalTrackingConfig(num_epochs=n_epochs),
            rgb_init_texture=StageRgbInitTextureConfig(num_steps=init_steps),
            rgb_init_all=StageRgbInitAllConfig(num_steps=init_steps),
            rgb_init_offset=StageRgbInitOffsetConfig(num_steps=init_steps),
            rgb_sequential_tracking=StageRgbSequentialTrackingConfig(num_steps=seq_steps),
            rgb_global_tracking=StageRgbGlobalTrackingConfig(num_epochs=n_epochs),
        ),
        device=device,
        batch_size=batch_size,
    )


def _export(
    cfg: BaseTrackingConfig,
    src_folder: Path,
    tgt_folder: Path,
    epoch: int = -1,
) -> None:
    """Export tracked FLAME parameters using the tracking config directly.

    Args:
        cfg: The tracking configuration (provides data and model configs).
        src_folder: Tracking output folder containing ``tracked_flame_params*.npz``.
        tgt_folder: Target folder for the exported dataset.
        epoch: Which epoch to export (``-1`` for latest).
    """
    if epoch != -1:
        tgt_folder = Path(str(tgt_folder) + f"_epoch{epoch}")

    writer = FLAMEParamDatasetWriter(
        cfg_data=cfg.data,
        cfg_model=cfg.model,
        src_folder=src_folder,
        tgt_folder=tgt_folder,
        epoch=epoch,
    )
    writer.write()


def _format_elapsed(seconds: float) -> str:
    """Format elapsed wall-clock time for logging."""
    minutes, remaining_seconds = divmod(seconds, 60)
    remaining_seconds = int(remaining_seconds)
    hours, remaining_minutes = divmod(int(minutes), 60)
    if hours > 0:
        return f"{hours:d}h {remaining_minutes:02d}m {remaining_seconds:02d}s"
    if minutes >= 1:
        return f"{int(minutes):d}m {remaining_seconds:02d}s"
    return f"{seconds:.2f}s"


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main(
    # --- Input ---
    input: Annotated[Path, arg(aliases=["-i"])],
    # --- Output ---
    output_folder: Annotated[Path, arg(aliases=["-t"])] = None,
    export_output_folder: Annotated[Path, arg(aliases=["-e"])] = None,
    # --- Preprocess ---
    target_fps: int = 25,
    matting_method: Optional[
        Literal["robust_video_matting", "style_matte", "background_matting_v2"]
    ] = "style_matte",
    # --- Track ---
    device: Literal["cuda", "cpu"] = "cuda",
    batch_size: int = 64,
    # --- Export ---
    epoch: int = 30,
    # --- Cleanup ---
    cleanup: bool = True,
    # --- Debug ---
    debug: bool = False,
) -> None:
    """Run the full preprocess -> track -> export pipeline.

    Args:
        input: Path to the raw video file (e.g. ``data/monocular/obama.mp4``).
        track_output_folder: Directory for tracking outputs.
        export_output_folder: Directory for the exported lightweight dataset.
        target_fps: Target frame rate for frame extraction.
        matting_method: Optional foreground matting method.
        device: Device for FLAME tracking (``cuda`` or ``cpu``).
        batch_size: Batch size for the tracking optimiser.
        epoch: Which tracking epoch to export (``-1`` for latest).
        cleanup: Remove intermediate alpha_maps, images, and landmark2d dirs after export.
    """
    pipeline_start = perf_counter()

    # ------------------------------------------------------------------
    # Input validation — fail fast before expensive GPU work
    # ------------------------------------------------------------------
    if not input.exists():
        raise FileNotFoundError(f"Input video not found: {input}")
    if input.suffix not in (".mov", ".mp4"):
        raise ValueError(f"Expected a video file (.mov/.mp4), got: {input}")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH (required for frame extraction)")

    if debug:
        logger.warning(
            "Running in debug mode: using minimal steps/epochs for quick testing.")
    # ------------------------------------------------------------------
    # Step 1: Preprocess — extract frames from video
    # ------------------------------------------------------------------
    logger.info("Step 1/3: Preprocessing video")
    preprocess_start = perf_counter()
    root_folder, sequence = input.parent, input.stem
    image_dir = root_folder / sequence / "images"
    if output_folder is None:
        output_folder = Path("output") / root_folder / f"{sequence}"
    if export_output_folder is None:
        export_output_folder = Path("export") / root_folder / f"{sequence}"

    cfg = _build_tracking_config(
        root_folder=root_folder,
        sequence=sequence,
        output_folder=output_folder,
        device=device,
        batch_size=batch_size,
        epoch=epoch,
        debug=debug,
    )

    video2frames(input, image_dir, target_fps=target_fps)

    if matting_method == "robust_video_matting":
        robust_video_matting(image_dir)
    if matting_method == "style_matte":
        style_matte(image_dir)
    if matting_method == "background_matting_v2":
        raise NotImplementedError(
            "BackgroundMattingV2 integration is not implemented yet."
        )

    detect_landmarks(cfg)
    logger.info(
        "Step 1/3 complete in %s",
        _format_elapsed(perf_counter() - preprocess_start),
    )

    # ------------------------------------------------------------------
    # Step 2: Track — FLAME optimisation
    # ------------------------------------------------------------------
    logger.info("Step 2/3: FLAME tracking")
    tracking_start = perf_counter()
      
    tracker = GlobalTracker(cfg)
    tracker.optimize()
    logger.info(
        "Step 2/3 complete in %s",
        _format_elapsed(perf_counter() - tracking_start),
    )

    # The tracker writes outputs into a timestamped subfolder.
    # load_config() resolves the actual subfolder containing config.yml.
    src_folder, cfg_saved = load_config(output_folder)

    # ------------------------------------------------------------------
    # Step 3: Export — lightweight FLAME parameter dataset
    # ------------------------------------------------------------------
    logger.info("Step 3/3: Exporting FLAME parameters")
    export_start = perf_counter()
    _export(
        cfg=cfg_saved,
        src_folder=src_folder,
        tgt_folder=export_output_folder,
    )
    logger.info(
        "Step 3/3 complete in %s",
        _format_elapsed(perf_counter() - export_start),
    )

    # ------------------------------------------------------------------
    # Optional cleanup — remove intermediate directories
    # ------------------------------------------------------------------
    if cleanup:
        cleanup_start = perf_counter()
        seq_dir = root_folder / sequence
        if seq_dir.exists():
            shutil.rmtree(seq_dir)
            logger.info(f"Removed intermediate data {seq_dir}")
        if output_folder.exists():
            shutil.rmtree(output_folder)
            logger.info(f"Removed intermediate tracking output {output_folder}")
        logger.info(
            "Cleanup complete in %s",
            _format_elapsed(perf_counter() - cleanup_start),
        )

    logger.info(
        "Pipeline complete in %s",
        _format_elapsed(perf_counter() - pipeline_start),
    )


if __name__ == "__main__":
    tyro.cli(main)
