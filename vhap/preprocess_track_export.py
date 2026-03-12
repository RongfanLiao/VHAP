"""End-to-end pipeline: preprocess video, track FLAME, and export parameters.

Integrates the three data-extraction steps into a single script:

1. **Preprocess** — extract frames from a raw video file.
2. **Track** — run multi-stage FLAME tracking optimisation.
3. **Export** — save camera transforms and FLAME parameters as a lightweight dataset.

Usage::

    # Minimal (all defaults)
    python vhap/preprocess_track_export.py \\
        --input data/monocular/obama.mp4 \\
        --track-output-folder output/monocular/obama \\
        --export-output-folder export/monocular/obama

    # With foreground matting and specific epoch
    python vhap/preprocess_track_export.py \\
        --input data/monocular/obama.mp4 \\
        --track-output-folder output/monocular/obama \\
        --export-output-folder export/monocular/obama \\
        --matting-method robust_video_matting \\
        --epoch 20
"""

from pathlib import Path
from typing import Annotated, Literal, Optional

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
from vhap.export_as_nerf_dataset import check_epoch, load_config
from vhap.export_flame_params import FLAMEParamDatasetWriter
from vhap.model.tracker import GlobalTracker
from vhap.preprocess_video import (
    background_matting_v2,
    robust_video_matting,
    video2frames,
)


# ------------------------------------------------------------------
# Step builders
# ------------------------------------------------------------------


def _preprocess(
    input_path: Path,
    target_fps: int,
    matting_method: Optional[str],
) -> tuple:
    """Extract frames from video and optionally run foreground matting.

    Args:
        input_path: Path to the raw video file.
        target_fps: Target frame rate for extraction.
        matting_method: ``'robust_video_matting'``, ``'background_matting_v2'``,
            or ``None``.

    Returns:
        A tuple of ``(root_folder, sequence, image_dir)`` derived from the
        input path.
    """
    assert input_path.suffix in (".mov", ".mp4"), (
        f"Expected a video file (.mov/.mp4), got: {input_path}"
    )
    root_folder = input_path.parent
    sequence = input_path.stem
    image_dir = root_folder / sequence / "images"

    # Extract frames
    video2frames(input_path, image_dir, target_fps=target_fps)

    # Foreground matting
    if matting_method == "robust_video_matting":
        robust_video_matting(image_dir)
    elif matting_method == "background_matting_v2":
        background_matting_v2(image_dir)

    return root_folder, sequence, image_dir


def _build_tracking_config(
    root_folder: Path,
    sequence: str,
    output_folder: Path,
    device: str,
    batch_size: int,
) -> BaseTrackingConfig:
    """Construct a :class:`BaseTrackingConfig` with sensible defaults.

    Args:
        root_folder: Parent directory containing the ``{sequence}/images/`` folder.
        sequence: Name of the sequence (video stem).
        output_folder: Where to write tracking outputs.
        device: ``'cuda'`` or ``'cpu'``.
        batch_size: Number of frames per batch.

    Returns:
        A fully-initialised tracking configuration.
    """
    return BaseTrackingConfig(
        data=DataConfig(root_folder=root_folder, sequence=sequence),
        model=ModelConfig(),
        render=RenderConfig(),
        log=LogConfig(),
        exp=ExperimentConfig(output_folder=output_folder),
        lr=LearningRateConfig(),
        w=LossWeightConfig(),
        pipeline=PipelineConfig(
            lmk_init_rigid=StageLmkInitRigidConfig(),
            lmk_init_all=StageLmkInitAllConfig(),
            lmk_sequential_tracking=StageLmkSequentialTrackingConfig(),
            lmk_global_tracking=StageLmkGlobalTrackingConfig(),
            rgb_init_texture=StageRgbInitTextureConfig(),
            rgb_init_all=StageRgbInitAllConfig(),
            rgb_init_offset=StageRgbInitOffsetConfig(),
            rgb_sequential_tracking=StageRgbSequentialTrackingConfig(),
            rgb_global_tracking=StageRgbGlobalTrackingConfig(),
        ),
        device=device,
        batch_size=batch_size,
    )


def _export(
    cfg: BaseTrackingConfig,
    src_folder: Path,
    tgt_folder: Path,
    epoch: int,
) -> None:
    """Export tracked FLAME parameters using the tracking config directly.

    Args:
        cfg: The tracking configuration (provides data and model configs).
        src_folder: Tracking output folder containing ``tracked_flame_params*.npz``.
        tgt_folder: Target folder for the exported dataset.
        epoch: Which epoch to export (``-1`` for latest).
    """
    check_epoch(src_folder, epoch)

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


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main(
    # --- Input ---
    input: Annotated[Path, arg(aliases=["-i"])],
    # --- Output ---
    track_output_folder: Annotated[Path, arg(aliases=["-t"])],
    export_output_folder: Annotated[Path, arg(aliases=["-e"])],
    # --- Preprocess ---
    target_fps: int = 25,
    matting_method: Optional[
        Literal["robust_video_matting", "background_matting_v2"]
    ] = None,
    # --- Track ---
    device: Literal["cuda", "cpu"] = "cuda",
    batch_size: int = 16,
    # --- Export ---
    epoch: int = -1,
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
    """
    # ------------------------------------------------------------------
    # Step 1: Preprocess — extract frames from video
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1/3: Preprocessing video")
    print("=" * 60)
    root_folder, sequence, _ = _preprocess(
        input_path=input,
        target_fps=target_fps,
        matting_method=matting_method,
    )

    # ------------------------------------------------------------------
    # Step 2: Track — FLAME optimisation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2/3: FLAME tracking")
    print("=" * 60)
    cfg = _build_tracking_config(
        root_folder=root_folder,
        sequence=sequence,
        output_folder=track_output_folder,
        device=device,
        batch_size=batch_size,
    )
    tracker = GlobalTracker(cfg)
    tracker.optimize()

    # The tracker writes outputs into a timestamped subfolder.
    # load_config() resolves the actual subfolder containing config.yml.
    src_folder, cfg_saved = load_config(track_output_folder)

    # ------------------------------------------------------------------
    # Step 3: Export — lightweight FLAME parameter dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3/3: Exporting FLAME parameters")
    print("=" * 60)
    _export(
        cfg=cfg_saved,
        src_folder=src_folder,
        tgt_folder=export_output_folder,
        epoch=epoch,
    )

    print("\nPipeline complete!")


if __name__ == "__main__":
    tyro.cli(main)
