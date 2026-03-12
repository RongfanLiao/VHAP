"""Render exported FLAME parameters into a video.

Loads the lightweight exported dataset and renders the FLAME mesh for each
frame, then encodes the result as an MP4 video.

Required files in ``src-folder``::

    src-folder/
    ├── transforms.json       # per-frame camera intrinsics & extrinsics
    └── flame_param.npz       # FLAME params (shape, expr, rotation, ...)
                              # optionally contains tex_extra & lights

If ``tex_extra`` and ``lights`` are present in ``flame_param.npz``, the mesh
is rendered with the tracked texture and SH lighting. Otherwise, a flat white
mesh is rendered.

Usage::

    python vhap/render_flame_video.py \\
        --src-folder export/monocular/obama \\
        --output export/monocular/obama/render.mp4

    # Custom FPS and background
    python vhap/render_flame_video.py \\
        --src-folder export/monocular/obama \\
        --output render.mp4 \\
        --fps 30 \\
        --background-color black
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from tqdm import tqdm

from vhap.model.flame import FlameHead, FlameTexPainted
from vhap.util.render_nvdiffrast import NVDiffRenderer


def _load_dataset(src_folder: Path) -> tuple:
    """Load transforms and FLAME parameters.

    Args:
        src_folder: Path containing ``transforms.json`` and ``flame_param.npz``.

    Returns:
        A tuple of ``(transforms_db, flame_params)``.
    """
    transforms_path = src_folder / "transforms.json"
    assert transforms_path.exists(), f"Not found: {transforms_path}"
    db = json.load(open(transforms_path, "r"))

    flame_param_path = src_folder / "flame_param.npz"
    assert flame_param_path.exists(), f"Not found: {flame_param_path}"
    flame_params = dict(np.load(flame_param_path))

    return db, flame_params


def _build_intrinsic(frame: Dict) -> torch.Tensor:
    """Build a 3x3 intrinsic matrix from a frame entry."""
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = frame["fl_x"]
    K[1, 1] = frame["fl_y"]
    K[0, 2] = frame["cx"]
    K[1, 2] = frame["cy"]
    return K


def _build_albedo(
    flame_params: Dict[str, np.ndarray],
    tex_resolution: int = 2048,
) -> Optional[torch.Tensor]:
    """Build the albedo texture map from FLAME parameters.

    Composes ``painted_base + tex_extra`` (residual mode) when ``tex_extra``
    is present in ``flame_params``.

    Args:
        flame_params: Dict of FLAME parameters, optionally containing ``tex_extra``.
        tex_resolution: Resolution to match the texture maps.

    Returns:
        Albedo tensor of shape ``(1, 3, H, W)`` on CUDA, or ``None``.
    """
    if "tex_extra" not in flame_params:
        return None

    # Base painted texture from FLAME assets
    tex_painted_model = FlameTexPainted(tex_size=tex_resolution).cuda()
    base_tex = tex_painted_model()  # (1, 3, H, W)

    # Residual texture from tracking
    tex_extra = torch.tensor(
        flame_params["tex_extra"], dtype=torch.float32
    ).cuda()[None]  # (1, 3, H, W)

    # Match resolution if needed
    if base_tex.shape[-1] != tex_extra.shape[-1]:
        base_tex = F.interpolate(base_tex, tex_extra.shape[-2:], mode="bilinear")

    albedo = base_tex + tex_extra  # residual composition
    return albedo


@torch.no_grad()
def render_frames(
    src_folder: Path,
    background_color: List[float],
    n_shape: int = 300,
    n_expr: int = 100,
) -> tuple:
    """Render all frames from the exported FLAME dataset.

    Args:
        src_folder: Path to the exported dataset folder.
        background_color: RGB background in ``[0, 1]`` range.
        n_shape: Number of FLAME shape basis components.
        n_expr: Number of FLAME expression basis components.

    Returns:
        A tuple of ``(frames_list, h, w)`` where ``frames_list`` is a list of
        uint8 numpy arrays of shape ``(H, W, 3)``.
    """
    db, flame_params = _load_dataset(src_folder)

    # Texture and lighting (before converting to tensors)
    albedo = _build_albedo(flame_params)

    lights = None
    if "lights" in flame_params:
        lights = torch.tensor(
            flame_params["lights"], dtype=torch.float32
        ).cuda()[None]  # (1, 9, 3)

    # Convert to tensors
    for k in flame_params:
        if isinstance(flame_params[k], np.ndarray):
            flame_params[k] = torch.tensor(flame_params[k], dtype=torch.float32)

    flame_model = FlameHead(n_shape, n_expr, add_teeth=True).cuda()
    renderer = NVDiffRenderer(use_opengl=False)

    faces = flame_model.faces
    shape = flame_params["shape"]  # (n_shape,)

    static_offset = flame_params.get("static_offset", None)
    if static_offset is not None:
        static_offset = static_offset.cuda()

    if albedo is not None:
        verts_uv = flame_model.verts_uvs.clone()
        verts_uv[:, 1] = 1 - verts_uv[:, 1]  # flip V for nvdiffrast
        faces_uv = flame_model.textures_idx.int()
    else:
        verts_uv = None
        faces_uv = None

    n_frames = len(db["frames"])
    rendered_frames: List[np.ndarray] = []
    h, w = 0, 0

    print(f"Rendering {n_frames} frames...")
    for i, frame in enumerate(tqdm(db["frames"], total=n_frames)):
        ti = frame["timestep_index"]

        # FLAME forward pass
        verts = flame_model(
            shape[None].cuda(),
            flame_params["expr"][[ti]].cuda(),
            flame_params["rotation"][[ti]].cuda(),
            flame_params["neck_pose"][[ti]].cuda(),
            flame_params["jaw_pose"][[ti]].cuda(),
            flame_params["eyes_pose"][[ti]].cuda(),
            flame_params["translation"][[ti]].cuda(),
            return_landmarks=False,
            static_offset=static_offset,
            dynamic_offset=(
                flame_params["dynamic_offset"][[ti]].cuda()
                if "dynamic_offset" in flame_params
                else None
            ),
        )

        # Camera: c2w → w2c
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        w2c = c2w.inverse().cuda()[None]  # (1, 4, 4)

        K = _build_intrinsic(frame).cuda()[None]  # (1, 3, 3)
        h = frame["h"]
        w = frame["w"]

        out = renderer.render_rgba_vis(
            verts,
            faces,
            w2c,
            K,
            (h, w),
            background_color=background_color,
            verts_uv=verts_uv,
            faces_uv=faces_uv,
            tex=albedo,
            lights=lights,
        )

        rgba = out["rgba"].squeeze(0)  # (H, W, 4)
        rgb = rgba[..., :3]  # (H, W, 3)
        rgb = (rgb * 255).clamp(0, 255).byte().cpu().numpy()
        rendered_frames.append(rgb)

    return rendered_frames, h, w


def main(
    src_folder: Path,
    output: Path,
    fps: int = 25,
    background_color: Literal["white", "black"] = "white",
) -> None:
    """Render exported FLAME parameters into a video.

    Args:
        src_folder: Folder containing ``transforms.json`` and ``flame_param.npz``.
        output: Output video file path (e.g. ``render.mp4``).
        fps: Frame rate of the output video.
        background_color: Background color for rendering.
    """
    bg = [1.0, 1.0, 1.0] if background_color == "white" else [0.0, 0.0, 0.0]

    frames, h, w = render_frames(src_folder, background_color=bg)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing video ({len(frames)} frames, {h}x{w}, {fps} fps) to: {output}")

    # Write frames to a temp dir, then encode with ffmpeg
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(Path(tmp_dir) / f"{i:06d}.png")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(Path(tmp_dir) / "%06d.png"),
                "-c:v", "libx264",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
    print("Done!")


if __name__ == "__main__":
    tyro.cli(main)
