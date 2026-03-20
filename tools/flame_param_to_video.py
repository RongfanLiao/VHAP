"""CLI video generation script for FLAME-parameter-driven LAM rendering.

Usage::

    python tools/flame_param_to_video.py export/data/avatar_dir
    python tools/flame_param_to_video.py export/data/avatar_dir --output output/result.mp4
"""

import os
from collections import defaultdict
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
import torch
import tyro
from tyro.conf import Positional

from vgen.inference import build_model, parse_configs, save_images2video, add_audio_to_video
from vgen.runners.infer.head_utils import preprocess_image
from vgen.runners.infer.vhap_motion import VhapMotionLoader
from vgen.models.rendering.gaussian_model import GaussianModel
from vgen.models.rendering.gs_renderer import GS3DRenderer


CACHE_FILENAME = "lam_gaussians.pt"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed wall-clock time for logging."""
    minutes, secs = divmod(seconds, 60)
    secs = int(secs)
    hours, mins = divmod(int(minutes), 60)
    if hours > 0:
        return f"{hours:d}h {mins:02d}m {secs:02d}s"
    if minutes >= 1:
        return f"{int(minutes):d}m {secs:02d}s"
    return f"{seconds:.2f}s"
GS_KEYS = ("xyz", "opacity", "rotation", "scaling", "shs", "offset")


def build_renderer(cfg):
    """Build only the GS3DRenderer from a LAM checkpoint.

    Skips DinoV2 encoder and Transformer — much faster and lighter than
    ``build_model`` when cached Gaussians are available.
    """
    from safetensors.torch import load_file

    m = cfg.model
    latent_query_points_type = m.get("latent_query_points_type", "e2e_flame")
    skip_decoder = latent_query_points_type.startswith("e2e_flame")

    renderer = GS3DRenderer(
        human_model_path=m.get("human_model_path", "./model_zoo/human_parametric_models"),
        subdivide_num=m.get("flame_subdivide_num", 2),
        smpl_type=m.get("flame_type", "flame"),
        feat_dim=m.get("transformer_dim"),
        query_dim=m.get("gs_query_dim"),
        use_rgb=m.get("gs_use_rgb", False),
        sh_degree=m.get("gs_sh", 3),
        mlp_network_config=m.get("gs_mlp_network_config"),
        xyz_offset_max_step=m.get("gs_xyz_offset_max_step", 1.8 / 32),
        clip_scaling=m.get("gs_clip_scaling", 0.2),
        scale_sphere=m.get("scale_sphere", False),
        shape_param_dim=m.get("shape_param_dim", 100),
        expr_param_dim=m.get("expr_param_dim", 50),
        fix_opacity=m.get("fix_opacity", False),
        fix_rotation=m.get("fix_rotation", False),
        skip_decoder=skip_decoder,
        decode_with_extra_info=m.get("decode_with_extra_info"),
        gradient_checkpointing=m.get("tf_grad_ckpt", False),
        add_teeth=m.get("add_teeth", True),
        teeth_bs_flag=m.get("teeth_bs_flag", False),
        oral_mesh_flag=m.get("oral_mesh_flag", False),
        use_mesh_shading=m.get("use_mesh_shading", False),
        render_rgb=m.get("render_rgb", True),
    )

    # load only renderer.* weights from the checkpoint
    resume = os.path.join(cfg.model_name, "model.safetensors")
    print(f"Loading renderer weights from: {resume}")
    ckpt = load_file(resume, device="cpu")
    state_dict = renderer.state_dict()
    prefix = "renderer."
    for k, v in ckpt.items():
        if k.startswith(prefix):
            key = k[len(prefix):]
            if key in state_dict and state_dict[key].shape == v.shape:
                state_dict[key].copy_(v)
    renderer.to("cuda")
    renderer.eval()
    return renderer


def _save_gaussians(cache_path, gs_model_list, query_points):
    """Save the canonical Gaussians and query points to disk."""
    gs = gs_model_list[0]  # batch size is always 1
    data = {"query_points": query_points.cpu()}
    for key in GS_KEYS:
        data[f"gs_{key}"] = getattr(gs, key).cpu()
    torch.save(data, cache_path)


def _load_gaussians(cache_path, device):
    """Load cached Gaussians and query points from disk."""
    data = torch.load(cache_path, map_location=device, weights_only=True)
    gs = GaussianModel(**{key: data[f"gs_{key}"] for key in GS_KEYS})
    query_points = data["query_points"]
    return [gs], query_points


def _encode_image(avatar_dir, lam, cfg, flame_params, device):
    """Encode foreground_image.png into canonical 3D Gaussians.

    This is the expensive step (DinoV2 + Transformer decoding). The output
    depends only on the image and identity shape (betas), not on expression
    or pose, so it can be cached and reused with different flame motions.

    Returns:
        gs_model_list: List containing one GaussianModel.
        query_points: Canonical mesh positions tensor.
    """
    image_path = os.path.join(avatar_dir, "foreground_image.png")
    image, _, _, shape_param = preprocess_image(
        image_path, mask_path=None, intr=None, pad_ratio=0,
        bg_color=1., max_tgt_size=None, aspect_standard=1.0,
        enlarge_ratio=[1.0, 1.0], render_tgt_size=cfg.source_size, multiply=14,
        need_mask=False, get_shape_param=False,
    )

    image_tensor = image.unsqueeze(0).to(device, torch.float32)

    query_points = None
    if lam.latent_query_points_type.startswith("e2e_flame"):
        query_points, flame_params = lam.renderer.get_query_points(
            flame_params, device=device)

    latent_points, _ = lam.forward_latent_points(
        image_tensor[:, 0], camera=None, query_points=query_points)

    gs_model_list, query_points, _, _ = lam.renderer.forward_gs(
        gs_hidden_features=latent_points,
        query_points=query_points,
        flame_data=flame_params,
    )

    return gs_model_list, query_points


def _render_frames(renderer, gs_model_list, query_points, motion_seq, device):
    """Animate cached Gaussians with flame params and render all frames.

    Args:
        renderer: GS3DRenderer instance (either standalone or ``lam.renderer``).

    Returns:
        rgb: np.ndarray of shape (N, H, W, 3), dtype uint8.
    """
    render_c2ws = motion_seq["render_c2ws"].to(device)
    render_intrs = motion_seq["render_intrs"].to(device)
    render_bg_colors = motion_seq["render_bg_colors"].to(device)
    flame_params = {k: v.to(device) for k, v in motion_seq["flame_params"].items()}

    render_h = int(render_intrs[0, 0, 1, 2] * 2)
    render_w = int(render_intrs[0, 0, 0, 2] * 2)
    num_views = render_c2ws.shape[1]

    render_res_list = []
    for view_idx in range(num_views):
        render_res = renderer.forward_animate_gs(
            gs_model_list,
            query_points,
            renderer.get_single_view_smpl_data(flame_params, view_idx),
            render_c2ws[:, view_idx:view_idx + 1],
            render_intrs[:, view_idx:view_idx + 1],
            render_h, render_w,
            render_bg_colors[:, view_idx:view_idx + 1],
        )
        render_res_list.append(render_res)

    # aggregate results (same logic as infer_single_view)
    out = defaultdict(list)
    for res in render_res_list:
        for k, v in res.items():
            out[k].append(v)
    for k, v in out.items():
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.concat(v, dim=1)
            if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                out[k] = out[k][0].permute(0, 2, 3, 1)

    # compose foreground over white background
    rgb = out["comp_rgb"].detach().cpu().numpy()
    mask = out["comp_mask"].detach().cpu().numpy()
    mask[mask < 0.5] = 0.0
    rgb = rgb * mask + (1 - mask) * 1
    return (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)


def _save_video(avatar_dir, output_path, rgb):
    """Encode RGB frames to an MP4 video, attaching audio if available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_video = os.path.join(tmpdir, "output_noaudio.mp4")
        save_images2video(rgb, tmp_video, fps=25)

        avatar_basename = os.path.basename(avatar_dir.rstrip('/'))
        audio_path = os.path.join(avatar_dir, f"{avatar_basename}.wav")
        if os.path.exists(audio_path):
            add_audio_to_video(tmp_video, output_path, audio_path)
        else:
            shutil.copy2(tmp_video, output_path)
            print(f"No audio found at {audio_path}, saved video without audio.")

    print(f"Done! Output: {output_path}")


def run_inference(avatar_dir, output_path, renderer, cfg, lam=None):
    """Run inference with Gaussian caching.

    On the first run, encodes the foreground image into canonical 3D Gaussians
    and saves them as ``lam_gaussians.pt`` in the avatar directory. On
    subsequent runs, loads the cache and skips image encoding. The rendering
    step (animate Gaussians with flame params) always runs, so different
    flame motions produce different videos without re-encoding the image.

    Args:
        renderer: GS3DRenderer for animation + rasterization (always needed).
        lam: Full ModelLAM (only needed when no cache exists).
    """
    total_start = perf_counter()
    device = torch.device("cuda")
    cache_path = os.path.join(avatar_dir, CACHE_FILENAME)

    # prepare motion sequence (always needed for rendering)
    t0 = perf_counter()
    image_path = os.path.join(avatar_dir, "foreground_image.png")
    image, _, _, shape_param = preprocess_image(
        image_path, mask_path=None, intr=None, pad_ratio=0,
        bg_color=1., max_tgt_size=None, aspect_standard=1.0,
        enlarge_ratio=[1.0, 1.0], render_tgt_size=cfg.source_size, multiply=14,
        need_mask=False, get_shape_param=False,
    )
    vis_motion = cfg.get("vis_motion", False)
    loader = VhapMotionLoader(avatar_dir)
    motion_seq = loader.prepare(
        bg_color=1.,
        shape_param=shape_param,
        vis_motion=vis_motion,
        render_image_res=cfg.render_size,
        test_sample=False,
    )
    print(f"[Preprocess] {_format_elapsed(perf_counter() - t0)}")

    # encode image or load cache
    with torch.no_grad():
        t0 = perf_counter()
        if os.path.exists(cache_path):
            print(f"Cache found: {cache_path}, skipping image encoding.")
            gs_model_list, query_points = _load_gaussians(cache_path, device)
            print(f"[Load cache] {_format_elapsed(perf_counter() - t0)}")
        else:
            assert lam is not None, "No cache and no LAM model — cannot encode image."
            print("Encoding foreground image (first run)...")
            flame_params = {k: v.to(device) for k, v in motion_seq["flame_params"].items()}
            gs_model_list, query_points = _encode_image(
                avatar_dir, lam, cfg, flame_params, device)
            _save_gaussians(cache_path, gs_model_list, query_points)
            print(f"[Encode + cache] {_format_elapsed(perf_counter() - t0)}")

        # render with current flame params (always runs)
        t0 = perf_counter()
        print("Rendering with flame params...")
        rgb = _render_frames(renderer, gs_model_list, query_points, motion_seq, device)
        print(f"[Render] {_format_elapsed(perf_counter() - t0)}")

    if vis_motion:
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        vis_ref_img = np.tile(
            cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]),
                       interpolation=cv2.INTER_AREA)[None, :, :, :],
            (rgb.shape[0], 1, 1, 1),
        )
        rgb = np.concatenate([vis_ref_img, rgb, motion_seq["vis_motion_render"]], axis=2)

    t0 = perf_counter()
    _save_video(avatar_dir, output_path, rgb)
    print(f"[Save video] {_format_elapsed(perf_counter() - t0)}")
    print(f"[Total] {_format_elapsed(perf_counter() - total_start)}")


def main(
    avatar: Positional[str],
    output: Optional[str] = None,
    model_name: str = "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
    infer_config: str = "./configs/inference/lam-20k-8gpu.yaml",
) -> None:
    """LAM CLI Inference (Lightweight Motion Format).

    Args:
        avatar: Path to avatar directory (containing flame_param.npz).
        output: Output video path (default: output/videos/<avatar>.mp4).
        model_name: Path to model checkpoint directory.
        infer_config: Inference config yaml.
    """
    # --- input validation ---
    if not os.path.isdir(avatar):
        print(f"Error: avatar directory not found: {avatar}")
        sys.exit(1)
    for required in ("foreground_image.png", "flame_param.npz", "transforms.json"):
        if not os.path.exists(os.path.join(avatar, required)):
            print(f"Error: {required} not found in {avatar}")
            sys.exit(1)
    if not os.path.isdir(model_name):
        print(f"Error: model directory not found: {model_name}")
        sys.exit(1)
    if not os.path.isfile(infer_config):
        print(f"Error: config file not found: {infer_config}")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("Error: CUDA is not available (required for LAM inference)")
        sys.exit(1)

    # set default output path
    if output is None:
        avatar_name = os.path.basename(avatar.rstrip('/'))
        output = os.path.join("output", "videos", f"{avatar_name}.mp4")

    # set env vars for parse_configs compatibility
    os.environ.update({
        'APP_ENABLED': '1',
        'APP_MODEL_NAME': model_name,
        'APP_INFER': infer_config,
        'APP_TYPE': 'infer.lam',
        'NUMBA_THREADING_LAYER': 'omp',
    })

    # override sys.argv so parse_configs doesn't choke on our CLI args
    original_argv = sys.argv
    sys.argv = [sys.argv[0], f"model_name={model_name}"]
    cfg, _ = parse_configs()
    sys.argv = [original_argv[0]]

    cache_path = os.path.join(avatar, CACHE_FILENAME)
    has_cache = os.path.exists(cache_path)

    t0 = perf_counter()
    if has_cache:
        # cache exists — only need the renderer (FLAME + rasterizer)
        renderer = build_renderer(cfg)
        print(f"[Build renderer] {_format_elapsed(perf_counter() - t0)}")
        run_inference(avatar, output, renderer, cfg)
    else:
        # no cache — need full model for image encoding
        lam = build_model(cfg)
        lam.to('cuda')
        lam.eval()
        print(f"[Build model] {_format_elapsed(perf_counter() - t0)}")
        run_inference(avatar, output, lam.renderer, cfg, lam=lam)


if __name__ == '__main__':
    tyro.cli(main)
