"""CLI video generation script for FLAME-parameter-driven LAM rendering.

Usage::

    python tools/flame_param_to_video.py export/data/avatar_dir
    python tools/flame_param_to_video.py export/data/avatar_dir --output output/result.mp4
"""

import os
from pathlib import Path
import shutil
import sys
import tempfile
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


def run_inference(avatar_dir, output_path, lam, cfg):
    """Run the full inference pipeline using a lightweight motion directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # prepare reference image
        image_path = os.path.join(avatar_dir, "foreground_image.png")
        image, _, _, shape_param = preprocess_image(
            image_path, mask_path=None, intr=None, pad_ratio=0,
            bg_color=1., max_tgt_size=None, aspect_standard=1.0,
            enlarge_ratio=[1.0, 1.0], render_tgt_size=cfg.source_size, multiply=14,
            need_mask=False, get_shape_param=False,
        )

        # prepare motion sequence using VHAP loader
        vis_motion = cfg.get("vis_motion", False)
        loader = VhapMotionLoader(avatar_dir)
        motion_seq = loader.prepare(
            bg_color=1.,
            shape_param=shape_param,
            vis_motion=vis_motion,
            render_image_res=cfg.render_size,
            test_sample=False,
        )

        # run model inference
        # motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        # torch.save(motion_seq, os.path.join("./", "motion_seq.pt"))
        device, dtype = "cuda", torch.float32
        print("Running LAM inference...")
        with torch.no_grad():
            res = lam.infer_single_view(
                image.unsqueeze(0).to(device, dtype), None, None,
                render_c2ws=motion_seq["render_c2ws"].to(device),
                render_intrs=motion_seq["render_intrs"].to(device),
                render_bg_colors=motion_seq["render_bg_colors"].to(device),
                flame_params={k: v.to(device) for k, v in motion_seq["flame_params"].items()},
            )

        # compose output frames
        rgb = res["comp_rgb"].detach().cpu().numpy()
        mask = res["comp_mask"].detach().cpu().numpy()
        mask[mask < 0.5] = 0.0
        rgb = rgb * mask + (1 - mask) * 1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)

        if vis_motion:
            vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
            vis_ref_img = np.tile(
                cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]),
                           interpolation=cv2.INTER_AREA)[None, :, :, :],
                (rgb.shape[0], 1, 1, 1),
            )
            rgb = np.concatenate([vis_ref_img, rgb, motion_seq["vis_motion_render"]], axis=2)

        # save video
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_video = os.path.join(tmpdir, "output_noaudio.mp4")
        save_images2video(rgb, tmp_video, fps=25)

        # add audio if available
        avatar_basename = os.path.basename(avatar_dir.rstrip('/'))
        audio_path = os.path.join(avatar_dir, f"{avatar_basename}.wav")
        if os.path.exists(audio_path):
            add_audio_to_video(tmp_video, output_path, audio_path)
        else:
            shutil.copy2(tmp_video, output_path)
            print(f"No audio found at {audio_path}, saved video without audio.")

    print(f"Done! Output: {output_path}")


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

    lam = build_model(cfg)
    lam.to('cuda')
    lam.eval()

    run_inference(avatar, output, lam, cfg)


if __name__ == '__main__':
    tyro.cli(main)
