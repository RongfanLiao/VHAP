# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
# CLI inference script for LAM using the lightweight motion format.
# Usage:
#   python inference_lightweight.py --image assets/sample_input/messi.png \
#       --motion /path/to/lightweight_motion_dir --output output/result.mp4

import os
import sys
import argparse
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch

from vgen.inference import build_model, parse_configs, save_images2video, add_audio_to_video
from vgen.runners.infer.head_utils import preprocess_image
from vgen.runners.infer.vhap_motion import VhapMotionLoader


def run_inference(avatar_dir, output_path, flametracking, lam, cfg):
    """Run the full inference pipeline using a lightweight motion directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # save raw input
        # image_raw = os.path.join(tmpdir, "raw.png")
        # with Image.open(image_path).convert('RGB') as img:
        #     img.save(image_raw)

        # # flame tracking on input image
        # print("Running flame tracking...")
        # return_code = flametracking.preprocess(image_raw)
        # assert return_code == 0, "flametracking preprocess failed!"
        # return_code = flametracking.optimize()
        # assert return_code == 0, "flametracking optimize failed!"
        # return_code, output_dir = flametracking.export()
        # assert return_code == 0, "flametracking export failed!"

        # tracked_image_path = os.path.join(output_dir, "images/00000_00.png")
        # mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")
        # print(f"Tracked image: {tracked_image_path}")
        # print(f"Mask: {mask_path}")

        # aspect_standard = 1.0
        # source_size = cfg.source_size
        # render_size = cfg.render_size
        # render_fps = 30

        # prepare reference image
        image_path = os.path.join(avatar_dir, "foreground_image.png")
        # mask_path = os.path.join(avatar_dir, "000000_mask.jpg")
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
        save_images2video(rgb, tmp_video, fps=30)

        # add audio if available
        avatar_basename = os.path.basename(avatar_dir.rstrip('/'))
        audio_path = os.path.join(avatar_dir, f"{avatar_basename}.wav")
        if os.path.exists(audio_path):
            add_audio_to_video(tmp_video, output_path, audio_path)
        else:
            import shutil
            shutil.copy2(tmp_video, output_path)
            print(f"No audio found at {audio_path}, saved video without audio.")

    print(f"Done! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LAM CLI Inference (Lightweight Motion Format)")
    # parser.add_argument("--image", type=str, required=True, help="Path to input face image")
    parser.add_argument("-a","--avatar", type=str, required=True,
                        help="Path to avatar directory (containing flame_param.npz)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: output/videos/<avatar>.mp4)")
    parser.add_argument("--model_name", type=str,
                        default="./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
                        help="Path to model checkpoint directory")
    parser.add_argument("--infer_config", type=str,
                        default="./configs/inference/lam-20k-8gpu.yaml",
                        help="Inference config yaml")
    args = parser.parse_args()

    if not os.path.isdir(args.avatar):
        print(f"Error: avatar directory not found: {args.avatar}")
        sys.exit(1)

    # set default output path
    if args.output is None:
        avatar_name = os.path.basename(args.avatar.rstrip('/'))
        args.output = os.path.join("output", "videos", f"{avatar_name}.mp4")

    # set env vars for parse_configs compatibility
    os.environ.update({
        'APP_ENABLED': '1',
        'APP_MODEL_NAME': args.model_name,
        'APP_INFER': args.infer_config,
        'APP_TYPE': 'infer.lam',
        'NUMBA_THREADING_LAYER': 'omp',
    })

    # override sys.argv so parse_configs and FlameTrackingSingleImage
    # don't choke on our CLI args
    original_argv = sys.argv
    sys.argv = [sys.argv[0], f"model_name={args.model_name}"]

    cfg, _ = parse_configs()

    # clear sys.argv to avoid conflicts with FlameTrackingSingleImage._parse_args
    sys.argv = [original_argv[0]]

    lam = build_model(cfg)
    lam.to('cuda')
    lam.eval()

    # flametracking = FlameTrackingSingleImage(
    #     output_dir='output/tracking',
    #     alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
    #     vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
    #     human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
    #     facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
    #     detect_iris_landmarks=False,
    # )
    run_inference(args.avatar, args.output, None, lam, cfg)


if __name__ == '__main__':
    main()
