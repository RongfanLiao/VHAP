# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
# CLI inference script for LAM (no Gradio required).
# Usage:
#   python inference.py --image assets/sample_input/messi.png --motion Anti_Drugs --output output/result.mp4
#   python inference.py --image assets/sample_input/messi.png --motion Anti_Drugs  # default output: output/videos/<image_name>_<motion_name>.mp4

import os
import sys
import argparse
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

from vgen.runners.infer.head_utils import prepare_motion_seqs, preprocess_image


def save_images2video(img_lst, v_pth, fps):
    from moviepy.editor import ImageSequenceClip
    images = [image.astype(np.uint8) for image in img_lst]
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(v_pth, codec='libx264')
    print(f"Video saved at {v_pth}")


def add_audio_to_video(video_path, out_path, audio_path):
    from moviepy.editor import VideoFileClip, AudioFileClip
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    video_clip_with_audio = video_clip.set_audio(audio_clip)
    video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac')
    print(f"Video with audio saved at {out_path}")


def build_model(cfg):
    from vgen.models import ModelLAM
    from safetensors.torch import load_file

    model = ModelLAM(**cfg.model)
    resume = os.path.join(cfg.model_name, "model.safetensors")
    # print("=" * 80)
    print("Loading pretrained weight from:", resume)
    if resume.endswith('safetensors'):
        ckpt = load_file(resume, device='cpu')
    else:
        ckpt = torch.load(resume, map_location='cpu')
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(
                    f"[WARN] mismatching shape for param {k}:" 
                    f" ckpt {v.shape} != model {state_dict[k].shape}, ignored."
                )
        else:
            print(f"[WARN] unexpected param {k}: {v.shape}")
    print("Finished loading pretrained weight.")
    # print("=" * 80)
    return model


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    parser.add_argument("--blender_path", type=str, default='blender')
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    cfg.blender_path = args.blender_path

    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")

    args.config = args.infer if args.config is None else args.config

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )
        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault("save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp"))
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault("video_dump", os.path.join("dumps", cli_cfg.model_name, "videos"))
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 30
    cfg.merge_with(cli_cfg)
    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"
    return cfg, cfg_train


def run_inference(image_path, motion_name, output_path, flametracking, lam, cfg):
    """Run the full inference pipeline for a single image + motion pair."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # save raw input
        image_raw = os.path.join(tmpdir, "raw.png")
        with Image.open(image_path).convert('RGB') as img:
            img.save(image_raw)

        # resolve motion params dir: accept either a name or a path
        if os.path.isdir(motion_name):
            # direct path provided
            motion_dir = motion_name
        else:
            motion_dir = os.path.join("./assets/sample_motion/export", motion_name)

        flame_params_dir = os.path.join(motion_dir, "flame_param")
        if not os.path.isdir(flame_params_dir):
            print(f"Error: motion flame_param dir not found: {flame_params_dir}")
            print(f"Available motions: {os.listdir('./assets/sample_motion/export/')}")
            sys.exit(1)

        # flame tracking on input image
        print("Running flame tracking...")
        return_code = flametracking.preprocess(image_raw)
        assert return_code == 0, "flametracking preprocess failed!"
        return_code = flametracking.optimize()
        assert return_code == 0, "flametracking optimize failed!"
        return_code, output_dir = flametracking.export()
        assert return_code == 0, "flametracking export failed!"

        tracked_image_path = os.path.join(output_dir, "images/00000_00.png")
        mask_path = os.path.join(output_dir, "fg_masks/00000_00.png")
        print(f"Tracked image: {tracked_image_path}")
        print(f"Mask: {mask_path}")

        aspect_standard = 1.0
        source_size = cfg.source_size
        render_size = cfg.render_size
        render_fps = 30

        # prepare reference image
        image, _, _, shape_param = preprocess_image(
            tracked_image_path, mask_path=mask_path, intr=None, pad_ratio=0,
            bg_color=1., max_tgt_size=None, aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0], render_tgt_size=source_size, multiply=14,
            need_mask=True, get_shape_param=True,
        )

        # prepare motion sequence
        src = tracked_image_path.split('/')[-3]
        driven = flame_params_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_img_need_mask = cfg.get("motion_img_need_mask", False)
        vis_motion = cfg.get("vis_motion", False)

        motion_seq = prepare_motion_seqs(
            flame_params_dir, None, save_root=tmpdir, fps=render_fps,
            bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
            render_image_res=render_size, multiply=16,
            need_mask=motion_img_need_mask, vis_motion=vis_motion,
            shape_param=shape_param, test_sample=False, cross_id=False,
            src_driven=src_driven,
        )
        # run model inference
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        torch.save(motion_seq, os.path.join("./", "motion_seq.pt"))
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
        save_images2video(rgb, tmp_video, render_fps)

        # add audio if available
        motion_basename = os.path.basename(motion_dir.rstrip('/'))
        audio_path = os.path.join(motion_dir, f"{motion_basename}.wav")
        if os.path.exists(audio_path):
            add_audio_to_video(tmp_video, output_path, audio_path)
        else:
            import shutil
            shutil.copy2(tmp_video, output_path)
            print(f"No audio found at {audio_path}, saved video without audio.")

    print(f"Done! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LAM CLI Inference")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input face image (e.g. assets/sample_input/messi.png)")
    parser.add_argument("--motion", type=str, required=True,
                        help="Motion name from assets/sample_motion/export/ (e.g. Anti_Drugs)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: output/videos/<image>_<motion>.mp4)")
    parser.add_argument("--model_name", type=str,
                        default="./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
                        help="Path to model checkpoint directory")
    parser.add_argument("--infer_config", type=str,
                        default="./configs/inference/lam-20k-8gpu.yaml",
                        help="Inference config yaml")
    args = parser.parse_args()

    # validate inputs
    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    # set default output path
    if args.output is None:
        img_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = os.path.join("output", "videos", f"{img_name}_{args.motion}.mp4")

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

    # print("Loading config...")
    cfg, _ = parse_configs()

    # clear sys.argv to avoid conflicts with FlameTrackingSingleImage._parse_args
    sys.argv = [original_argv[0]]

    # print("Building LAM model...")
    lam = build_model(cfg)
    lam.to('cuda')
    lam.eval()

    # print("Initializing flame tracking...")
    flametracking = FlameTrackingSingleImage(
        output_dir='output/tracking',
        alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
        vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
        human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
        facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
        detect_iris_landmarks=False,
    )

    run_inference(args.image, args.motion, args.output, flametracking, lam, cfg)


if __name__ == '__main__':
    main()
