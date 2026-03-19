"""VHAP motion sequence loader for consolidated FLAME param format.

Handles the VHAP export format where all per-frame FLAME parameters
are stored in a single ``flame_param.npz`` file with stacked arrays, instead of
individual per-frame ``.npz`` files.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch

from vgen.runners.infer.head_utils import render_flame_mesh, scale_intrs


class VhapMotionLoader:
    """Load motion sequences from the VHAP consolidated FLAME format.

    Expected directory layout::

        motion_dir/
            flame_param.npz        # stacked per-frame FLAME params
            transforms.json        # per-frame camera params
            canonical_flame_param.npz  # canonical/neutral reference

    ``flame_param.npz`` keys:
        - expr:        (N, 100)
        - rotation:    (N, 3)
        - neck_pose:   (N, 3)
        - jaw_pose:    (N, 3)
        - eyes_pose:   (N, 6)
        - translation: (N, 3)
        - shape:       (300,)
        - static_offset: (1, 5143, 3)  [optional, not consumed by LAM]

    Parameters
    ----------
    motion_dir : str
        Path to the VHAP dataset directory.
    """

    FLAME_DYNAMIC_KEYS = ("expr", "rotation", "neck_pose", "jaw_pose",
                          "eyes_pose", "translation")

    def __init__(self, motion_dir: str):
        self.motion_dir = motion_dir
        self._flame_npz_path = os.path.join(motion_dir, "flame_param.npz")
        self._transforms_path = os.path.join(motion_dir, "transforms.json")
        self._canonical_path = os.path.join(motion_dir, "canonical_flame_param.npz")

        if not os.path.isfile(self._flame_npz_path):
            raise FileNotFoundError(
                f"flame_param.npz not found in {motion_dir}")
        if not os.path.isfile(self._transforms_path):
            raise FileNotFoundError(
                f"transforms.json not found in {motion_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        bg_color: float = 1.0,
        shape_param: torch.Tensor = None,
        vis_motion: bool = False,
        render_image_res: int = 512,
        test_sample: bool = False,
    ) -> dict:
        """Build the motion sequence dict expected by ``ModelLAM.infer_single_view``.

        Returns
        -------
        dict with keys:
            render_c2ws       – [1, N, 4, 4]
            render_intrs      – [1, N, 4, 4]
            render_bg_colors  – [1, N, 3]
            flame_params      – dict of [1, N, ...] or [1, 300] tensors
            vis_motion_render – ndarray [N, H, W, 3] or None
        """
        # --- load flame params (stacked) ---
        flame_data = dict(np.load(self._flame_npz_path, allow_pickle=True))
        num_frames = flame_data["expr"].shape[0]

        # --- load transforms.json ---
        with open(self._transforms_path) as fp:
            tf_data = json.load(fp)
        all_frames = tf_data["frames"]
        all_frames = sorted(all_frames, key=lambda x: x["timestep_index"])
        print(f"len motion_seq: {len(all_frames)}")

        # --- optional subsampling ---
        frame_ids = np.arange(min(num_frames, len(all_frames)))
        if test_sample:
            sample_num = 50
            frame_ids = frame_ids[
                np.linspace(0, len(frame_ids) - 1, sample_num).astype(np.int32)
            ]
            print(f"sub sample {sample_num} frames for testing, ids: {frame_ids}")

        # --- build camera matrices ---
        c2ws, intrs = [], []
        for idx in frame_ids:
            frame_info = all_frames[idx]
            c2w, intrinsic = self._load_pose(frame_info)
            intrinsic = scale_intrs(intrinsic, 0.5, 0.5)
            c2ws.append(c2w)
            intrs.append(intrinsic)

        c2ws = torch.stack(c2ws, dim=0)   # [N, 4, 4]
        intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
        bg_colors = torch.full((len(frame_ids), 3), bg_color, dtype=torch.float32)

        # --- build FLAME param tensors ---
        flame_params = {}
        for key in self.FLAME_DYNAMIC_KEYS:
            arr = flame_data[key]  # (total_frames, D)
            selected = arr[frame_ids]  # (N, D)
            flame_params[key] = torch.FloatTensor(selected)

        # shape / betas
        if shape_param is not None:
            flame_params["betas"] = shape_param  # (300,)
        else:
            flame_params["betas"] = torch.FloatTensor(flame_data["shape"])  # (300,)

        # --- optional motion visualization ---
        if vis_motion:
            motion_render = render_flame_mesh(flame_params, intrs, c2ws)
        else:
            motion_render = None

        # --- add batch dimension ---
        for k, v in flame_params.items():
            flame_params[k] = v.unsqueeze(0)
        c2ws = c2ws.unsqueeze(0)
        intrs = intrs.unsqueeze(0)
        bg_colors = bg_colors.unsqueeze(0)

        return {
            "render_c2ws": c2ws,
            "render_intrs": intrs,
            "render_bg_colors": bg_colors,
            "flame_params": flame_params,
            "vis_motion_render": motion_render,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pose(frame_info: dict):
        """Extract c2w and intrinsic matrices from a transforms.json frame entry."""
        c2w = np.array(frame_info["transform_matrix"])
        c2w[:3, 1:3] *= -1  # NeRF convention → OpenCV convention
        c2w = torch.FloatTensor(c2w)

        intrinsic = torch.eye(4, dtype=torch.float32)
        intrinsic[0, 0] = frame_info["fl_x"]
        intrinsic[1, 1] = frame_info["fl_y"]
        intrinsic[0, 2] = frame_info["cx"]
        intrinsic[1, 2] = frame_info["cy"]

        return c2w, intrinsic
