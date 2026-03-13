"""Export tracked FLAME parameters as a lightweight dataset (no images/masks).

Outputs camera transforms and a single all-frame FLAME parameter file
(including canonical parameters, appearance, and per-frame dynamics).

Output structure::

    tgt_folder/
    ├── transforms.json       # per-frame camera intrinsics & extrinsics
    └── flame_param.npz       # all data in one file

Usage::

    # Export using latest epoch (default)
    python vhap/export_flame_params.py \\
        --src_folder output/monocular/2920_right_lmkOnly \\
        --tgt_folder export/2920_right_flame

    # Export using a specific epoch
    python vhap/export_flame_params.py \\
        --src_folder output/monocular/2920_right_lmkOnly \\
        --tgt_folder export/2920_right_flame \\
        --epoch 20
"""

import math
import multiprocessing
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader
from tqdm import tqdm

from vhap.config.base import DataConfig, ModelConfig, import_module
from vhap.export_as_nerf_dataset import (
    check_epoch,
    load_config,
    write_data,
    write_json,
)


def _identity(x):
    return x


class FLAMEParamDatasetWriter:
    """Export camera transforms, canonical FLAME params, and a single all-frame FLAME param file.

    This writer combines camera parameter extraction and FLAME parameter export
    into a single lightweight pipeline. Unlike :class:`NeRFDatasetWriter` +
    :class:`TrackedFLAMEDatasetWriter`, it does **not** export images or masks.

    Args:
        cfg_data: Data configuration from the tracking run.
        cfg_model: Model configuration from the tracking run.
        src_folder: Path to the tracking output folder containing
            ``tracked_flame_params*.npz`` files.
        tgt_folder: Path to the target export folder.
        subset: Optional data subset filter.
        scale_factor: Optional scaling factor for the dataset.
        epoch: Which training epoch to export. ``-1`` (default) selects the latest.
    """

    def __init__(
        self,
        cfg_data: DataConfig,
        cfg_model: ModelConfig,
        src_folder: Path,
        tgt_folder: Path,
        subset: Optional[str] = None,
        scale_factor: Optional[float] = None,
        epoch: int = -1,
    ) -> None:
        self.tgt_folder = tgt_folder
        self.src_folder = src_folder

        # --- prepare dataloader for camera params ---
        print("==== Config: data ====")
        print(tyro.to_yaml(cfg_data))

        cfg_data.target_extrinsic_type = "c2w"
        cfg_data.background_color = None
        cfg_data.use_alpha_map = False
        dataset = import_module(cfg_data._target)(cfg=cfg_data, batchify_all_views=False)
        self.dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=None,
            collate_fn=_identity,
            num_workers=min(multiprocessing.cpu_count(), 8),
        )

        # --- load tracked FLAME params ---
        print("---- Config: model ----")
        print(tyro.to_yaml(cfg_model))

        paths = [Path(p) for p in glob(str(src_folder / "tracked_flame_params*.npz"))]
        epochs = [int(p.stem.split("_")[-1]) for p in paths]
        if epoch == -1:
            index = int(np.argmax(epochs))
        else:
            index = epochs.index(epoch)
        flame_params_path = paths[index]

        print(f"Loading FLAME parameters from: {flame_params_path}")
        self.flame_params: Dict[str, np.ndarray] = dict(np.load(flame_params_path))

        if "focal_length" in self.flame_params:
            self.focal_length: Optional[float] = self.flame_params["focal_length"].item()
        else:
            self.focal_length = None

        # Relocate FLAME to the origin
        self.M: np.ndarray = self._relocate_flame_meshes(self.flame_params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Run the full export pipeline.

        Creates ``transforms.json`` and ``flame_param.npz`` under
        :attr:`tgt_folder`.
        """
        if not self.tgt_folder.exists():
            self.tgt_folder.mkdir(parents=True)

        db = self._build_transforms()
        write_json(db, self.tgt_folder)

        self._write_flame_param(db)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relocate_flame_meshes(flame_param: Dict[str, np.ndarray]) -> np.ndarray:
        """Relocate FLAME to the origin and return the 4x4 transformation matrix.

        Args:
            flame_param: Mutable dict of FLAME parameters. The ``translation``
                entry is modified **in-place**.

        Returns:
            A ``(4, 4)`` homogeneous transformation matrix that centres the
            FLAME mesh sequence at the origin.
        """
        Ts = torch.tensor(flame_param["translation"])
        T_mean = Ts.mean(0)
        M = torch.eye(4)
        M[:3, 3] = -T_mean
        flame_param["translation"] = (M[:3, 3] + Ts).numpy()
        return M.numpy()

    def _build_transforms(self) -> Dict:
        """Build the ``transforms.json`` database with camera params only.

        Returns:
            A dict ready to be serialised as ``transforms.json``.
        """
        db: Dict = {"frames": []}
        timestep_indices: set = set()
        camera_indices: set = set()

        print(f"Building transforms for {self.tgt_folder}")
        for item in tqdm(self.dataloader, total=len(self.dataloader)):
            timestep_indices.add(item["timestep_index"])
            camera_indices.add(item["camera_index"])

            intrinsic = item["intrinsic"].double().numpy()
            h = item["rgb"].shape[0]
            w = item["rgb"].shape[1]

            if self.focal_length is not None:
                # Replace camera params with FLAME-estimated focal length
                fl_x = self.focal_length * max(h, w)
                fl_y = fl_x
                cx = w / 2
                cy = h / 2
                c2w = np.eye(4)
                c2w[2, 3] = 1
                transform_matrix = self.M @ c2w
            else:
                cx = intrinsic[0, 2]
                cy = intrinsic[1, 2]
                fl_x = intrinsic[0, 0]
                fl_y = intrinsic[1, 1]
                extrinsic = item["extrinsic"]
                transform_matrix = torch.cat(
                    [extrinsic, torch.tensor([[0, 0, 0, 1]])], dim=0
                ).numpy()
                transform_matrix = self.M @ transform_matrix

            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2

            frame_item = {
                "timestep_index": item["timestep_index"],
                "timestep_index_original": item["timestep_index_original"],
                "timestep_id": item["timestep_id"],
                "camera_index": item["camera_index"],
                "camera_id": item["camera_id"],
                "cx": cx,
                "cy": cy,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "h": h,
                "w": w,
                "camera_angle_x": angle_x,
                "camera_angle_y": angle_y,
                "transform_matrix": transform_matrix.tolist(),
            }
            db["frames"].append(frame_item)

        # Shared intrinsic params (for compatibility with other NeRF libraries)
        db.update(
            {
                "cx": cx,
                "cy": cy,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "h": h,
                "w": w,
                "camera_angle_x": angle_x,
                "camera_angle_y": angle_y,
            }
        )
        db["timestep_indices"] = sorted(timestep_indices)
        db["camera_indices"] = sorted(camera_indices)
        return db

    def _write_flame_param(self, db: Dict) -> None:
        """Write a single ``flame_param.npz`` containing all exported frames.

        Args:
            db: The transforms database used to determine which original
                timestep indices to include.
        """
        # Collect original timestep indices (deduplicated, insertion-ordered)
        seen: set = set()
        ti_orig_list: List[int] = []
        for frame in db["frames"]:
            ti_orig = frame["timestep_index_original"]
            if ti_orig not in seen:
                seen.add(ti_orig)
                ti_orig_list.append(ti_orig)

        params: Dict[str, np.ndarray] = {
            "translation": self.flame_params["translation"][ti_orig_list],
            "rotation": self.flame_params["rotation"][ti_orig_list],
            "neck_pose": self.flame_params["neck_pose"][ti_orig_list],
            "jaw_pose": self.flame_params["jaw_pose"][ti_orig_list],
            "eyes_pose": self.flame_params["eyes_pose"][ti_orig_list],
            "shape": self.flame_params["shape"],
            "expr": self.flame_params["expr"][ti_orig_list],
        }
        if "static_offset" in self.flame_params:
            params["static_offset"] = self.flame_params["static_offset"]
        if "dynamic_offset" in self.flame_params:
            params["dynamic_offset"] = self.flame_params["dynamic_offset"][ti_orig_list]
        if "tex_extra" in self.flame_params:
            params["tex_extra"] = self.flame_params["tex_extra"]
        if "lights" in self.flame_params:
            params["lights"] = self.flame_params["lights"]

        # Canonical (neutral-pose) parameters
        params["canonical_translation"] = np.zeros_like(self.flame_params["translation"][:1])
        params["canonical_rotation"] = np.zeros_like(self.flame_params["rotation"][:1])
        params["canonical_neck_pose"] = np.zeros_like(self.flame_params["neck_pose"][:1])
        params["canonical_jaw_pose"] = np.array([[0.3, 0, 0]])  # slightly open mouth
        params["canonical_eyes_pose"] = np.zeros_like(self.flame_params["eyes_pose"][:1])
        params["canonical_expr"] = np.zeros_like(self.flame_params["expr"][:1])

        path = self.tgt_folder / "flame_param.npz"
        print(f"Writing all-frame FLAME parameters ({len(ti_orig_list)} frames) to: {path}")
        write_data({path: params})


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------


def main(
    src_folder: Path,
    tgt_folder: Path,
    subset: Optional[str] = None,
    scale_factor: Optional[float] = None,
    epoch: int = -1,
) -> None:
    """Export tracked FLAME parameters as a lightweight dataset.

    Args:
        src_folder: Tracking output folder (contains ``config.yml`` and
            ``tracked_flame_params*.npz``).
        tgt_folder: Target folder for the exported dataset.
        subset: Optional data subset filter.
        scale_factor: Optional scaling factor.
        epoch: Training epoch to export (``-1`` for latest).
    """
    print(f"Begin lightweight exportation from {src_folder}")
    assert src_folder.exists(), f"Folder not found: {src_folder}"
    src_folder, cfg = load_config(src_folder)

    check_epoch(src_folder, epoch)

    if epoch != -1:
        tgt_folder = Path(str(tgt_folder) + f"_epoch{epoch}")

    writer = FLAMEParamDatasetWriter(
        cfg.data,
        cfg.model,
        src_folder,
        tgt_folder,
        subset=subset,
        scale_factor=scale_factor,
        epoch=epoch,
    )
    writer.write()

    print("Finished!")


if __name__ == "__main__":
    tyro.cli(main)
