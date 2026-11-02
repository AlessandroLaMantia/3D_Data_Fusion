"""Coordinate omogenee: p ~ P_rect_02 R_rect_00 T_velo_cam [X,Y,Z,1]."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def read_calibration(path):
    """Legge i campi numerici KITTI, ignorando metadati come calib_time."""
    fields = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            fields[key] = np.array([float(x) for x in value.split()])
        except ValueError:
            continue
    return fields


@dataclass(frozen=True)
class Calibration:
    projection: np.ndarray       # 3x4: Velodyne -> pixel omogeneo
    rectified_transform: np.ndarray  # 4x4: Velodyne -> camera 0 rettificata
    camera2_transform: np.ndarray    # 4x4: Velodyne -> camera 2 rettificata


def load_calibration(cam_path, velo_path):
    cam, velo = read_calibration(cam_path), read_calibration(velo_path)
    def field(data, key, shape):
        if key not in data or data[key].size != np.prod(shape):
            raise ValueError(f"Calibrazione: campo {key} assente o dimensione errata")
        result = data[key].reshape(shape)
        if not np.isfinite(result).all():
            raise ValueError(f"Calibrazione non finita: {key}")
        return result
    p = field(cam, "P_rect_02", (3, 4))
    rect = np.eye(4)
    rect[:3, :3] = field(cam, "R_rect_00", (3, 3))
    transform = np.eye(4)
    transform[:3, :3] = field(velo, "R", (3, 3))
    transform[:3, 3] = field(velo, "T", (3,))
    rectified = rect @ transform
    # P = K [I | t]: include anche la traslazione della Camera 2.
    k = p[:, :3]
    if not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("P_rect_02 non è una proiezione KITTI rettificata")
    cam2_offset = np.eye(4)
    cam2_offset[:3, 3] = np.linalg.solve(k, p[:, 3])
    return Calibration(p @ rectified, rectified, cam2_offset @ rectified)
