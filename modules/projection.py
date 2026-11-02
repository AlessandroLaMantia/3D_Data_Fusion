"""Proiezione vettoriale; gli indici mantengono le corrispondenze 2D/3D."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .calibration import Calibration


@dataclass(frozen=True)
class Projection:
    uv: np.ndarray
    depth: np.ndarray
    indices: np.ndarray
    camera_xyz: np.ndarray


def load_velodyne(path):
    path = Path(path)
    if path.stat().st_size % 16:
        raise ValueError("File Velodyne non valido: attesi record float32 X,Y,Z,intensità")
    return np.fromfile(path, dtype="<f4").reshape(-1, 4)


def project_points(points, calibration: Calibration, image_shape):
    xyz = np.asarray(points)[:, :3]
    homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
    pixels = homogeneous @ calibration.projection.T
    camera = homogeneous @ calibration.camera2_transform.T
    d = pixels[:, 2]
    valid = np.isfinite(homogeneous).all(axis=1) & np.isfinite(pixels).all(axis=1)
    valid &= (d > 1e-6) & (camera[:, 2] > 1e-6)
    indices = np.flatnonzero(valid)
    # Divisione prospettica soltanto per i punti davanti alla camera.
    uv = pixels[indices, :2] / d[indices, None]
    h, w = image_shape[:2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    indices = indices[inside]
    return Projection(uv[inside], camera[indices, 2], indices, camera[indices, :3])


def visible_pixel_mask(projection):
    """Z-buffer discreto: mantiene il ritorno più vicino per ciascun pixel.

    Riduce sovrapposizioni LiDAR; non risolve occlusioni senza ritorni sul primo piano.
    """
    uv = np.floor(projection.uv).astype(np.int64)
    order = np.argsort(projection.depth, kind="stable")
    _, first = np.unique(uv[order], axis=0, return_index=True)
    mask = np.zeros(len(uv), dtype=bool)
    mask[order[first]] = True
    return mask
