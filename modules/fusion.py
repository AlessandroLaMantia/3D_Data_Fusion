"""Colorizzazione RGB nello spazio originale Velodyne e salvataggio PLY."""
from pathlib import Path
import numpy as np
from .projection import visible_pixel_mask


def colorize_points(points, image_bgr, projection, z_buffer=True):
    keep = visible_pixel_mask(projection) if z_buffer else np.ones(len(projection.depth), bool)
    uv = np.floor(projection.uv[keep]).astype(int)
    colors = image_bgr[uv[:, 1], uv[:, 0], ::-1].copy()  # BGR -> RGB
    xyz = np.asarray(points)[projection.indices[keep], :3].copy()
    return xyz, colors


def save_ply(path, xyz, rgb):
    """PLY ASCII senza dipendenza da Open3D: metri e colori uint8."""
    if xyz.shape != rgb.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz e rgb devono avere forma (N,3)")
    with Path(path).open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(stream, np.column_stack((xyz, rgb)), fmt="%.6f %.6f %.6f %d %d %d")
