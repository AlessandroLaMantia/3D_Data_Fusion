"""Overlay 2D e visualizzatore Open3D opzionale."""
import cv2
import numpy as np


def draw_overlay(image, projection, detections, max_depth=80.0, language='it'):
    output = image.copy()
    lidar_layer = image.copy()
    values = np.round(255*np.clip(projection.depth/max_depth, 0, 1)).astype(np.uint8)
    colors = cv2.applyColorMap(values.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3) if len(values) else []
    # Lontani prima: quelli vicini rimangono visibili in sovrapposizione.
    for i in np.argsort(projection.depth)[::-1]:
        pixel = tuple(np.floor(projection.uv[i]).astype(int))
        cv2.circle(lidar_layer, pixel, 1, tuple(map(int, colors[i])), -1)
    # Trasparenza per mantenere leggibili persone e dettagli RGB.
    output = cv2.addWeighted(image, 0.45, lidar_layer, 0.55, 0)
    for det in detections:
        x, y, w, h = det.box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        depth = ("Z N/A" if language == 'en' else "Z n/d") if det.depth_m is None else f"Z {det.depth_m:.1f} m"
        points_label = 'points' if language == 'en' else 'punti'
        text = f"{det.label}: {depth} | {points_label} {det.points_used}/{det.points_in_box}"
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        origin = (max(0, min(x, output.shape[1]-text_width-4)), max(18, y-8))
        cv2.putText(output, text, origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, text, origin,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"LiDAR: TURBO, Z 0..{max_depth:g} m (clipping)", (10, output.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def show_cloud(xyz, rgb):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Installa requirements-open3d.txt per --show-3d") from exc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(rgb.astype(float)/255.0)
    o3d.visualization.draw_geometries([cloud], window_name="KITTI RGB - coordinate Velodyne")
