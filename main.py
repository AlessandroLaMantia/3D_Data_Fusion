"""CLI per un frame KITTI Raw sincronizzato e rettificato."""
import argparse
import json
from pathlib import Path
import cv2
from modules.calibration import load_calibration
from modules.projection import load_velodyne, project_points
from modules.detection_fusion import TraditionalDetector, fuse_detections
from modules.fusion import colorize_points, save_ply
from modules.visualization import draw_overlay, show_cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True, help="Directory *_sync")
    parser.add_argument("--calib-dir", type=Path, help="Directory della data; default: parent della sequenza")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--detector", choices=["hog", "haar"], default="hog")
    parser.add_argument("--cascade", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--iqr-factor", type=float, default=1.5)
    parser.add_argument("--show-2d", action="store_true")
    parser.add_argument("--show-3d", action="store_true")
    args = parser.parse_args()
    if args.frame < 0:
        parser.error("--frame deve essere non negativo")
    stem = f"{args.frame:010d}"
    calib_dir = args.calib_dir or args.sequence.parent
    try:
        calibration = load_calibration(calib_dir/"calib_cam_to_cam.txt", calib_dir/"calib_velo_to_cam.txt")
        image_path = args.sequence/"image_02"/"data"/f"{stem}.png"
        # imdecode supporta percorsi Windows con caratteri Unicode.
        import numpy as np
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Immagine non leggibile: {image_path}")
        points = load_velodyne(args.sequence/"velodyne_points"/"data"/f"{stem}.bin")
        projection = project_points(points, calibration, image.shape)
        detector = TraditionalDetector(args.detector, args.cascade, args.label)
        detections = fuse_detections(detector.detect(image), projection, args.min_points, args.iqr_factor)
        xyz, rgb = colorize_points(points, image, projection)
        overlay = draw_overlay(image, projection, detections)
        args.output.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", overlay)
        if not ok:
            raise RuntimeError("Codifica PNG fallita")
        encoded.tofile(args.output/f"{stem}_overlay.png")
        save_ply(args.output/f"{stem}_rgb.ply", xyz, rgb)
        report = {"frame": stem, "sequence": str(args.sequence), "detector": args.detector,
                  "depth_definition": "Z Camera 2 rettificata, metri; non distanza euclidea",
                  "lidar_points": len(points), "projected_points": len(projection.depth),
                  "colored_points": len(xyz), "detections": [d.to_dict() for d in detections]}
        (args.output/f"{stem}_detections.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.show_2d:
            cv2.imshow("KITTI - geometria e profondita", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.show_3d:
            show_cloud(xyz, rgb)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Errore: {exc}\n")


if __name__ == "__main__":
    main()
