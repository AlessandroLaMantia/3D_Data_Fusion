"""Analizza i frame locali, salva un esempio reale e una galleria offline."""
import html
import json
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
from modules.calibration import load_calibration
from modules.projection import load_velodyne, project_points
from modules.detection_fusion import TraditionalDetector, fuse_detections
from modules.visualization import draw_overlay
from download_demo import DATE, DRIVE


def main():
    root = Path(__file__).resolve().parent
    sequence = root/'data'/DATE/DRIVE
    calibration = load_calibration(sequence.parent/'calib_cam_to_cam.txt', sequence.parent/'calib_velo_to_cam.txt')
    detector = TraditionalDetector()
    output = root/'outputs'/'demo'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((sequence/'image_02'/'data').glob('*.png')):
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        points = load_velodyne(sequence/'velodyne_points'/'data'/f'{path.stem}.bin')
        projection = project_points(points, calibration, image.shape)
        detections = fuse_detections(detector.detect(image), projection)
        overlay = draw_overlay(image, projection, detections)
        for kind, img in [('rgb', image), ('overlay', overlay)]:
            ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError('Codifica JPEG fallita')
            encoded.tofile(output/f'{path.stem}_{kind}.jpg')
        valid = sum(d.depth_m is not None for d in detections)
        record = {'frame': int(path.stem), 'projected_points': len(projection.depth),
                  'detections': [d.to_dict() for d in detections], 'valid_distances': valid}
        records.append(record)
        print(f'Frame {path.stem}: {len(detections)} detection, {valid} stime, {len(projection.depth)} punti proiettati', flush=True)
    if not records:
        raise RuntimeError('Nessun frame locale: eseguire download_demo.py')
    # Una stima supportata da piu punti favorisce un esempio leggibile.
    best = max(records, key=lambda r: (r['valid_distances'], max((d['points_used'] for d in r['detections'] if d['depth_m'] is not None), default=0)))
    subprocess.run([sys.executable, str(root/'main.py'), '--sequence', str(sequence),
                    '--frame', str(best['frame']), '--output', str(output/'best')], check=True)
    summary = {'best_frame': best['frame'], 'frames_analyzed': len(records),
               'frames_with_detections': sum(bool(r['detections']) for r in records),
               'frames_with_distances': sum(r['valid_distances'] > 0 for r in records),
               'records': records}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    cards = []
    for record in sorted(records, key=lambda r: (r['frame'] != best['frame'], r['frame'])):
        stem = f"{record['frame']:010d}"
        details = ', '.join(f"Z {d['depth_m']:.1f} m ({d['points_used']}/{d['points_in_box']} punti)" if d['depth_m'] is not None else 'Z non disponibile' for d in record['detections'])
        cards.append(f'<section><h2>Frame {stem}{" · esempio selezionato" if record["frame"] == best["frame"] else ""}</h2>'
                     f'<p>{len(record["detections"])} detection HOG · {record["projected_points"]} punti proiettati. {html.escape(details)}</p>'
                     f'<div><figure><img loading="lazy" src="{stem}_rgb.jpg" alt="Immagine Camera 2 del frame {stem}"><figcaption>RGB originale</figcaption></figure>'
                     f'<figure><img loading="lazy" src="{stem}_overlay.jpg" alt="Overlay LiDAR e detection del frame {stem}"><figcaption>Proiezione LiDAR + box + profondità</figcaption></figure></div></section>')
    page = '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Demo reale KITTI</title><style>body{font:17px/1.6 system-ui;background:#0b1424;color:#edf3ff;margin:30px auto;padding:0 20px;max-width:1500px}h1,h2,a{color:#58ddca}section{border-top:1px solid #40506a;padding:20px 0}section div{display:grid;grid-template-columns:1fr 1fr;gap:15px}figure{margin:0}img{width:100%;height:auto}code{background:#203047;padding:4px}@media(max-width:850px){section div{grid-template-columns:1fr}}</style><h1>Camera RGB + LiDAR: demo KITTI reale</h1>'
    page += f'<p>Sequenza {DRIVE}. {len(records)} frame analizzati; {summary["frames_with_detections"]} con detection, {summary["frames_with_distances"]} con stime metriche.</p>'
    page += '<p>Le detection provengono dal detector HOG, senza box manuali. Z è profondità nella Camera 2. I punti sono colorati per profondità TURBO (0–80 m); il PLY usa i colori RGB della foto. Una detection può essere un falso positivo e un box può includere sfondo.</p>'
    page += f'<p><a href="best/{best["frame"]:010d}_rgb.ply">Scarica la nuvola RGB PLY dell’esempio</a> · <a href="summary.json">Risultati di tutti i frame</a></p>'
    page += ''.join(cards)+'</html>'
    (output/'index.html').write_text(page, encoding='utf-8')
    print(f'Galleria: {output / "index.html"}\nFrame consigliato: {best["frame"]}')


if __name__ == '__main__':
    main()
