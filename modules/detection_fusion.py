"""HOG + SVM classico e associazione geometrica con statistiche robuste."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple
    score: float
    label: str = "Pedone"
    points_in_box: int = 0
    points_used: int = 0
    depth_m: float | None = None
    range_m: float | None = None
    iqr_m: float | None = None

    def to_dict(self):
        return asdict(self)


class TraditionalDetector:
    """OpenCV fornisce un HOG preaddestrato per persone, non per automobili.

    Nessun addestramento nel progetto; SVM e Haar restano metodi di ML classico.
    Per auto si può fornire esplicitamente una cascade XML esterna.
    """
    def __init__(self, backend="hog", cascade_path=None, label=None):
        self.backend = backend
        self.label = label or ("Pedone" if backend == "hog" else "Oggetto")
        if backend == "hog":
            self.detector = cv2.HOGDescriptor()
            self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        elif backend == "haar":
            if not cascade_path:
                raise ValueError("Haar richiede --cascade con un file XML addestrato")
            self.detector = cv2.CascadeClassifier(str(cascade_path))
            if self.detector.empty():
                raise ValueError("Cascade XML non valida o non leggibile")
        else:
            raise ValueError("Detector ammessi: hog, haar")

    def detect(self, image):
        if self.backend == "hog":
            if image.shape[0] < 128 or image.shape[1] < 64:
                return []
            boxes, weights = self.detector.detectMultiScale(
                image, winStride=(8, 8), padding=(8, 8), scale=1.05)
            scores = np.asarray(weights).reshape(-1).tolist()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            boxes = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            scores = [1.0] * len(boxes)
        # NMS comune ai due backend; score HOG non è una probabilità.
        if len(boxes) == 0:
            return []
        # OpenCV richiede score_threshold >= 0. Traslare i margini SVM
        # conserva l'ordinamento, includendo eventuali margini nulli/negativi.
        offset = min(0.0, min(scores))
        nms_scores = [score-offset+1e-6 for score in scores]
        keep = cv2.dnn.NMSBoxes([list(map(int, b)) for b in boxes], nms_scores, 0.0, 0.4)
        return [Detection(tuple(map(int, boxes[i])), float(scores[i]), self.label)
                for i in np.asarray(keep).reshape(-1)]


def fuse_detections(detections, projection, min_points=3, iqr_factor=1.5):
    if min_points < 1 or iqr_factor < 0:
        raise ValueError("min_points >= 1 e iqr_factor >= 0 richiesti")
    for detection in detections:
        detection.depth_m = detection.range_m = detection.iqr_m = None
        x, y, w, h = detection.box
        u, v = projection.uv.T
        selected = (u >= x) & (u < x+w) & (v >= y) & (v < y+h)
        depths = projection.depth[selected]
        xyz = projection.camera_xyz[selected]
        detection.points_in_box = len(depths)
        detection.points_used = 0
        if len(depths) < min_points:
            continue
        q1, q3 = np.percentile(depths, [25, 75])
        spread = q3 - q1
        keep = (depths >= q1-iqr_factor*spread) & (depths <= q3+iqr_factor*spread)
        detection.points_used = int(keep.sum())
        if keep.sum() >= min_points:
            detection.depth_m = float(np.median(depths[keep]))
            # Distanza euclidea separata dalla profondità lungo Z.
            detection.range_m = float(np.median(np.linalg.norm(xyz[keep], axis=1)))
            detection.iqr_m = float(spread)
    return detections
