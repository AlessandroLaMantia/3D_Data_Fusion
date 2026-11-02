"""Verifiche numeriche su scene sintetiche; nessun download del dataset."""
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
import numpy as np
import cv2
from modules.calibration import load_calibration, Calibration
from modules.projection import project_points, Projection, load_velodyne
from modules.detection_fusion import Detection, TraditionalDetector, fuse_detections
from modules.fusion import colorize_points, save_ply
from modules.visualization import draw_overlay


class GeometryTests(unittest.TestCase):
    def test_calibration_camera2_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'cam.txt').write_text('calib_time: yesterday\nR_rect_00: 1 0 0 0 1 0 0 0 1\nP_rect_02: 100 0 20 -50 0 100 10 0 0 0 1 0\n')
            (root/'velo.txt').write_text('R: 1 0 0 0 1 0 0 0 1\nT: 1 0 0\n')
            calib = load_calibration(root/'cam.txt', root/'velo.txt')
            projected = project_points(np.array([[0, 0, 10, .9]]), calib, (40, 80))
            np.testing.assert_allclose(projected.uv, [[25, 10]])
            np.testing.assert_allclose(projected.camera_xyz, [[.5, 0, 10]])

    def test_filter_and_original_indices(self):
        p = np.array([[10, 0, 5, 0], [0, 10, 5, 0], [0, 0, 1, 0]], float)
        calib = Calibration(p, np.eye(4), np.eye(4))
        points = np.array([[0, 0, -1], [0, 0, 2], [20, 0, 1], [np.nan, 0, 1], [0, 0, 0]])
        out = project_points(points, calib, (10, 10))
        np.testing.assert_array_equal(out.indices, [1])
        np.testing.assert_allclose(out.uv, [[5, 5]])

    def test_iqr_and_euclidean_distance(self):
        depths = np.array([10, 10, 11, 11, 100], float)
        xyz = np.column_stack((np.full(5, 3), np.zeros(5), depths))
        proj = Projection(np.ones((5, 2)), depths, np.arange(5), xyz)
        result = fuse_detections([Detection((0, 0, 3, 3), 1)], proj)[0]
        self.assertEqual((result.points_in_box, result.points_used), (5, 4))
        self.assertEqual(result.depth_m, 10.5)
        self.assertGreater(result.range_m, result.depth_m)
        empty = fuse_detections([Detection((10, 10, 2, 2), 1)], proj)[0]
        self.assertIsNone(empty.depth_m)

    def test_rgb_and_occlusion(self):
        points = np.array([[0, 0, 9], [0, 0, 2]], float)
        proj = Projection(np.array([[0.1, 0.1], [.8, .8]]), np.array([9., 2.]), np.array([0, 1]), points)
        image = np.array([[[10, 20, 30]]], np.uint8)
        xyz, rgb = colorize_points(points, image, proj)
        np.testing.assert_array_equal(xyz, [points[1]])
        np.testing.assert_array_equal(rgb, [[30, 20, 10]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cloud.ply'
            save_ply(path, xyz, rgb)
            self.assertIn('element vertex 1', path.read_text())
            self.assertIn('30 20 10', path.read_text())

    def test_empty_cloud_and_detector(self):
        p = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], float)
        proj = project_points(np.empty((0, 4)), Calibration(p, np.eye(4), np.eye(4)), (128, 64))
        image = np.zeros((128, 64, 3), np.uint8)
        self.assertEqual(colorize_points(np.empty((0, 4)), image, proj)[0].shape, (0, 3))
        self.assertEqual(draw_overlay(image, proj, []).shape, image.shape)
        self.assertEqual(TraditionalDetector().detect(image), [])

    def test_nms_with_actual_boxes_and_svm_margins(self):
        from types import SimpleNamespace
        detector = TraditionalDetector()
        detector.detector = SimpleNamespace(detectMultiScale=lambda *a, **kw: (
            np.array([[0, 0, 64, 128], [1, 1, 64, 128], [100, 0, 64, 128]]),
            np.array([-1.0, 0.0, 2.0])))
        detections = detector.detect(np.zeros((256, 256, 3), np.uint8))
        self.assertEqual(len(detections), 2)
        self.assertEqual([d.score for d in detections], [2.0, 0.0])

    def test_malformed_velodyne(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bad.bin'
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                load_velodyne(path)

    def test_cli_synthetic_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root/'synthetic_sync'
            images = sequence/'image_02'/'data'
            lidar = sequence/'velodyne_points'/'data'
            images.mkdir(parents=True)
            lidar.mkdir(parents=True)
            (root/'calib_cam_to_cam.txt').write_text('R_rect_00: 1 0 0 0 1 0 0 0 1\nP_rect_02: 10 0 32 0 0 10 64 0 0 0 1 0\n')
            (root/'calib_velo_to_cam.txt').write_text('R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n')
            ok, encoded = cv2.imencode('.png', np.zeros((128, 64, 3), np.uint8))
            self.assertTrue(ok)
            encoded.tofile(images/'0000000000.png')
            np.array([[0, 0, 10, .5]], dtype='<f4').tofile(lidar/'0000000000.bin')
            output = root/'result'
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'main.py'),
                                     '--sequence', str(sequence), '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            import json
            report = json.loads((output/'0000000000_detections.json').read_text())
            self.assertEqual(report['projected_points'], 1)
            self.assertEqual(report['detections'], [])
            self.assertTrue((output/'0000000000_overlay.png').is_file())
            self.assertTrue((output/'0000000000_rgb.ply').is_file())


if __name__ == '__main__':
    unittest.main()
