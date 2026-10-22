import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stitching import StitchConfig, StitchingError, stitch
from stitching.io import stitch_files, write_image, write_report


ROOT = Path(__file__).resolve().parents[1]


def overlapping_pair(offset=400, darker=False):
    scene = np.random.default_rng(420).integers(16, 240, (224, 1800, 3), dtype=np.uint8)
    left, right = scene[:, :1000].copy(), scene[:, offset:offset + 1000].copy()
    if darker:
        right = cv2.convertScaleAbs(right, alpha=.8, beta=12)
    return left, right


def save_artifact(name, result):
    destination = os.environ.get('STITCHING_ARTIFACT_DIR')
    if destination:
        root = Path(destination)
        write_image(root / f'{name}.jpg', result.panorama)
        write_report(root / f'{name}.json', result.diagnostics)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        cv2.setRNGSeed(42)

    def test_distinct_overlapping_images_recover_known_geometry(self):
        # Held-out coordinates test geometry independently of the fitted keypoints.
        points = np.float32([[[20, 20], [500, 110], [950, 200]]])
        for offset, darker in ((400, False), (800, False), (400, True)):
            with self.subTest(offset=offset, darker=darker):
                images = overlapping_pair(offset, darker)
                result = stitch(*images)
                predicted = cv2.perspectiveTransform(points, result.homography)
                np.testing.assert_allclose(predicted, points + [offset, 0], atol=1)
                self.assertEqual(result.panorama.shape[0], 224)
                self.assertLessEqual(abs(result.panorama.shape[1] - (1000 + offset)), 2)
                self.assertGreater(result.diagnostics['registration']['inliers'], 30)
                self.assertLess(result.diagnostics['registration']['p95_inlier_error_px'], 1)
                save_artifact(f'overlap-{offset}-darker-{darker}', result)

    def test_pure_api_leaves_inputs_unchanged_and_does_not_write(self):
        images = overlapping_pair()
        originals = [image.copy() for image in images]
        with patch('cv2.imwrite', side_effect=AssertionError('Unexpected file output')):
            result = stitch(*images, include_matching=True)
        for image, original in zip(images, originals):
            np.testing.assert_array_equal(image, original)
        self.assertEqual(result.matching_image.shape, (224, 2000, 3))
        self.assertEqual(result.diagnostics['status'], 'success')
        self.assertEqual(result.diagnostics['schema_version'], 1)
        self.assertIn('compute_total', result.diagnostics['timings_seconds'])
        json.dumps(result.diagnostics, allow_nan=False)

    def test_real_building_pair(self):
        images = [cv2.resize(cv2.imread(str(ROOT / 'images' / name)), None,
                             fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
                  for name in ('q11.jpg', 'q22.jpg')]
        result = stitch(*images)
        report = result.diagnostics
        self.assertEqual(result.panorama.shape[0], 1632)
        self.assertGreater(result.panorama.shape[1], 1900)
        self.assertLess(result.panorama.shape[1], 2300)
        self.assertGreater(report['registration']['inliers'], 100)
        # Repeated building details produce outliers; use the measured baseline
        # (about 41% inliers, 2.27 px median), with room for platform differences.
        self.assertGreater(report['registration']['inlier_ratio'], .3)
        self.assertLess(report['registration']['median_inlier_error_px'], 3)
        self.assertLess(report['registration']['p95_inlier_error_px'], 5)
        self.assertTrue(np.isfinite(result.panorama).all())
        save_artifact('building-half', result)

    def test_failure_reports_remain_json_serializable(self):
        cases = [
            (np.zeros((10, 10), np.uint8), 'invalid_image'),
            (np.zeros((20, 600, 3), np.uint8), 'image_too_small'),
            (np.zeros((100, 1000, 3), np.uint8), 'insufficient_features'),
        ]
        for image, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(StitchingError) as caught:
                    stitch(image, image)
                report = caught.exception.diagnostics
                self.assertEqual(report['status'], 'error')
                self.assertEqual(report['error']['code'], code)
                self.assertIn('compute_total', report['timings_seconds'])
                json.dumps(report, allow_nan=False)

    def test_failed_homography_retains_feature_and_match_counts(self):
        with patch('cv2.findHomography', return_value=(None, None)):
            with self.assertRaises(StitchingError) as caught:
                stitch(*overlapping_pair())
        report = caught.exception.diagnostics
        self.assertEqual(report['error']['stage'], 'homography')
        self.assertGreater(report['matches']['ratio_passed'], 10)
        self.assertGreater(report['features']['image1'], 0)

    def test_config_validation(self):
        for kwargs in ({'ratio': 1}, {'min_match': 2}, {'smoothing_window_size': 7},
                       {'ransac_reprojection_threshold': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StitchConfig(**kwargs)

    def test_cli_writes_success_and_failure_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, image in zip(('left.png', 'right.png'), overlapping_pair()):
                write_image(root / name, image)
            for input_name, expected_status in (('left.png', 'success'), ('missing.png', 'error')):
                with self.subTest(expected_status=expected_status):
                    report_path = root / f'{expected_status}.json'
                    command = [sys.executable, str(ROOT / 'Image_Stitching.py'),
                               str(root / input_name), str(root / 'right.png'),
                               '-o', str(root / 'result.png'), '--no-matching', '--report', str(report_path)]
                    completed = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=30)
                    self.assertEqual(completed.returncode, 0 if expected_status == 'success' else 1, completed.stderr)
                    report = json.loads(report_path.read_text())
                    self.assertEqual(report['status'], expected_status)
                    self.assertIn('file_pipeline_total', report['timings_seconds'])
                    self.assertNotIn('Traceback', completed.stderr)

    def test_report_cannot_overwrite_input_or_panorama(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.png'
            write_image(source, overlapping_pair()[0])
            before = source.read_bytes()
            output = root / 'output.png'
            for report in (source, output):
                with self.subTest(report=report), self.assertRaises(ValueError):
                    stitch_files(source, source, output, matching_output=None, report_output=report)
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(output.exists())

    def test_compute_failure_is_written_to_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blank = root / 'blank.png'
            write_image(blank, np.zeros((100, 1000, 3), np.uint8))
            with self.assertRaises(StitchingError):
                stitch_files(blank, blank, root / 'panorama.png', matching_output=None,
                             report_output=root / 'report.json')
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(report['error']['code'], 'insufficient_features')
            self.assertFalse((root / 'panorama.png').exists())


if __name__ == '__main__':
    unittest.main()
