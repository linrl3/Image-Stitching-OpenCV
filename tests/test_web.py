import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from stitching.io import encode_image
from stitching.web import create_app


@unittest.skipUnless(importlib.util.find_spec('flask'), 'Install the web extra to test the local UI')
class WebTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.testing = True
        self.client = self.app.test_client()
        scene = np.random.default_rng(420).integers(16, 240, (160, 1400, 3), dtype=np.uint8)
        self.images = [encode_image('.png', scene[:, :1000]), encode_image('.png', scene[:, 400:])]
        cv2.setRNGSeed(42)

    def upload(self, images=None, matching=True):
        response = self.client.post('/api/stitch', data={
            'image1': (BytesIO((images or self.images)[0]), 'left.png'),
            'image2': (BytesIO((images or self.images)[1]), 'right.png'),
            'matching': str(matching).lower(),
        })
        self.addCleanup(response.close)
        self.addCleanup(response.request.close)
        self.addCleanup(response.request.input_stream.close)
        return response

    @unittest.skipUnless(os.environ.get('STITCHING_FULL_SAMPLE'), 'Full-resolution sample runs in the primary CI job')
    def test_original_size_sample_matches_repository_reference(self):
        response = self.client.post('/api/sample', data={'matching': 'false'})
        self.addCleanup(response.close)
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload['report']['source']['sample_scale'], 1)
        self.assertEqual(payload['report']['inputs'][0]['width'], 2448)
        self.assertEqual(payload['report']['inputs'][0]['height'], 3264)
        with self.client.get(payload['files']['panorama.png']) as download:
            image = cv2.imdecode(np.frombuffer(download.data, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[0], 3264)
        self.assertLess(abs(image.shape[1] - 4289), 40)
        reference = cv2.imread(str(Path(__file__).resolve().parents[1] / 'images' / 'panorama.jpg'))
        # Compare small previews to tolerate historic OpenCV/JPEG differences,
        # while detecting the large visible seam caused by resizing sample inputs.
        preview = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        expected = cv2.resize(reference, (640, 480), interpolation=cv2.INTER_AREA)
        self.assertLess(np.abs(preview.astype(float) - expected).mean(), 6)
        if os.environ.get('STITCHING_ARTIFACT_DIR'):
            from stitching.io import write_image, write_report
            destination = Path(os.environ['STITCHING_ARTIFACT_DIR'])
            write_image(destination / 'building-original.jpg', image)
            write_report(destination / 'building-original.json', payload['report'])

    def test_page_and_packaged_assets(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        for path in ('/static/style.css', '/static/app.js'):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.data), 100)
            response.close()

    def test_upload_preview_and_download_without_disk_writes(self):
        with patch('cv2.imwrite', side_effect=AssertionError('Unexpected file output')):
            response = self.upload()
        self.assertEqual(response.status_code, 200, response.get_json())
        payload = response.get_json()
        self.assertEqual(payload['report']['status'], 'success')
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        for name, url in payload['files'].items():
            download = self.client.get(url + '?download=1')
            self.assertEqual(download.status_code, 200)
            self.assertIn('attachment;', download.headers['Content-Disposition'])
            if name == 'report.json':
                self.assertEqual(json.loads(download.data), payload['report'])
            else:
                image = cv2.imdecode(np.frombuffer(download.data, np.uint8), cv2.IMREAD_COLOR)
                self.assertIsNotNone(image)
            download.close()

    def test_optional_matching_and_latest_result_retention(self):
        first = self.upload(matching=False).get_json()
        self.assertNotIn('matching.jpg', first['files'])
        second = self.upload(matching=False).get_json()
        self.assertEqual(self.client.get(first['files']['panorama.png']).status_code, 404)
        response = self.client.get(second['files']['panorama.png'])
        self.assertEqual(response.status_code, 200)
        response.close()

    def test_sample_can_be_reordered(self):
        images = [cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) for data in self.images]
        with patch('stitching.web.read_image', side_effect=images):
            response = self.client.post('/api/sample', data={'reverse': 'true', 'matching': 'false'})
        self.addCleanup(response.close)
        self.assertEqual(response.status_code, 200, response.get_json())
        report = response.get_json()['report']
        self.assertIn('q22.jpg', report['source']['names'][0])
        self.assertAlmostEqual(report['registration']['homography_image2_to_image1'][0][2], -400, delta=1)

    def test_bad_input_reports_and_releases_processing_lock(self):
        for images, code in (([b'not an image', b'x'], 'decode_failed'),
                             ([b'', b''], 'decode_failed'),
                             ([encode_image('.png', np.zeros((20, 20, 3), np.uint8))] * 2, 'image_too_small')):
            with self.subTest(code=code):
                response = self.upload(images)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.get_json()['report']['error']['code'], code)
                self.assertFalse(self.app.extensions['stitching_processing'].locked())
        self.assertEqual(self.client.post('/api/stitch').status_code, 422)
        self.assertEqual(self.upload().status_code, 200)

    def test_upload_and_pixel_limits(self):
        self.app.config['MAX_CONTENT_LENGTH'] = 100
        self.assertEqual(self.upload().status_code, 413)
        self.app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
        self.app.config['MAX_IMAGE_PIXELS'] = 100
        response = self.upload()
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()['report']['error']['code'], 'image_too_large')

    def test_busy_and_cross_origin_requests(self):
        lock = self.app.extensions['stitching_processing']
        with lock:
            self.assertEqual(self.upload().status_code, 409)
        response = self.client.post('/api/stitch', headers={'Origin': 'https://example.com'})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get('/results/missing/report.json').status_code, 404)


if __name__ == '__main__':
    unittest.main()
