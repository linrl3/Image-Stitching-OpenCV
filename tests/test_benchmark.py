import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stitching.benchmark import run_benchmark
from stitching.io import write_image


@unittest.skipUnless(importlib.util.find_spec('psutil'), 'Install the benchmark extra to test RSS collection')
class BenchmarkTests(unittest.TestCase):
    def test_fresh_worker_records_native_memory_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.png'
            image = np.random.default_rng(42).integers(0, 256, (96, 1000, 3), dtype=np.uint8)
            write_image(source, image)
            report = run_benchmark(source, source, scales=[1, .5], repeats=1)
        self.assertEqual([run['status'] for run in report['runs']], ['success', 'error'])
        self.assertGreater(report['runs'][0]['sampled_peak_rss_bytes'], 0)
        self.assertGreater(report['summary'][0]['median_compute_seconds'], 0)
        self.assertEqual(report['runs'][1]['error']['code'], 'image_too_small')
        self.assertEqual(report['summary'][1]['successful_runs'], 0)
        self.assertIsNone(report['summary'][1]['median_compute_seconds'])


if __name__ == '__main__':
    unittest.main()
