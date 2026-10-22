import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from Image_Stitching import Image_Stitching, cli, main, read_image, write_image


SCRIPT = Path(__file__).resolve().parents[1] / 'Image_Stitching.py'


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / 'input.png'
        self.image = np.random.default_rng(42).integers(
            0, 256, size=(96, 1000, 3), dtype=np.uint8
        )
        self.assertTrue(cv2.imwrite(str(self.input), self.image))

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)],
            cwd=self.root, text=True, capture_output=True, timeout=30,
        )

    def test_help_and_missing_arguments(self):
        help_result = self.run_cli('--help')
        self.assertEqual(help_result.returncode, 0)
        self.assertIn('--no-matching', help_result.stdout)
        missing_result = self.run_cli()
        self.assertEqual(missing_result.returncode, 2)
        self.assertIn('required', missing_result.stderr)

    def test_missing_image_is_reported_without_traceback(self):
        result = self.run_cli('missing.png', self.input)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Input image is not a file', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_default_outputs_and_identity_stitch(self):
        result = self.run_cli(self.input, self.input)
        self.assertEqual(result.returncode, 0, result.stderr)
        panorama = cv2.imread(str(self.root / 'panorama.jpg'))
        matching = cv2.imread(str(self.root / 'matching.jpg'))
        self.assertIsNotNone(panorama)
        self.assertIsNotNone(matching)
        self.assertEqual(panorama.shape, self.image.shape)
        self.assertEqual(matching.shape[1], 2 * self.image.shape[1])

    def test_custom_output_without_matching_preserves_pixels(self):
        output = self.root / 'output with spaces' / 'panorama.png'
        with patch('Image_Stitching.cv2.drawMatchesKnn') as draw:
            with contextlib.redirect_stdout(io.StringIO()):
                status = cli([
                    str(self.input), str(self.input), '-o', str(output),
                    '--no-matching',
                ])
        self.assertEqual(status, 0)
        draw.assert_not_called()
        np.testing.assert_array_equal(cv2.imread(str(output)), self.image)

    def test_custom_matching_path(self):
        panorama = self.root / 'results' / 'panorama.png'
        matching = self.root / 'debug' / 'matches.png'
        result = self.run_cli(
            self.input, self.input, '-o', panorama,
            '--matching-output', matching,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(panorama.is_file())
        self.assertTrue(matching.is_file())
        self.assertFalse((self.root / 'matching.jpg').exists())

    def test_matching_options_are_mutually_exclusive(self):
        result = self.run_cli(
            self.input, self.input, '--no-matching',
            '--matching-output', 'debug.png',
        )
        self.assertEqual(result.returncode, 2)

    def test_output_cannot_overwrite_an_input(self):
        original_bytes = self.input.read_bytes()
        for output, matching in (
            (self.input, None),
            (self.root / 'panorama.png', self.input),
        ):
            with self.subTest(output=output, matching=matching):
                with self.assertRaisesRegex(ValueError, 'different from the input'):
                    main(self.input, self.input, output, matching_output=matching)
        self.assertEqual(self.input.read_bytes(), original_bytes)

    def test_outputs_cannot_share_a_path(self):
        output = self.root / 'output.png'
        with self.assertRaisesRegex(ValueError, 'paths must be different'):
            main(self.input, self.input, output, matching_output=output)

    def test_corrupt_image(self):
        corrupt = self.root / 'corrupt.png'
        corrupt.write_text('not an image')
        with self.assertRaisesRegex(ValueError, 'Cannot decode'):
            read_image(corrupt)

    def test_unsupported_output_fails_before_stitching(self):
        with patch('stitching.io.stitch') as blend:
            with self.assertRaisesRegex(ValueError, 'Unsupported output'):
                main(
                    self.input, self.input, self.root / 'output.invalid',
                    matching_output=None,
                )
        blend.assert_not_called()

    def test_encoder_failure_is_reported(self):
        with patch('Image_Stitching.cv2.imwrite', return_value=False):
            with self.assertRaisesRegex(OSError, 'Could not write'):
                write_image(self.root / 'output.png', self.image)

    def test_write_rounds_and_clips_float_pixels(self):
        output = self.root / 'pixels.png'
        write_image(output, np.array([[[-1.0, 1.6, 300.0]]]))
        np.testing.assert_array_equal(
            cv2.imread(str(output)), np.array([[[0, 2, 255]]], dtype=np.uint8)
        )

    def test_small_input_has_actionable_error(self):
        small = self.root / 'small.png'
        self.assertTrue(cv2.imwrite(str(small), self.image[:, :640]))
        result = self.run_cli(small, self.input, '--no-matching')
        self.assertEqual(result.returncode, 1)
        self.assertIn('at least 800 pixels wide', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_featureless_input_has_actionable_error(self):
        blank = np.zeros_like(self.image)
        with self.assertRaisesRegex(ValueError, 'Not enough image features'):
            Image_Stitching(matching_output=None).blending(blank, blank)

    def test_failed_homography_is_reported(self):
        with patch('Image_Stitching.cv2.findHomography', return_value=(None, None)):
            with self.assertRaisesRegex(ValueError, 'valid homography'):
                Image_Stitching(matching_output=None).registration(
                    self.image, self.image
                )

    def test_empty_crop_is_reported(self):
        red = np.full_like(self.image, (0, 0, 255))
        stitcher = Image_Stitching(matching_output=None)
        with patch.object(stitcher, 'registration', return_value=np.eye(3)):
            with self.assertRaisesRegex(ValueError, 'no valid output pixels'):
                stitcher.blending(red, red)


if __name__ == '__main__':
    unittest.main()
