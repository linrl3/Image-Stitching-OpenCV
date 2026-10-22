"""Exact regression checks against the original dense float64 blending formula."""
import unittest

import cv2
import numpy as np

from stitching import StitchConfig
from stitching.core import blend_images, create_mask, new_report
from stitching.io import encode_image


def reference_mask(left, right, version, window):
    # Frozen dense implementation, independent of the optimized weight helper.
    height, width = left.shape[0], left.shape[1] + right.shape[1]
    offset = window // 2
    barrier = left.shape[1] - offset
    mask = np.zeros((height, width))
    if version == 'left_image':
        mask[:, barrier - offset:barrier + offset] = np.tile(
            np.linspace(1, 0, 2 * offset), (height, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset:barrier + offset] = np.tile(
            np.linspace(0, 1, 2 * offset), (height, 1))
        mask[:, barrier + offset:] = 1
    return cv2.merge([mask, mask, mask])


def reference_blend(left, right, homography, window):
    height, width = left.shape[0], left.shape[1] + right.shape[1]
    panorama1 = np.zeros((height, width, 3))
    panorama1[:, :left.shape[1]] = left
    panorama1 *= reference_mask(left, right, 'left_image', window)
    panorama2 = cv2.warpPerspective(right, homography, (width, height))
    panorama2 = panorama2 * reference_mask(left, right, 'right_image', window)
    result = panorama1 + panorama2
    rows, cols = np.where(result[:, :, 0] != 0)
    bounds = [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]
    left_edge, top, right_edge, bottom = bounds
    return result[top:bottom, left_edge:right_edge], bounds


class BlendingRegressionTests(unittest.TestCase):
    def test_block_blending_is_bit_exact_for_different_shapes_and_warps(self):
        rng = np.random.default_rng(91)
        for width, window in ((1000, 800), (800, 800), (1000, 2)):
            # Enough rows to cross scratch-buffer boundaries, with an uneven tail.
            left = rng.integers(0, 256, (321, width, 3), dtype=np.uint8)
            right = rng.integers(0, 256, (267, 713, 3), dtype=np.uint8)
            originals = left.copy(), right.copy()
            transforms = (
                np.eye(3),
                np.array([[1, 0, 420], [0, 1, 31], [0, 0, 1.]]),
                np.array([[.97, .02, -73], [-.01, 1.02, -21], [.0001, -.0002, 1.]]),
            )
            config = StitchConfig(smoothing_window_size=window)
            for transform in transforms:
                with self.subTest(width=width, window=window, transform=transform.tolist()):
                    expected, bounds = reference_blend(left, right, transform, window)
                    report = new_report(config)
                    actual = blend_images(left, right, transform, config, report)
                    self.assertEqual(actual.dtype, np.float64)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.tobytes(), expected.tobytes())
                    self.assertEqual(report['output']['crop_bounds'], bounds)
                    self.assertEqual(encode_image('.png', actual), encode_image('.png', expected))
                    for image, original in zip((left, right), originals):
                        np.testing.assert_array_equal(image, original)

    def test_crop_keeps_original_blue_channel_bounds(self):
        left = np.zeros((19, 800, 3), np.uint8)
        right = np.zeros((15, 37, 3), np.uint8)
        left[:, :, 1:] = 255  # Red/green-only borders must not expand the crop.
        left[3, 11, 0] = 17
        left[13, 711, 0] = 250
        expected, bounds = reference_blend(left, right, np.eye(3), 800)
        report = new_report()
        actual = blend_images(left, right, np.eye(3), StitchConfig(), report)
        self.assertEqual(bounds, [11, 3, 712, 14])
        self.assertEqual(report['output']['crop_bounds'], bounds)
        np.testing.assert_array_equal(actual, expected)

    def test_legacy_mask_values_and_writable_array_are_preserved(self):
        left = np.zeros((3, 1000, 3), np.uint8)
        right = np.zeros((5, 123, 3), np.uint8)
        for version in ('left_image', 'right_image'):
            with self.subTest(version=version):
                mask = create_mask(left, right, version, StitchConfig())
                np.testing.assert_array_equal(mask, reference_mask(left, right, version, 800))
                self.assertEqual(mask.dtype, np.float64)
                self.assertTrue(mask.flags.c_contiguous)
                unchanged = mask[1, 1, 1]
                mask[0, 1, 0] = 42
                self.assertEqual(mask[1, 1, 1], unchanged)


if __name__ == '__main__':
    unittest.main()
