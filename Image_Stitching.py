"""Backward-compatible script and class; new code can import from stitching."""
import sys
import cv2
from stitching.cli import cli
from stitching.core import (
    StitchConfig, blend_images, create_mask, new_report, register_images, validate_images,
)
from stitching.io import main, prepare_output, read_image, write_image


class Image_Stitching:
    def __init__(self, matching_output='matching.jpg'):
        self.ratio = 0.85
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.smoothing_window_size = 800
        self.matching_output = matching_output

    def _config(self):
        return StitchConfig(self.ratio, self.min_match, self.smoothing_window_size)

    def registration(self, img1, img2):
        config = self._config()
        H, matching = register_images(
            img1, img2, config, new_report(config),
            include_matching=self.matching_output is not None, sift=self.sift,
        )
        if self.matching_output is not None:
            write_image(self.matching_output, matching)
        return H

    def create_mask(self, img1, img2, version):
        return create_mask(img1, img2, version, self._config())

    def blending(self, img1, img2):
        config = self._config()
        report = new_report(config)
        validate_images(img1, img2, config, report)
        return blend_images(img1, img2, self.registration(img1, img2), config, report)


if __name__ == '__main__':
    sys.exit(cli())
