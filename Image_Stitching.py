import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


class Image_Stitching():
    def __init__(self, matching_output='matching.jpg') :
        self.ratio=0.85
        self.min_match=10
        self.sift=cv2.SIFT_create()
        self.smoothing_window_size=800
        self.matching_output = matching_output

    def registration(self,img1,img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des2) < 2:
            raise ValueError('Not enough image features to find matches.')
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        for candidates in raw_matches:
            if len(candidates) < 2:
                continue
            m1, m2 = candidates
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        if self.matching_output is not None:
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
            write_image(self.matching_output, img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
        else:
            raise ValueError(
                f'Not enough good matches: found {len(good_points)}, '
                f'need more than {self.min_match}.'
            )
        if H is None or not np.isfinite(H).all():
            raise ValueError('Could not estimate a valid homography from the matches.')
        return H

    def create_mask(self,img1,img2,version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2):
        if img1.shape[1] < self.smoothing_window_size:
            raise ValueError(
                f'The first image must be at least {self.smoothing_window_size} '
                'pixels wide for the current blending window.'
            )
        H = self.registration(img1,img2)
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1,img2,version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1,img2,version='right_image')
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        result=panorama1+panorama2

        rows, cols = np.where(result[:, :, 0] != 0)
        if rows.size == 0:
            raise ValueError('The current cropping method found no valid output pixels.')
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result


def read_image(path):
    """Read a color image, reporting invalid paths before calling OpenCV."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Input image is not a file: {path}')
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot decode input image: {path}')
    return image


def prepare_output(path):
    path = Path(path)
    if path.exists() and not path.is_file():
        raise ValueError(f'Output path is not a file: {path}')
    if not cv2.haveImageWriter(str(path)):
        raise ValueError(f'Unsupported output image format: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_image(path, image):
    path = prepare_output(path)
    if image.dtype != np.uint8:
        # Make OpenCV's 8-bit output conversion explicit; blending stays float64.
        image = cv2.add(image, 0, dtype=cv2.CV_8U)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write output image: {path}')


def main(argv1, argv2, output='panorama.jpg', *, matching_output='matching.jpg'):
    """Stitch two files, retaining the original default output filenames."""
    inputs = [Path(argv1), Path(argv2)]
    outputs = [Path(output)]
    if matching_output is not None:
        outputs.append(Path(matching_output))
    resolved_inputs = {path.resolve() for path in inputs}
    resolved_outputs = [path.resolve() for path in outputs]
    if any(path in resolved_inputs for path in resolved_outputs):
        raise ValueError('Output paths must be different from the input images.')
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError('Panorama and matching image paths must be different.')
    img1, img2 = (read_image(path) for path in inputs)
    for path in outputs:
        prepare_output(path)
    final = Image_Stitching(matching_output=matching_output).blending(img1, img2)
    write_image(output, final)
    return Path(output)


def cli(argv=None):
    parser = argparse.ArgumentParser(
        description='Stitch two overlapping images using SIFT and homography.'
    )
    parser.add_argument('image1', type=Path, help='left/reference image')
    parser.add_argument('image2', type=Path, help='right image')
    parser.add_argument(
        '-o', '--output', type=Path, default=Path('panorama.jpg'),
        help='panorama output path (default: panorama.jpg)',
    )
    matching = parser.add_mutually_exclusive_group()
    matching.add_argument(
        '--matching-output', type=Path, default=Path('matching.jpg'),
        help='matching visualization path (default: matching.jpg)',
    )
    matching.add_argument(
        '--no-matching', action='store_true',
        help='skip drawing and saving the matching visualization',
    )
    args = parser.parse_args(argv)
    try:
        output = main(
            args.image1, args.image2, args.output,
            matching_output=None if args.no_matching else args.matching_output,
        )
    except (ValueError, OSError, cv2.error) as exc:
        print(f'{parser.prog}: error: {exc}', file=sys.stderr)
        return 1
    print(f'Saved panorama to {output}')
    return 0


if __name__ == '__main__':
    sys.exit(cli())
