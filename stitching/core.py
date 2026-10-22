"""The original SIFT / homography / weighted blending algorithm, with diagnostics."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import platform
from time import perf_counter

import cv2
import numpy as np


@dataclass(frozen=True)
class StitchConfig:
    ratio: float = 0.85
    min_match: int = 10
    smoothing_window_size: int = 800
    ransac_reprojection_threshold: float = 5.0

    def __post_init__(self):
        if not 0 < self.ratio < 1:
            raise ValueError('ratio must be between 0 and 1.')
        if not isinstance(self.min_match, int) or self.min_match < 3:
            raise ValueError('min_match must be an integer of at least 3.')
        if (not isinstance(self.smoothing_window_size, int)
                or self.smoothing_window_size < 2 or self.smoothing_window_size % 2):
            raise ValueError('smoothing_window_size must be a positive even integer.')
        if (not np.isfinite(self.ransac_reprojection_threshold)
                or self.ransac_reprojection_threshold <= 0):
            raise ValueError('ransac_reprojection_threshold must be positive and finite.')


@dataclass
class StitchResult:
    panorama: np.ndarray
    homography: np.ndarray
    diagnostics: dict
    matching_image: np.ndarray | None = None


class StitchingError(ValueError):
    """A failure with a JSON-serializable partial diagnostic report."""
    def __init__(self, message, *, code, stage, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics
        diagnostics['status'] = 'error'
        diagnostics['error'] = {'code': code, 'stage': stage, 'message': message}


def new_report(config=None):
    return {
        'schema_version': 1, 'status': 'running',
        'environment': {
            'python': platform.python_version(), 'opencv': cv2.__version__,
            'numpy': np.__version__, 'platform': platform.platform(),
            'opencv_threads': cv2.getNumThreads(),
        },
        'config': asdict(config or StitchConfig()), 'timings_seconds': {},
    }


@contextmanager
def timed(report, stage):
    start = perf_counter()
    try:
        yield
    finally:
        report['timings_seconds'][stage] = perf_counter() - start


def fail(report, code, stage, message):
    raise StitchingError(message, code=code, stage=stage, diagnostics=report)


def validate_images(img1, img2, config, report):
    for index, image in enumerate((img1, img2), 1):
        if (not isinstance(image, np.ndarray) or image.ndim != 3
                or image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0):
            fail(report, 'invalid_image', 'input',
                 f'Image {index} must be a nonempty uint8 BGR array with three channels.')
    report['inputs'] = [
        {'width': image.shape[1], 'height': image.shape[0], 'dtype': str(image.dtype)}
        for image in (img1, img2)
    ]
    if img1.shape[1] < config.smoothing_window_size:
        fail(report, 'image_too_small', 'input',
             f'The first image must be at least {config.smoothing_window_size} '
             'pixels wide for the current blending window.')


def register_images(img1, img2, config, report, *, include_matching=False, sift=None):
    with timed(report, 'features'):
        sift = sift if sift is not None else cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        report['features'] = {'image1': len(kp1), 'image2': len(kp2)}
        if des1 is None or des2 is None or len(des2) < 2:
            fail(report, 'insufficient_features', 'features', 'Not enough image features to find matches.')
    with timed(report, 'matching'):
        raw_matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
        good_points, good_matches = [], []
        for candidates in raw_matches:
            if len(candidates) < 2:
                continue
            m1, m2 = candidates
            if m1.distance < config.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        report['matches'] = {'knn_queries': len(raw_matches), 'ratio_passed': len(good_points)}
        if len(good_points) <= config.min_match:
            fail(report, 'insufficient_matches', 'matching',
                 f'Not enough good matches: found {len(good_points)}, need more than {config.min_match}.')
    with timed(report, 'homography'):
        image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
        H, status = cv2.findHomography(
            image2_kp, image1_kp, cv2.RANSAC, config.ransac_reprojection_threshold
        )
        if H is None or not np.isfinite(H).all():
            fail(report, 'invalid_homography', 'homography',
                 'Could not estimate a valid homography from the matches.')
    with timed(report, 'diagnostics'):
        inliers = status.ravel().astype(bool) if status is not None else np.zeros(len(good_points), bool)
        projected = cv2.perspectiveTransform(image2_kp.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(projected - image1_kp, axis=1)[inliers]
        finite_errors = errors[np.isfinite(errors)]
        report['registration'] = {
            'homography_image2_to_image1': H.tolist(),
            'inliers': int(inliers.sum()), 'inlier_ratio': float(inliers.mean()),
            'median_inlier_error_px': float(np.median(finite_errors)) if finite_errors.size else None,
            'p95_inlier_error_px': float(np.percentile(finite_errors, 95)) if finite_errors.size else None,
        }
        h2, w2 = img2.shape[:2]
        corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        h1, w1 = img1.shape[:2]
        if np.isfinite(warped).all():
            bounds = [*warped.min(axis=0).tolist(), *warped.max(axis=0).tolist()]
            report['geometry'] = {
                'canvas_width': w1 + w2, 'canvas_height': h1,
                'warped_image2_bounds': bounds,
                'extends_beyond_canvas': bool(bounds[0] < -1 or bounds[1] < -1
                    or bounds[2] > w1 + w2 + 1 or bounds[3] > h1 + 1),
            }
    matching_image = None
    if include_matching:
        with timed(report, 'matching_visualization'):
            matching_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
    return H, matching_image


def _blend_weights(img1, img2, version, config):
    """One float64 weight per column; all rows and channels use the same values."""
    width_panorama = img1.shape[1] + img2.shape[1]
    offset = int(config.smoothing_window_size / 2)
    barrier = img1.shape[1] - offset
    weights = np.zeros(width_panorama)
    if version == 'left_image':
        weights[barrier - offset:barrier + offset] = np.linspace(1, 0, 2 * offset)
        weights[:barrier - offset] = 1
    else:
        weights[barrier - offset:barrier + offset] = np.linspace(0, 1, 2 * offset)
        weights[barrier + offset:] = 1
    return weights


def create_mask(img1, img2, version, config):
    # Preserve the legacy public API's writable, independent H x W x 3 array.
    weights = _blend_weights(img1, img2, version, config)
    return np.broadcast_to(weights[None, :, None], (img1.shape[0], weights.size, 3)).copy()


def blend_images(img1, img2, H, config, report):
    height_panorama = img1.shape[0]
    width_panorama = img1.shape[1] + img2.shape[1]
    with timed(report, 'blend_left'):
        result = np.zeros((height_panorama, width_panorama, 3))
        weights = _blend_weights(img1, img2, 'left_image', config)
        np.multiply(img1, weights[None, :img1.shape[1], None],
                    out=result[:, :img1.shape[1], :])
    with timed(report, 'warp'):
        warped = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))
    with timed(report, 'blend_right'):
        weights = _blend_weights(img1, img2, 'right_image', config)
        # Keep the original multiply-then-add order and float64 precision, with
        # at most 8 MiB of scratch space (or one row for an unusually wide input).
        block_rows = max(1, min(height_panorama, (8 * 1024 * 1024) // (width_panorama * 3 * 8)))
        scratch = np.empty((block_rows, width_panorama, 3))
        for start in range(0, height_panorama, block_rows):
            stop = min(start + block_rows, height_panorama)
            block = scratch[:stop - start]
            np.multiply(warped[start:stop], weights[None, :, None], out=block)
            np.add(result[start:stop], block, out=result[start:stop])
        del warped, scratch, block
    with timed(report, 'crop'):
        nonzero = result[:, :, 0] != 0
        rows = np.flatnonzero(nonzero.any(axis=1))
        if rows.size == 0:
            fail(report, 'empty_crop', 'crop', 'The current cropping method found no valid output pixels.')
        cols = np.flatnonzero(nonzero.any(axis=0))
        min_row, max_row = rows[0], rows[-1] + 1
        min_col, max_col = cols[0], cols[-1] + 1
        final = result[min_row:max_row, min_col:max_col, :]
        report['output'] = {
            'width': final.shape[1], 'height': final.shape[0], 'dtype': str(final.dtype),
            'crop_bounds': [int(min_col), int(min_row), int(max_col), int(max_row)],
        }
    return final


def stitch(image1, image2, config=None, *, include_matching=False):
    """Stitch uint8 BGR arrays without file I/O or implicit input resizing.

    Diagnostic metrics describe the fit; they do not change acceptance thresholds.
    """
    config = config or StitchConfig()
    report = new_report(config)
    start = perf_counter()
    try:
        validate_images(image1, image2, config, report)
        H, matching = register_images(image1, image2, config, report, include_matching=include_matching)
        panorama = blend_images(image1, image2, H, config, report)
        report['status'] = 'success'
        return StitchResult(panorama, H, report, matching)
    except cv2.error as exc:
        fail(report, 'opencv_error', 'compute', str(exc))
    finally:
        report['timings_seconds']['compute_total'] = perf_counter() - start
