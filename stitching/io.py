"""File handling for the CLI and callers that work with paths."""
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .core import StitchingError, new_report, stitch


def read_image(path):
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


def to_uint8(image):
    return image if image.dtype == np.uint8 else cv2.add(image, 0, dtype=cv2.CV_8U)


def encode_image(extension, image):
    ok, encoded = cv2.imencode(extension, to_uint8(image))
    if not ok:
        raise OSError(f'Could not encode output image as {extension}.')
    return encoded.tobytes()


def write_image(path, image):
    path = prepare_output(path)
    if not cv2.imwrite(str(path), to_uint8(image)):
        raise OSError(f'Could not write output image: {path}')


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_paths(inputs, outputs):
    resolved_inputs = {Path(path).resolve() for path in inputs}
    resolved_outputs = [Path(path).resolve() for path in outputs if path is not None]
    if any(path in resolved_inputs for path in resolved_outputs):
        raise ValueError('Output paths must be different from the input images.')
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError('Panorama, matching image and report paths must be different.')
    for index, output in enumerate(resolved_outputs):
        if output.exists() and any(other.exists() and output.samefile(other)
                                   for other in resolved_outputs[index + 1:]):
            raise ValueError('Panorama, matching image and report paths must be different.')
    for output in resolved_outputs:
        if output.exists():
            if not output.is_file():
                raise ValueError(f'Output path is not a file: {output}')
            if any(path.exists() and output.samefile(path) for path in resolved_inputs):
                raise ValueError('Output paths must be different from the input images.')


def stitch_files(image1, image2, output='panorama.jpg', *,
                 matching_output='matching.jpg', report_output=None, config=None):
    # Validate collisions before writing anything, including failure reports.
    validate_paths([image1, image2], [output, matching_output, report_output])
    report = new_report(config)
    start = perf_counter()
    stage = 'input'
    read_seconds = 0.0
    try:
        read_start = perf_counter()
        img1, img2 = read_image(image1), read_image(image2)
        read_seconds = perf_counter() - read_start
        stage = 'output'
        prepare_output(output)
        if matching_output is not None:
            prepare_output(matching_output)
        stage = 'compute'
        result = stitch(img1, img2, config, include_matching=matching_output is not None)
        report = result.diagnostics
        stage = 'output'
        write_start = perf_counter()
        write_image(output, result.panorama)
        if matching_output is not None:
            write_image(matching_output, result.matching_image)
        report['timings_seconds']['write'] = perf_counter() - write_start
        return result
    except (ValueError, OSError, cv2.error) as exc:
        if isinstance(exc, StitchingError):
            report = exc.diagnostics
        else:
            report['status'] = 'error'
            report['error'] = {'code': 'file_error', 'stage': stage, 'message': str(exc)}
        raise
    finally:
        report['timings_seconds']['read'] = read_seconds
        report['timings_seconds']['file_pipeline_total'] = perf_counter() - start
        report['files'] = {
            'image1': str(image1), 'image2': str(image2), 'panorama': str(output),
            'matching': str(matching_output) if matching_output is not None else None,
        }
        if report_output is not None:
            write_report(report_output, report)


def main(argv1, argv2, output='panorama.jpg', *, matching_output='matching.jpg', report_output=None):
    stitch_files(argv1, argv2, output, matching_output=matching_output, report_output=report_output)
    return Path(output)
