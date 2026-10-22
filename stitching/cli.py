import argparse
from pathlib import Path
import sys

import cv2

from .io import main


def cli(argv=None):
    parser = argparse.ArgumentParser(description='Stitch two overlapping images using SIFT and homography.')
    parser.add_argument('image1', type=Path, help='left/reference image')
    parser.add_argument('image2', type=Path, help='right image')
    parser.add_argument('-o', '--output', type=Path, default=Path('panorama.jpg'),
                        help='panorama output path (default: panorama.jpg)')
    parser.add_argument('--report', type=Path, help='write JSON diagnostics, including processing failures')
    matching = parser.add_mutually_exclusive_group()
    matching.add_argument('--matching-output', type=Path, default=Path('matching.jpg'),
                          help='matching visualization path (default: matching.jpg)')
    matching.add_argument('--no-matching', action='store_true',
                          help='skip drawing and saving the matching visualization')
    args = parser.parse_args(argv)
    try:
        output = main(args.image1, args.image2, args.output,
                      matching_output=None if args.no_matching else args.matching_output,
                      report_output=args.report)
    except (ValueError, OSError, cv2.error) as exc:
        print(f'{parser.prog}: error: {exc}', file=sys.stderr)
        return 1
    print(f'Saved panorama to {output}')
    return 0
