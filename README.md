# Image-Stitching-OpenCV

[![CI](https://github.com/linrl3/Image-Stitching-OpenCV/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/linrl3/Image-Stitching-OpenCV/actions/workflows/ci.yml)

Simple image stitching algorithm using SIFT, homography, KNN and RANSAC in Python.

The project is to implement a featured based automatic image stitching algorithm. When we input two images with overlapped fields, we expect to obtain a wide seamless panorama.

We use scale invariant features transform(SIFT) to extract local features of the input images, K nearest neighbors algorithms to match these features and Random sample consensus(Ransac) to calculate the homograph matrix, which will be used for image warping. Finally we apply a weighted matrix as a mask for image blending.

## Setup with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository root:

```sh
uv sync --locked
uv run image-stitching images/q11.jpg images/q22.jpg
```

The project selects Python 3.12 via `.python-version` and supports Python 3.10+.
uv manages the local `.venv`; `uv.lock` records the resolved dependency versions.
The runtime dependencies are NumPy and OpenCV 4.x. The headless OpenCV package
provides image processing without GUI dependencies; no contrib package is needed
for `cv2.SIFT_create()`.

## Usage

Both the installed command and the original script entry point are supported:

```sh
uv run image-stitching images/q11.jpg images/q22.jpg
uv run python Image_Stitching.py images/q11.jpg images/q22.jpg
uv run image-stitching --help
```

By default, `panorama.jpg` and `matching.jpg` are written to the current directory.
Existing output files are replaced. Input files cannot be used as output paths.
Choose output paths with:

```sh
uv run image-stitching images/q11.jpg images/q22.jpg \
  --output output/panorama.png \
  --matching-output output/matching.jpg
```

Parent directories are created automatically. To skip drawing and saving the
matching visualization, reducing work and temporary memory use:

```sh
uv run image-stitching images/q11.jpg images/q22.jpg \
  --output output/panorama.jpg --no-matching
```

`--matching-output` and `--no-matching` are mutually exclusive. The command exits
with status 0 on success, 1 for image processing or file errors, and 2 for invalid
arguments. Errors are printed to stderr.

The Python API remains available:

```python
from Image_Stitching import main

main("images/q11.jpg", "images/q22.jpg", "output/panorama.jpg", matching_output=None)
```

## Current algorithm constraints

The SIFT parameters, ratio test (0.85), minimum match condition (more than 10),
RANSAC threshold (5 pixels), fixed canvas, 800-pixel blending window and cropping
calculation are retained. The modern SIFT API replaces the old `xfeatures2d`
entry point; failure checks report unsupported inputs without changing successful
stitching calculations.

- Supply the left/reference image first and an overlapping right image second.
- The first image must be at least 800 pixels wide.
- Images need enough texture and overlap for feature matching.
- The fixed canvas can clip content after vertical shifts or negative-coordinate
  warps; the fixed blending window may darken areas when overlap is unsuitable.
- Cropping still uses nonzero blue-channel pixels. Some dark or pure red/green
  inputs are unsupported by this method.
- Large images can require substantial memory because blending uses full-size
  float64 arrays. This version does not resize input images automatically.

## Development

Run the CLI, file handling and failure-path tests with the standard library test
runner:

```sh
uv run python -m unittest discover -s tests -v
```

After intentionally changing dependencies, run `uv lock` and commit both
`pyproject.toml` and `uv.lock`. For a reproducible environment, use
`uv sync --locked`. See [uv's project guide](https://docs.astral.sh/uv/guides/projects/).

## Continuous integration

The [CI workflow](.github/workflows/ci.yml) runs on pushes, pull requests and manual
dispatches from the GitHub Actions tab. It uses uv 0.9.17 and checks:

- Locked dependency installation, the test suite and the installed CLI on Linux
  with Python 3.10, 3.12 and 3.14, and on macOS and Windows with Python 3.12.
- Source distribution and wheel builds after all test jobs pass.
- Installation of the wheel with locked runtime dependencies into a clean
  environment, followed by a CLI check outside the source checkout.

Successful runs provide the wheel and source archive as a `python-distributions`
artifact, available for 14 days from the run page. The workflow does not publish
to PyPI or deploy a service.

To reproduce the primary checks locally:

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked image-stitching --help
uv build --no-sources
```

## Sample 

## Input images
<img src="https://github.com/linrl3/Image-Stitching-OpenCV/blob/main/images/q11.jpg" width=300 height=400 >   <img src="https://github.com/linrl3/Image-Stitching-OpenCV/blob/main/images/q22.jpg" width=300 height=400 >

## Matching
![matching](https://github.com/linrl3/Image-Stitching-OpenCV/blob/main/images/matching.jpg)

## Output image
![pano](https://github.com/linrl3/Image-Stitching-OpenCV/blob/main/images/panorama.jpg)

## other examples

Room:
![room](https://github.com/linrl3/Image-Stitching-OpenCV/blob/main/images/room.jpg)

Another building: This is a stitching of three related images. It doesn't work so well. Trying to improve.
![gym](https://github.com/linrl3/Image-Stitching-OpenCV/blob/main/images/gym.jpg)
