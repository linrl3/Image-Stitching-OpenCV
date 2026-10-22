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

Save a machine-readable diagnostic report alongside the panorama:

```sh
uv run image-stitching images/q11.jpg images/q22.jpg \
  -o output/panorama.png --no-matching --report output/report.json
```

Reports include input/output sizes, dependency versions, parameters, feature and
match counts, RANSAC inliers, reprojection errors, the homography, canvas bounds,
and per-stage timings. Processing failures produce partial reports with an error
code, stage and message. Path collisions are rejected before any file is written.
Metrics describe the fitted geometry; they are not a visual quality score and do
not change the algorithm's acceptance thresholds.

## Python API

The computation API accepts uint8 BGR NumPy arrays, leaves inputs unchanged, and
does no file I/O or automatic resizing:

```python
import cv2
from stitching import StitchConfig, StitchingError, stitch

left = cv2.imread("images/q11.jpg")
right = cv2.imread("images/q22.jpg")
try:
    result = stitch(left, right, StitchConfig(), include_matching=True)
    panorama = result.panorama          # Original float64 output
    homography = result.homography      # Maps right-image coordinates to the left
    report = result.diagnostics         # JSON-serializable dict
    matches = result.matching_image     # None unless requested
except StitchingError as error:
    print(error.diagnostics)
```

For file handling, `stitching.io.stitch_files()` returns the same result and saves
the requested images/report. `stitching.io.to_uint8()` converts the float output
with the same rounding/clipping used by the CLI. The original API remains available:

```python
from Image_Stitching import main

main("images/q11.jpg", "images/q22.jpg", "output/panorama.jpg", matching_output=None)
```

## Local browser interface

```sh
uv run --locked --extra web image-stitching-web
```

Open **http://127.0.0.1:7860**. Choose two images, stitch, then download the PNG.
The **中文 / EN** switch translates controls, status and errors without resetting
your images or result. The initial language follows your browser; your choice is
remembered locally. Feature matches, metrics and the JSON report are under
**Details / 详情**, collapsed by default. Failure reports have a download link
beside the error message. Use `--port 7861` to choose another port.

The server binds only to loopback, processes on your computer and requires no
external frontend services. It handles one stitch at a time and retains only the
latest successful result in memory; starting another successful stitch invalidates
older download URLs. Stopping the server clears results. Uploads are limited to
32 MiB per request and 12 million pixels per image.

In a repository checkout, **Try example / 试用示例** runs `q11.jpg` and `q22.jpg` at
their original 2448×3264 size. Only display previews are reduced. Resizing the
inputs changes feature matches and the relative width of the fixed 800-pixel
blend, so a half-size computation is not an equivalent preview of the full-size
result. Historic OpenCV versions may produce slightly different output geometry.
The sample photos are not bundled into the wheel.

### Vercel deployment

Vercel supports Flask, but this local application needs deployment changes first.
The original-size example produces a PNG of about 15.1 MB and used about 2.50 GiB
peak sampled RSS in a local measurement. These conflict with the default function
payload/memory budgets, and the current in-process result store cannot reliably
serve downloads across function instances. A practical option is to host the
frontend on Vercel and run OpenCV in a separate container, with object storage for
uploads/results. See the [deployment assessment](docs/vercel.md) for measurements,
official limits and the required changes. No cloud deployment is configured yet.

## Performance measurements

```sh
uv run --locked --extra benchmark image-stitching-benchmark \
  images/q11.jpg images/q22.jpg --scales .5 .75 --repeats 3 \
  --output output/benchmark.json
```

Each repetition runs in a fresh process. The report records per-stage wall time,
sampled process RSS (including OpenCV/native allocations), input checksums and
environment details. `--scales` explicitly resizes inputs for measurement only;
the CLI and web interface keep original input dimensions. Use `--scales 1` for an
original-size measurement. Matching visualization is disabled during benchmarks.

Version 0.2 releases blending masks as soon as they have been used, preserving the
float64 arithmetic. In the recorded half-size comparison, maximum sampled RSS
fell from 1.08 GiB to 0.85 GiB (about 21%), while compute time remained similar.
Raw panorama pixels and encoded JPEG hashes were identical in all six comparison
runs. These are local observations, not performance guarantees. See the
[measurement details and recorded reports](benchmarks/README.md).

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

Install optional dependencies to run the complete suite with the standard library
test runner:

```sh
uv sync --locked --all-extras
uv run --locked --all-extras python -m unittest discover -s tests -v
```

Tests cover known translations, low overlap, brightness differences, the real
building pair, CLI/error reports, the computation API, web uploads/downloads,
request limits and benchmark workers. Set `STITCHING_FULL_SAMPLE=1` to additionally
compare the original-size web sample with the repository's reference image.
Set `STITCHING_ARTIFACT_DIR=output/regression` to save regression previews/reports.
The regular suite uses half-size building inputs to keep resource use modest;
cross-platform assertions use geometric tolerances instead of JPEG byte equality.

After intentionally changing dependencies, run `uv lock` and commit both
`pyproject.toml` and `uv.lock`. For a reproducible environment, use
`uv sync --locked`. See [uv's project guide](https://docs.astral.sh/uv/guides/projects/).

## Continuous integration

The [CI workflow](.github/workflows/ci.yml) runs on pushes, pull requests and manual
dispatches from the GitHub Actions tab. It uses uv 0.9.17 and checks:

- Locked dependency installation with all extras, the test suite and the installed CLI on Linux
  with Python 3.10, 3.12 and 3.14, and on macOS and Windows with Python 3.12.
- Source distribution and wheel builds after all test jobs pass.
- Installation of the wheel with locked runtime dependencies into a clean
  environment, followed by CLI and computation checks outside the source checkout.
- Installation of web extras and verification of packaged templates/static assets.
- The original-size web sample and a performance sample on Linux/Python 3.12.

Successful runs provide the wheel and source archive as a `python-distributions`
artifact, available for 14 days from the run page. Regression previews and JSON
reports are saved in separate artifacts for each test environment. Benchmark
results are recorded without a timing threshold because shared runners vary.
The workflow does not publish
to PyPI or deploy a service.

To reproduce the primary checks locally:

```sh
uv sync --locked --all-extras
uv run --locked --all-extras python -m unittest discover -s tests -v
uv run --locked --all-extras image-stitching --help
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
