# Recorded performance measurements

These reports use the checked-in building pair, Python 3.12.12, OpenCV 4.14.0,
NumPy 2.5.4 and macOS/arm64. Full environment details and input SHA-256 hashes are
in the JSON files. Matching visualization is disabled unless noted otherwise.

## Original-size memory optimization

[memory-optimization.json](memory-optimization.json) compares the core at commit
`e1fa54bddd7d58d6745f5a9eee4fcc36e4c765ab` with column weights, block blending and
axis-based crop bounds. Both versions use the original 2448 x 3264 inputs and
float64 arithmetic, with the same dependency versions and OpenCV thread settings.

| Measurement | Before | After |
| --- | ---: | ---: |
| Core maximum sampled RSS, three fresh processes | 3.01 GiB | 2.30 GiB |
| Core median compute time | 3.18 s | 0.94 s |
| Web peak sampled RSS, three requests in one fresh process | 3.11 GiB | 2.40 GiB |

The core measurement uses the existing benchmark CLI with `--scales 1 --repeats 3`.
The web measurement includes multipart uploads, matching visualization, PNG/JPEG
encoding, retaining the latest result, and fetching every download after each
request. It uses Flask's test client, not a network server or Vercel deployment.
RSS includes native OpenCV allocations and is sampled every 10 ms; short peaks
can be missed. Prior single-run observations differ because allocator retention
and scheduling vary. Use paired measurements, not unrelated historical peaks.

A separate full-size comparison produced identical raw float64 pixels, PNG bytes
and matching-image pixels. All three web requests also produced byte-identical
panorama, matching and input-preview downloads. The report includes their hashes.
Regression tests compare the optimized blend with the frozen dense formula for
multiple window sizes, uneven image sizes, block boundaries and perspective warps.

To repeat the web comparison from a repository checkout:

```sh
mkdir -p output
git show e1fa54b:stitching/core.py > output/reference-core.py
uv run --locked --all-extras python benchmarks/web_memory.py \
  --reference-core output/reference-core.py --output output/web-before.json
uv run --locked --all-extras python benchmarks/web_memory.py \
  --output output/web-after.json
```

The reference file is executed as Python; use only trusted code. SIFT still
accounts for substantial native temporary memory. No inputs are resized, no
float precision is reduced, and OpenCV's thread count is unchanged. These local
results do not establish that the workload fits a 2 GB hosted function.

## Before/after comparison

[baseline.json](baseline.json) compares the script at commit
`feaf41a76e04640ab6413dd047fa92893fb3dbdb` with the 0.2 computation API. Each version
ran three times in separate fresh processes, using the same environment, half-size
inputs (`INTER_AREA`) and OpenCV RNG seed 42.

| Measurement | Before | After |
| --- | ---: | ---: |
| Median stitching wall time | 0.551 s | 0.537 s |
| Maximum sampled process RSS | 1.081 GiB | 0.854 GiB |
| Output shape | 1632 × 2096 × 3 | 1632 × 2096 × 3 |

Raw float64 pixels and JPEG-encoded bytes have identical SHA-256 hashes in every
run. Peak sampled RSS is about 21% lower after releasing used masks before the
next allocation. Timing differences are small and should not be treated as an
established speed improvement.

An additional original-size comparison with matching visualization enabled also
produced identical panorama pixels, JPEG bytes and matching-image pixels against
the same reference commit. Its hashes are recorded as `original_size_parity_check`.

RSS was sampled every 10 ms for the complete worker lifecycle, including imports,
decoding, stitching, hashing and encoding. Short peaks can be missed. Wall time
in this comparison covers stitching only. This is a measured process footprint,
not a calculation based only on NumPy buffer sizes.

## Earlier multi-size baseline

[current.json](current.json) was produced with:

```sh
uv run --locked --extra benchmark image-stitching-benchmark \
  images/q11.jpg images/q22.jpg --scales .5 .75 --repeats 3 \
  --output benchmarks/current.json
```

The benchmark samples each fresh worker's full lifecycle, including imports,
reading, resizing, stitching and JPEG encoding. Its memory figure therefore
includes a slightly different set of operations from the comparison harness.
The report separates compute, read, resize, encode and process wall time.

Input scaling is an explicit benchmark setting. It changes SIFT features and the
relative width of the fixed blend, so it can change output appearance. The web
sample always computes from the original-size photos. To measure that workload,
use `--scales 1`; allow several GiB of available memory.

Re-run on the same hardware and dependencies before drawing conclusions about a
change. CI saves half-size and original-size observations to its regression artifact without
enforcing timing or memory thresholds on shared runners.
