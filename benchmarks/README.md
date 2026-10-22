# Recorded performance measurements

These reports use the checked-in building pair, Python 3.12.12, OpenCV 4.14.0,
NumPy 2.5.4 and macOS/arm64. Full environment details and input SHA-256 hashes are
in the JSON files. Matching visualization is disabled.

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

## Current multi-size baseline

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
change. CI saves one half-size observation to its regression artifact without
enforcing timing or memory thresholds on shared runners.
