# Vercel deployment assessment

Checked on 2026-10-23. The bilingual UI runs locally; this change does not create
a Vercel project or provision cloud services.

Vercel supports Flask, but the current local application is not ready to be
deployed unchanged. Serving a page successfully would not prove that original-size
stitching and subsequent downloads work.

## Current workload and platform limits

| Area | This project | Vercel implication |
| --- | --- | --- |
| Uploads | The local limit is 32 MiB; the sample pair totals 3,641,533 bytes | Function request bodies are limited to 4.5 MB. The sample fits, but the advertised upload limit does not. |
| Downloads | The original-size sample PNG is 15,107,570 bytes | The response exceeds the 4.5 MB function payload limit. Store large results outside the function and download directly. |
| Memory | After optimization, macOS measurements reached about 2.30 GiB for core and 2.40 GiB for repeated web requests. Linux CI measured about 1.88 GiB and 1.90 GiB respectively. | Hobby functions have 2 GB; Pro/Enterprise allow up to 4 GB. Linux results are close to the Hobby budget. Actual Vercel testing with larger allowed inputs is needed before deciding whether a paid compute tier is necessary. |
| Runtime | Median compute time was about 0.94 seconds in the fresh-process comparison | Time is not the observed bottleneck locally. Fluid Compute allows 300 seconds on Hobby; this is not a deployed-runtime benchmark. |
| Result state | `create_app()` retains the latest encoded result in one Python process | A download can reach a different instance or run after state is discarded. Use per-job shared storage. |
| Static assets | Flask currently serves packaged CSS/JS | Vercel's Flask integration expects static files under `public/`. |

Limits are from Vercel's [function limits](https://vercel.com/docs/functions/limitations)
and [Flask deployment guide](https://vercel.com/docs/frameworks/backend/flask).
The result-state issue follows from this repository's process-local store and
Vercel's [function scaling model](https://vercel.com/docs/functions).

Measurements use macOS/arm64 and the locked dependencies, including native OpenCV
allocations. RSS was sampled every 10 ms; short peaks can be missed. The core
benchmark disables matching visualization; the separate web benchmark enables
it and exercises uploads, encoding, retained results and downloads. Linux
CPU/threading and allocators will differ. This indicates a resource risk, not a
measured Vercel out-of-memory failure. Before/after measurements and exact-output
checks are in [memory-optimization.json](../benchmarks/memory-optimization.json),
with reproduction commands in [benchmarks/README.md](../benchmarks/README.md).
The earlier single-run 2.50 GiB observation is retained in
[vercel-original-size.json](../benchmarks/vercel-original-size.json).
The follow-up Linux CI observation is recorded in the comparison report's
`linux_ci` section. It covers the approximately 8-megapixel sample, not the web
interface's full 12-megapixel allowance or concurrent hosted requests.

Reproduce the compute measurement:

```sh
uv run --locked --extra benchmark image-stitching-benchmark \
  images/q11.jpg images/q22.jpg --scales 1 --repeats 3 \
  --output output/vercel-assessment.json
```

## Recommended deployment shape

Host a static version of the bilingual frontend on Vercel. Run the existing
Python computation API in a separate container with measured memory capacity.
Upload originals to object storage, submit their object IDs to the compute
service, and return a job ID plus result URLs. The browser should fetch the PNG
directly from storage, so the large image does not pass through a Vercel Function.
Vercel documents [direct client uploads](https://vercel.com/kb/guide/how-to-bypass-vercel-body-size-limit-serverless-functions)
as a way to avoid its function body limit.

This requires exporting the Jinja page to static HTML, configuring the API origin,
adding per-job storage and status, and replacing the single-process lock/result
cache with deployment-wide job handling. The UI's local 32 MiB limit must agree
with the chosen upload path. User photos should have private, expiring download
URLs. None of these changes require changing SIFT, matching, homography or blending.

Keeping the computation on Vercel is another option after adding object storage:
test the Hobby Python runtime first with explicit input and concurrency limits,
move assets into `public/`,
provide a Flask entrypoint, install the web extra, and validate the Linux bundle
and full-resolution request lifecycle. A larger paid tier is an option if actual
measurements require it, not an established prerequisite. The storage/job changes
are still required; adding only `vercel.json` would not solve the blockers.

Do not shrink the inputs silently to fit a deployment budget. The fixed blending
window and feature matching make scaled inputs produce different panoramas.
