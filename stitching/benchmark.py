"""Repeatable benchmark with fresh workers and sampled process RSS (including OpenCV)."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter, sleep

import cv2

from .core import StitchingError, new_report, stitch
from .io import encode_image, read_image, validate_paths, write_report


def worker(image1, image2, scale, destination):
    report = new_report()
    started = perf_counter()
    try:
        images = [read_image(image1), read_image(image2)]
        read_seconds = perf_counter() - started
        resized_at = perf_counter()
        if scale != 1:
            images = [cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for im in images]
        resize_seconds = perf_counter() - resized_at
        cv2.setRNGSeed(42)
        result = stitch(*images)
        report = result.diagnostics
        encode_at = perf_counter()
        encode_image('.jpg', result.panorama)
        report['timings_seconds'].update(
            read=read_seconds, resize=resize_seconds, encode=perf_counter() - encode_at
        )
    except (ValueError, OSError, cv2.error) as exc:
        report = exc.diagnostics if isinstance(exc, StitchingError) else report
        report['status'] = 'error'
        report.setdefault('error', {'code': 'benchmark_input_error', 'stage': 'input', 'message': str(exc)})
    report['scale'] = scale
    report['timings_seconds']['worker_total'] = perf_counter() - started
    write_report(destination, report)
    return 0 if report['status'] == 'success' else 1


def run_benchmark(image1, image2, scales, repeats):
    try:
        import psutil
    except ImportError as exc:
        raise ValueError('Install benchmark support with: uv sync --extra benchmark') from exc
    runs = []
    with tempfile.TemporaryDirectory(prefix='stitching-benchmark-') as tmp:
        for scale in scales:
            for repetition in range(1, repeats + 1):
                destination = Path(tmp) / 'worker.json'
                destination.unlink(missing_ok=True)
                command = [sys.executable, '-m', 'stitching.benchmark', '--worker',
                           str(Path(image1).resolve()), str(Path(image2).resolve()),
                           '--scales', str(scale), '--output', str(destination)]
                started = perf_counter()
                # A file keeps diagnostic stderr bounded independently of the child pipe buffer.
                with (Path(tmp) / 'stderr.txt').open('w+') as errors:
                    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors)
                    monitored = psutil.Process(process.pid)
                    peak = 0
                    try:
                        while process.poll() is None:
                            try:
                                peak = max(peak, monitored.memory_info().rss)
                            except psutil.NoSuchProcess:
                                break
                            if perf_counter() - started > 300:
                                raise TimeoutError('Benchmark worker exceeded 300 seconds.')
                            sleep(0.01)
                        process.wait()
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                    if destination.exists():
                        report = json.loads(destination.read_text())
                    else:
                        errors.seek(0)
                        raise RuntimeError(f'Benchmark worker failed ({process.returncode}): {errors.read()[-2000:]}')
                report.update(repetition=repetition, sampled_peak_rss_bytes=peak,
                              process_wall_seconds=perf_counter() - started)
                runs.append(report)
    summary = []
    for scale in scales:
        successful = [r for r in runs if r['scale'] == scale and r['status'] == 'success']
        summary.append({
            'scale': scale, 'successful_runs': len(successful),
            'median_compute_seconds': statistics.median(
                r['timings_seconds']['compute_total'] for r in successful) if successful else None,
            'max_sampled_peak_rss_bytes': max(
                (r['sampled_peak_rss_bytes'] for r in successful), default=None),
        })
    return {'schema_version': 1, 'inputs': [
        {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()
         if Path(path).is_file() else None} for path in (image1, image2)
    ], 'measurement': {
        'memory': 'Fresh worker process RSS, sampled every 10 ms; short peaks may be missed.',
        'timing': 'Per-stage wall time; process_wall_seconds also includes interpreter startup.',
        'matching_visualization': False, 'repeats': repeats,
    }, 'summary': summary, 'runs': runs}


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image1', type=Path)
    parser.add_argument('image2', type=Path)
    parser.add_argument('--scales', type=float, nargs='+', default=[0.5],
                        help='explicit input scales to benchmark (default: 0.5)')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('output/benchmark.json'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeats < 1 or any(not 0 < scale <= 1 for scale in args.scales):
        parser.error('repeats must be positive and scales must be between 0 and 1.')
    try:
        validate_paths([args.image1, args.image2], [args.output])
        if args.worker:
            return worker(args.image1, args.image2, args.scales[0], args.output)
        report = run_benchmark(args.image1, args.image2, list(dict.fromkeys(args.scales)), args.repeats)
        write_report(args.output, report)
        print(json.dumps(report['summary'], indent=2))
        print(f'Saved benchmark to {args.output}')
        return 0 if all(r['status'] == 'success' for r in report['runs']) else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'{parser.prog}: error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(cli())
