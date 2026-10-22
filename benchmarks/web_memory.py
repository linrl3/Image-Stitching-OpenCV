"""Measure original-size web uploads/downloads, including repeated result retention."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import cv2
    from stitching import web

    if args.reference_core:
        # Compare a trusted historical core exported from this repository.
        spec = importlib.util.spec_from_file_location('reference_core', args.reference_core)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        web.stitch = reference.stitch
    client = web.create_app().test_client()
    runs = []
    for index in range(args.requests):
        cv2.setRNGSeed(42)
        with (ROOT / 'images/q11.jpg').open('rb') as left, (ROOT / 'images/q22.jpg').open('rb') as right:
            response = client.post('/api/stitch', data={
                'image1': (left, 'q11.jpg'), 'image2': (right, 'q22.jpg'), 'matching': 'true',
            })
        try:
            if response.status_code != 200:
                raise RuntimeError(f'Upload failed: {response.get_json()}')
            payload = response.get_json()
            files = {}
            for name, url in payload['files'].items():
                with client.get(url) as download:
                    if download.status_code != 200:
                        raise RuntimeError(f'Download failed: {name}')
                    files[name] = {
                        'bytes': len(download.data),
                        'sha256': hashlib.sha256(download.data).hexdigest(),
                    }
            runs.append({'request': index + 1, 'report': payload['report'], 'files': files})
        finally:
            response.request.input_stream.close()
            response.request.close()
            response.close()
    args.output.write_text(json.dumps({'runs': runs}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('output/web-memory.json'))
    parser.add_argument('--reference-core', type=Path, help='trusted historical core.py for comparison')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error('requests must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return

    import psutil
    with tempfile.TemporaryDirectory(prefix='stitching-web-memory-') as directory:
        destination = Path(directory) / 'worker.json'
        command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                   '--requests', str(args.requests), '--output', str(destination)]
        if args.reference_core:
            command.extend(['--reference-core', str(args.reference_core.resolve())])
        started = perf_counter()
        process = subprocess.Popen(command)
        monitored = psutil.Process(process.pid)
        peak = 0
        try:
            while process.poll() is None:
                try:
                    peak = max(peak, monitored.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                if perf_counter() - started > 300:
                    raise TimeoutError('Web benchmark exceeded 300 seconds')
                sleep(.01)
            if process.wait() != 0:
                raise RuntimeError('Web benchmark worker failed')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        report = json.loads(destination.read_text())
    report.update(schema_version=1, sampled_peak_rss_bytes=peak,
                  process_wall_seconds=perf_counter() - started, measurement={
                      'memory': 'Fresh worker RSS sampled every 10 ms; short peaks may be missed.',
                      'workload': 'Sequential original-size multipart uploads via Flask test client, '
                                  'matching enabled, fetching every result after each request.',
                      'requests': args.requests,
                  })
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Peak RSS: {peak / 1024**3:.3f} GiB; saved to {args.output}')


if __name__ == '__main__':
    main()
