"""Local browser interface; all processing and result storage stay on this machine."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np

from .core import StitchingError, new_report, stitch
from .io import encode_image, read_image


def create_app():
    try:
        from flask import Flask, abort, jsonify, render_template, request, send_file
    except ImportError as exc:
        raise ValueError('Install the local web interface with: uv sync --extra web') from exc
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=32 * 1024 * 1024, MAX_IMAGE_PIXELS=12_000_000)
    processing = Lock()
    app.extensions['stitching_processing'] = processing
    app.extensions['stitching_result'] = None
    samples = Path(__file__).resolve().parents[1] / 'images'
    sample_available = all((samples / name).is_file() for name in ('q11.jpg', 'q22.jpg'))

    def error_response(message, code='invalid_request', status=422, report=None):
        if report is None:
            report = new_report()
            report.update(status='error', error={'code': code, 'stage': 'input', 'message': message})
        return jsonify(report=report, error=message), status

    @app.before_request
    def same_origin():
        if request.method == 'POST':
            origin = request.headers.get('Origin')
            if origin and origin != request.host_url.rstrip('/'):
                return error_response('Please submit images from this local page.', 'invalid_origin', 403)

    @app.after_request
    def no_result_cache(response):
        if request.path.startswith(('/api/', '/results/')):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(413)
    def too_large(_error):
        return error_response('The combined upload must be smaller than 32 MiB.', 'upload_too_large', 413)

    @app.get('/')
    def index():
        return render_template('index.html', sample_available=sample_available)

    def process_images(sample=False):
        if not processing.acquire(blocking=False):
            return error_response('Another stitch is running. Please wait and retry.', 'busy', 409)
        try:
            started = perf_counter()
            if sample:
                if not sample_available:
                    return error_response('Sample images are available in the repository checkout.', 'sample_unavailable', 404)
                images = [read_image(samples / name) for name in ('q11.jpg', 'q22.jpg')]
                names = ['q11.jpg · 原尺寸', 'q22.jpg · 原尺寸']
                if request.form.get('reverse') == 'true':
                    images.reverse()
                    names.reverse()
            else:
                images, names = [], []
                for field in ('image1', 'image2'):
                    uploaded = request.files.get(field)
                    if uploaded is None or not uploaded.filename:
                        return error_response('Please choose both a left and a right image.')
                    data = uploaded.read()
                    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
                    if image is None:
                        return error_response(f'Cannot decode {field}. Please use a valid image file.', 'decode_failed')
                    images.append(image)
                    names.append(uploaded.filename)
            for image in images:
                if image.shape[0] * image.shape[1] > app.config['MAX_IMAGE_PIXELS']:
                    return error_response('Each image must contain at most 12 million pixels.', 'image_too_large')
            read_seconds = perf_counter() - started
            result = stitch(*images, include_matching=request.form.get('matching', 'true') != 'false')
            report = result.diagnostics
            report['timings_seconds']['read'] = read_seconds
            report['source'] = {'names': names, 'sample_scale': 1 if sample else None}
            encoded_at = perf_counter()
            files = {'panorama.png': (encode_image('.png', result.panorama), 'image/png')}
            if result.matching_image is not None:
                files['matching.jpg'] = (encode_image('.jpg', result.matching_image), 'image/jpeg')
            for index, image in enumerate(images, 1):
                scale = min(1, 800 / image.shape[1])
                preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                files[f'input{index}.jpg'] = (encode_image('.jpg', preview), 'image/jpeg')
            report['timings_seconds']['encode'] = perf_counter() - encoded_at
            report['timings_seconds']['web_request_total'] = perf_counter() - started
            files['report.json'] = (json.dumps(report, indent=2, allow_nan=False).encode(), 'application/json')
            result_id = uuid4().hex
            # Only the latest successful result is retained, as encoded bytes, not float64 arrays.
            app.extensions['stitching_result'] = (result_id, files)
            return jsonify(report=report, files={name: f'/results/{result_id}/{name}' for name in files})
        except StitchingError as exc:
            return error_response(str(exc), report=exc.diagnostics)
        except (ValueError, OSError, cv2.error) as exc:
            return error_response(str(exc), 'processing_error')
        finally:
            processing.release()

    @app.post('/api/stitch')
    def upload():
        return process_images()

    @app.post('/api/sample')
    def sample():
        return process_images(sample=True)

    @app.get('/results/<result_id>/<filename>')
    def download(result_id, filename):
        latest = app.extensions['stitching_result']
        if latest is None or latest[0] != result_id or filename not in latest[1]:
            abort(404)
        data, mime = latest[1][filename]
        return send_file(BytesIO(data), mimetype=mime, download_name=filename,
                         as_attachment=request.args.get('download') == '1', max_age=0)

    return app


def cli(argv=None):
    parser = argparse.ArgumentParser(description='Open a local image stitching web interface.')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535.')
    try:
        app = create_app()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f'Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C to stop.', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True)
    return 0
