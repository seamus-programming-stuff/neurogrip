"""Calibrate the upright sensor image, then teach finger and target foregrounds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from app import load_camera, load_config
from capture import NeonCapture
from vision_tracking import RegionTracker

ROOT = Path(__file__).resolve().parent


def save_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(str(temporary), str(path))


def revoke_validation(config):
    path = Path(config['strike_config'])
    values = json.loads(path.read_text(encoding='utf-8'))
    values.update(measurement_validated=False, capture_timing_validated=False,
                  stroke_geometry_validated=False)
    save_json(path, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--square-mm', type=float, help='Measured checkerboard square edge in mm')
    parser.add_argument('--board', default='9x6', help='INNER corners, columns x rows')
    parser.add_argument('--output', default=None)
    parser.add_argument('--teach-only', action='store_true')
    parser.add_argument('--no-teach', action='store_true')
    parser.add_argument('--no-run', action='store_true', help='Save calibration without starting the depth feed')
    args = parser.parse_args()
    if os.name == 'posix' and not os.environ.get('DISPLAY'):
        parser.error('Calibration needs a NEON monitor/desktop or SSH X forwarding (DISPLAY). Depth feed runs headless via run.sh.')
    config = load_config(args.config)
    output = Path(args.output or config['camera_calibration']).resolve()
    if output != Path(config['camera_calibration']):
        parser.error('--output must match camera_calibration in the selected config.json')
    try:
        columns, rows = map(int, args.board.lower().split('x'))
        if min(columns, rows) < 3 or max(columns, rows) > 30:
            raise ValueError()
    except ValueError:
        parser.error('Use inner-corner dimensions like 9x6')
    if not args.teach_only and args.square_mm is None:
        try:
            args.square_mm = float(input('Measured checkerboard square edge (mm): '))
        except (ValueError, EOFError):
            parser.error('Provide --square-mm with the measured square edge')
    if not args.teach_only and (not np.isfinite(args.square_mm) or args.square_mm <= 0):
        parser.error('--square-mm must be positive and finite')
    if args.teach_only and not output.exists():
        parser.error('--teach-only requires a camera calibration; run without it first')
    camera = NeonCapture(config['device']).start()
    try:
        if not args.teach_only:
            world = np.zeros((columns*rows, 3), np.float32)
            world[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1,2) * args.square_mm
            objects, images = [], []
            last_sequence = -1
            print('Hold the flat board still. SPACE saves a view; C calibrates >=15 views; Q cancels.', flush=True)
            deadline = time.monotonic()+10
            while True:
                frame = camera.read()
                if frame is None:
                    if time.monotonic() > deadline:
                        raise RuntimeError('No camera frame received within 10 seconds')
                    time.sleep(.02)
                    continue
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCornersSB(gray, (columns,rows))
                preview = frame.image.copy()
                if found:
                    cv2.drawChessboardCorners(preview, (columns,rows), corners, found)
                cv2.putText(preview, 'Views %d | SPACE save | C finish | Q cancel' % len(images),
                            (8,28), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
                cv2.imshow('NEON calibration', preview)
                key = cv2.waitKey(10)&255
                if key == ord('q'):
                    return 1
                if key == 32 and found and frame.sequence != last_sequence:
                    images.append(corners.astype(np.float32))
                    objects.append(world.copy())
                    last_sequence = frame.sequence
                    print('Saved view %d; vary position, tilt and distance.' % len(images), flush=True)
                if key == ord('c'):
                    if len(images) < 15:
                        print('Collect at least 15 varied views, including all image corners.', flush=True)
                        continue
                    rms, k, distortion, _, _ = cv2.calibrateCamera(objects,images,(540,960),None,None)
                    if not np.isfinite(rms) or rms > 2.0:
                        print('RMS %.3fpx exceeds 2px. Collect clearer, varied views and retry.' % rms, flush=True)
                        continue
                    save_json(output, {'schema_version':1,'image_size':[540,960], 'K':k.tolist(),
                        'distortion':distortion.ravel().tolist(),'rms_px':float(rms),
                        'square_mm':args.square_mm,'board_inner_corners':[columns,rows],
                        'views':len(images),'timestamp_kind':'host_read_completion'})
                    revoke_validation(config)
                    print('Saved %s; RMS %.3fpx. This calibrates image geometry, not learned depth accuracy.' % (output,rms), flush=True)
                    break
            cv2.destroyAllWindows()
        if not args.no_teach:
            k, maps = load_camera(output)
            print('Place finger and target in view. SPACE freezes the reference; Q cancels.', flush=True)
            while True:
                frame = camera.read()
                if frame is None:
                    time.sleep(.01)
                    continue
                reference = cv2.remap(frame.image,maps[0],maps[1],cv2.INTER_LINEAR)
                cv2.imshow('NEON teaching reference: SPACE freeze',reference)
                key = cv2.waitKey(10)&255
                if key == ord('q'):
                    return 1
                if key == 32:
                    break
            regions = {}
            for name in ('finger','target'):
                while True:
                    print('Draw a box around %s with a small background margin; ENTER confirms.' % name, flush=True)
                    box = tuple(int(v) for v in cv2.selectROI('Teach '+name, reference, False, False))
                    cv2.destroyWindow('Teach '+name)
                    if box[2] == 0 or box[3] == 0:
                        return 1
                    result = RegionTracker().initialize(reference,box)
                    if result.valid:
                        regions[name] = list(box)
                        break
                    print('Try again: %s. Add visible texture if needed.' % result.reason, flush=True)
            x,y,w,h = regions['finger']
            a,b,c,d = regions['target']
            if max(x,a) < min(x+w,a+c) and max(y,b) < min(y+h,b+d):
                raise RuntimeError('Finger and target boxes overlap. Rerun with separate visible regions.')
            teaching_path = Path(config['teaching'])
            reference_path = teaching_path.with_name('teaching_reference.png')
            if not cv2.imwrite(str(reference_path),reference):
                raise RuntimeError('Could not write teaching reference')
            regions.update(image_size=[540,960],reference=reference_path.name,
                           camera_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            save_json(teaching_path,regions)
            revoke_validation(config)
            print('Saved tracking reference. Physical validation remains disabled until measured separately.', flush=True)
    finally:
        camera.close()
        cv2.destroyAllWindows()
    if not args.no_run:
        os.execv(sys.executable, [sys.executable, '-u', str(ROOT / 'app.py'),
                                 '--config', str(Path(args.config).resolve())])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
