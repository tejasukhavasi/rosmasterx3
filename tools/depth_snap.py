#!/usr/bin/env python3
"""Runs ON THE ROBOT: save matching colour and depth pictures, taken together.

Used to work out how a spot in the colour picture maps onto the depth
picture. Needs PYTHONPATH=~/pyorbbecsdk/install/lib. Writes
/tmp/depth_snap/<time>-color.jpg and <time>-depth.png (16-bit, millimetres).
"""
import os
import sys
import time
import traceback

import cv2
import numpy as np
import pyorbbecsdk as ob

OUT = "/tmp/depth_snap"
os.makedirs(OUT, exist_ok=True)
try:
    ob.Context().set_logger_level(ob.OBLogLevel.ERROR)
except Exception:
    pass

try:
    cam = cv2.VideoCapture("/dev/camera_depth")          # the colour camera (see video.rules)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("## colour camera opened:", cam.isOpened(), flush=True)

    pipeline = ob.Pipeline()
    profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    config = ob.Config()
    config.enable_stream(profiles.get_video_stream_profile(640, 480, ob.OBFormat.Y11, 30))
    pipeline.start(config)

    depth, t0, n_depth, n_color, color = None, time.time(), 0, 0, None
    while time.time() - t0 < 3.0:                        # let exposure and depth settle
        ok, c = cam.read()
        if ok:
            color, n_color = c, n_color + 1
        frames = pipeline.wait_for_frames(100)
        d = frames.get_depth_frame() if frames is not None else None
        if d is not None:
            raw = np.frombuffer(d.get_data(), dtype=np.uint16).reshape((d.get_height(), d.get_width()))
            if np.count_nonzero(raw) > 0.02 * raw.size:
                depth, n_depth = (raw.astype(np.float32) * d.get_depth_scale()).astype(np.uint16), n_depth + 1
    pipeline.stop()
    cam.release()
    print("## in 3 s: %d colour frames, %d good depth frames" % (n_color, n_depth), flush=True)
    if color is None or depth is None:
        print("## RESULT: failed (colour %s, depth %s)" % (color is not None, depth is not None))
        sys.exit(1)
    stamp = time.strftime("%H%M%S")
    cv2.imwrite("%s/%s-color.jpg" % (OUT, stamp), color, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite("%s/%s-depth.png" % (OUT, stamp), depth)
    v = depth[depth > 0]
    print("## depth %dx%d, %.0f%% valid, %.2f-%.2f m; colour %dx%d"
          % (depth.shape[1], depth.shape[0], 100.0 * v.size / depth.size, v.min() / 1000.0, v.max() / 1000.0,
             color.shape[1], color.shape[0]))
    print("## RESULT: saved pair", stamp)
except Exception:
    traceback.print_exc()
    print("## RESULT: failed")
    sys.exit(1)
