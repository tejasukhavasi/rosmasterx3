#!/usr/bin/env python3
"""Runs ON THE ROBOT: open the Orbbec depth stream and report what comes out.

Needs PYTHONPATH=~/pyorbbecsdk/install/lib. Saves a picture of the depth
image to /tmp/depth.png.
"""
import sys
import time
import traceback

import numpy as np

try:
    import pyorbbecsdk as ob
    print("pyorbbecsdk imported from", ob.__file__)
except Exception:
    traceback.print_exc()
    print("RESULT: import failed")
    sys.exit(1)

try:
    ctx = ob.Context()
    devs = ctx.query_devices()
    print("devices found:", devs.get_count())
    pipeline = ob.Pipeline()
    try:
        info = pipeline.get_device().get_device_info()
        print("device:", info.get_name(), "pid", hex(info.get_pid()), "firmware", info.get_firmware_version())
    except Exception as e:
        print("(could not read device info: %s)" % e)

    profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    print("depth profiles:", profiles.get_count())
    for i in range(min(profiles.get_count(), 12)):
        try:
            p = profiles.get_stream_profile_by_index(i).as_video_stream_profile()
            print("   %dx%d @%d %s" % (p.get_width(), p.get_height(), p.get_fps(), p.get_format()))
        except Exception as e:
            print("   profile %d: %s" % (i, e))
    profile = profiles.get_default_video_stream_profile()
    print("using default: %dx%d @%d %s" % (profile.get_width(), profile.get_height(), profile.get_fps(), profile.get_format()))

    config = ob.Config()
    config.enable_stream(profile)
    pipeline.start(config)
    got, last, t0 = 0, None, time.time()
    while time.time() - t0 < 5.0:
        frames = pipeline.wait_for_frames(200)
        if frames is None:
            continue
        d = frames.get_depth_frame()
        if d is None:
            continue
        got += 1
        w, h, scale = d.get_width(), d.get_height(), d.get_depth_scale()
        last = np.frombuffer(d.get_data(), dtype=np.uint16).reshape((h, w)).astype(np.float32) * scale   # mm
    pipeline.stop()
    print("frames in 5 s:", got)
    if last is None:
        print("RESULT: stream opened but no depth frames arrived")
        sys.exit(1)
    valid = last[last > 0]
    h, w = last.shape
    c = last[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
    c = c[c > 0]
    print("size %dx%d, valid pixels %.0f%%, range %.2f-%.2f m, centre %.2f m"
          % (w, h, 100.0 * valid.size / last.size, valid.min() / 1000 if valid.size else 0,
             valid.max() / 1000 if valid.size else 0, np.median(c) / 1000 if c.size else 0))
    try:
        import cv2
        img = np.clip(last / 4000.0 * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(255 - img, cv2.COLORMAP_JET)
        img[last == 0] = 0
        cv2.imwrite("/tmp/depth.png", img)
        print("saved /tmp/depth.png (near = red, far = blue, black = no reading)")
    except Exception as e:
        print("(could not save picture: %s)" % e)
    print("RESULT: depth works")
except Exception:
    traceback.print_exc()
    print("RESULT: failed")
    sys.exit(1)
