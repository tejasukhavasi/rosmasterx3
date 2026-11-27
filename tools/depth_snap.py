#!/usr/bin/env python3
"""Runs ON THE ROBOT: save matching colour and depth pictures, taken together.

Used to work out how a spot in the colour picture maps onto the depth
picture. Needs PYTHONPATH=~/pyorbbecsdk/install/lib. Writes
/tmp/depth_snap/<time>-color.jpg and <time>-depth.png (16-bit, millimetres).

It tries, in order:
  A. colour AND depth through Orbbec's SDK, with the SDK lining depth up to colour
  B. the same without lining up
  C. depth through the SDK, colour through OpenCV (how follow_bot.py reads it)
"""
import glob
import os
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import pyorbbecsdk as ob

OUT = "/tmp/depth_snap"
os.makedirs(OUT, exist_ok=True)


def say(*a):
    print("##", *a, flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return "(%s)" % e


def first_line(e):
    return (str(e).strip().splitlines() or [repr(e)])[0][:160]


def color_to_bgr(frame):
    w, h, fmt = frame.get_width(), frame.get_height(), frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == ob.OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == ob.OBFormat.RGB:
        return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
    if fmt == getattr(ob.OBFormat, "BGR", None):
        return data.reshape((h, w, 3)).copy()
    if fmt in (getattr(ob.OBFormat, "YUYV", None), getattr(ob.OBFormat, "YUY2", None)):
        return cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_YUY2)
    say("colour format %s not handled" % fmt)
    return None


def depth_to_mm(frame):
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((frame.get_height(), frame.get_width()))
    if np.count_nonzero(raw) < 0.02 * raw.size:
        return None
    return (raw.astype(np.float32) * frame.get_depth_scale()).astype(np.uint16)


def sdk_both(pipeline, depth_profile, color_profile, align):
    config = ob.Config()
    config.enable_stream(depth_profile)
    config.enable_stream(color_profile)
    if align is not None:
        config.set_align_mode(align)
    pipeline.start(config)
    color = depth = None
    nc = nd = 0
    t0 = time.time()
    try:
        while time.time() - t0 < 3.0:
            frames = pipeline.wait_for_frames(200)
            if frames is None:
                continue
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if c is not None:
                img = color_to_bgr(c)
                if img is not None:
                    color, nc = img, nc + 1
            if d is not None:
                mm = depth_to_mm(d)
                if mm is not None:
                    depth, nd = mm, nd + 1
    finally:
        pipeline.stop()
    say("   %d colour frames, %d good depth frames" % (nc, nd))
    return color, depth


def save(color, depth, tag):
    stamp = time.strftime("%H%M%S")
    cv2.imwrite("%s/%s-%s-color.jpg" % (OUT, stamp, tag), color, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite("%s/%s-%s-depth.png" % (OUT, stamp, tag), depth)
    v = depth[depth > 0]
    say("depth %dx%d, %.0f%% valid, %.2f-%.2f m; colour %dx%d"
        % (depth.shape[1], depth.shape[0], 100.0 * v.size / depth.size, v.min() / 1000.0, v.max() / 1000.0,
           color.shape[1], color.shape[0]))
    say("RESULT: saved pair %s (%s)" % (stamp, tag))


try:
    say("video devices:", sh("ls -l /dev/camera_depth /dev/camera_usb /dev/video* 2>&1 | awk '{print $NF, $(NF-1), $(NF-2)}' | tr '\\n' ';'"))
    say("colour camera driver:", sh("for d in /sys/bus/usb/devices/*; do [ \"$(cat $d/idProduct 2>/dev/null)\" = 050f ] && "
                                    "for i in $d:*; do echo -n \"$(basename $i)=$(basename $(readlink $i/driver 2>/dev/null) 2>/dev/null) \"; done; done"))
    try:
        ob.Context().set_logger_level(ob.OBLogLevel.ERROR)
    except Exception:
        pass

    pipeline = ob.Pipeline()
    depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR) \
        .get_video_stream_profile(640, 480, ob.OBFormat.Y11, 30)

    color_profile = None
    try:
        plist = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        say("SDK colour profiles: %d" % plist.get_count())
        for i in range(min(plist.get_count(), 40)):
            p = plist.get_stream_profile_by_index(i).as_video_stream_profile()
            if (p.get_width(), p.get_height()) == (640, 480):
                say("   640x480 @%d %s" % (p.get_fps(), p.get_format()))
        for fmt in ("MJPG", "RGB", "YUYV"):
            try:
                color_profile = plist.get_video_stream_profile(640, 480, getattr(ob.OBFormat, fmt), 30)
                say("using SDK colour 640x480 %s" % fmt)
                break
            except Exception:
                pass
    except Exception as e:
        say("SDK offers no colour stream:", first_line(e))

    if color_profile is not None:
        for tag, align in (("sdk-aligned", getattr(ob.OBAlignMode, "SW_MODE", None)), ("sdk", None)):
            if tag == "sdk-aligned" and align is None:
                continue
            try:
                say("trying %s" % tag)
                color, depth = sdk_both(pipeline, depth_profile, color_profile, align)
                if color is not None and depth is not None:
                    save(color, depth, tag)
                    # Second pair with the depth "soft filter" off. The filter removes
                    # small isolated patches, which may include a small ball.
                    prop = getattr(ob.OBPropertyID, "OB_PROP_DEPTH_SOFT_FILTER_BOOL", None)
                    try:
                        dev = pipeline.get_device()
                        dev.set_bool_property(prop, False)
                        say("soft filter now:", dev.get_bool_property(prop))
                        time.sleep(0.3)
                        color, depth = sdk_both(pipeline, depth_profile, color_profile, align)
                        if color is not None and depth is not None:
                            save(color, depth, tag + "-nofilter")
                        dev.set_bool_property(prop, True)
                    except Exception as e:
                        say("   no-filter attempt failed: %s" % first_line(e))
                    sys.exit(0)
            except SystemExit:
                raise
            except Exception as e:
                say("   %s failed: %s" % (tag, first_line(e)))
            time.sleep(0.5)

    say("trying depth via SDK + colour via OpenCV")
    cam = None
    for dev in ["/dev/camera_depth"] + sorted(glob.glob("/dev/video[0-9]"))[:6]:
        c = cv2.VideoCapture(dev)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok = c.isOpened() and c.read()[0]
        say("   %s: %s" % (dev, "works" if ok else "no"))
        if ok:
            cam = c
            break
        c.release()
    if cam is None:
        say("RESULT: failed, no colour picture by any route")
        sys.exit(1)
    config = ob.Config()
    config.enable_stream(depth_profile)
    pipeline.start(config)
    color = depth = None
    t0 = time.time()
    while time.time() - t0 < 3.0:
        ok, c = cam.read()
        if ok:
            color = c
        frames = pipeline.wait_for_frames(100)
        d = frames.get_depth_frame() if frames is not None else None
        if d is not None:
            mm = depth_to_mm(d)
            if mm is not None:
                depth = mm
    pipeline.stop()
    cam.release()
    if color is None or depth is None:
        say("RESULT: failed (colour %s, depth %s)" % (color is not None, depth is not None))
        sys.exit(1)
    save(color, depth, "opencv")
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    say("RESULT: failed")
    sys.exit(1)
