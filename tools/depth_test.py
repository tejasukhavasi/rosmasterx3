#!/usr/bin/env python3
"""Runs ON THE ROBOT: open the Orbbec depth stream and report what comes out.

Needs PYTHONPATH=~/pyorbbecsdk/install/lib. Saves a picture of the depth
image to /tmp/depth.png. The SDK is very chatty; lines from this script
start with "##".
"""
import sys
import time
import traceback

import numpy as np


def say(*a):
    print("##", *a, flush=True)


try:
    import pyorbbecsdk as ob
except Exception:
    traceback.print_exc()
    say("RESULT: import failed")
    sys.exit(1)

try:
    ob.Context().set_logger_level(ob.OBLogLevel.ERROR)      # quieten the SDK
except Exception as e:
    say("(could not quieten SDK log: %s)" % e)


def show_props(device):
    for name in ("OB_PROP_LASER_BOOL", "OB_PROP_LDP_BOOL", "OB_PROP_DEPTH_MIRROR_BOOL",
                 "OB_PROP_DEPTH_SOFT_FILTER_BOOL", "OB_PROP_LASER_CONTROL_INT"):
        prop = getattr(ob.OBPropertyID, name, None)
        if prop is None:
            continue
        try:
            if not device.is_property_supported(prop, ob.OBPermissionType.PERMISSION_READ):
                say("  %s: not supported" % name)
                continue
            val = device.get_int_property(prop) if name.endswith("_INT") else device.get_bool_property(prop)
            say("  %s = %s" % (name, val))
        except Exception as e:
            say("  %s: %s" % (name, str(e).splitlines()[0]))


def try_profile(pipeline, profile, seconds=4.0):
    """Stream one profile; returns the best (most valid pixels) raw uint16 frame."""
    label = "%dx%d@%d %s" % (profile.get_width(), profile.get_height(), profile.get_fps(), profile.get_format())
    config = ob.Config()
    config.enable_stream(profile)
    pipeline.start(config)
    best, best_valid, got, scale, nbytes = None, -1, 0, None, None
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            frames = pipeline.wait_for_frames(200)
            d = frames.get_depth_frame() if frames is not None else None
            if d is None:
                continue
            got += 1
            data = np.frombuffer(d.get_data(), dtype=np.uint8)
            nbytes, scale = data.size, d.get_depth_scale()
            w, h = d.get_width(), d.get_height()
            if data.size != w * h * 2:
                continue
            raw = data.view(np.uint16).reshape((h, w))
            valid = int(np.count_nonzero(raw))
            if valid > best_valid:
                best, best_valid = raw.copy(), valid
    finally:
        pipeline.stop()
    say("%s: %d frames, %s bytes/frame, depth scale %s" % (label, got, nbytes, scale))
    if best is None:
        say("   no usable frames")
        return None, None
    nz = best[best > 0]
    say("   best frame: %.1f%% of pixels have a reading; raw values %s..%s (median %s)"
        % (100.0 * nz.size / best.size, nz.min() if nz.size else 0, nz.max() if nz.size else 0,
           int(np.median(nz)) if nz.size else 0))
    return best, scale


try:
    pipeline = ob.Pipeline()
    device = pipeline.get_device()
    info = device.get_device_info()
    say("device:", info.get_name(), "firmware", info.get_firmware_version())
    say("settings before streaming:")
    show_props(device)

    profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    tried, result = [], (None, None)
    for fmt in ("Y11", "Y12"):
        try:
            p = profiles.get_video_stream_profile(640, 480, getattr(ob.OBFormat, fmt), 30)
        except Exception as e:
            say("no 640x480 %s profile: %s" % (fmt, str(e).splitlines()[0]))
            continue
        raw, scale = try_profile(pipeline, p)
        tried.append(fmt)
        if raw is not None and np.count_nonzero(raw) > 0.02 * raw.size:
            result = (raw, scale)
            break
        time.sleep(0.5)

    raw, scale = result
    if raw is None:
        # Laser may be off: try switching it on and streaming once more.
        prop = getattr(ob.OBPropertyID, "OB_PROP_LASER_BOOL", None)
        try:
            if prop is not None and device.is_property_supported(prop, ob.OBPermissionType.PERMISSION_WRITE):
                say("no depth readings; switching the laser on and trying again")
                device.set_bool_property(prop, True)
                time.sleep(0.5)
                raw, scale = try_profile(pipeline, profiles.get_default_video_stream_profile())
                if raw is not None and np.count_nonzero(raw) <= 0.02 * raw.size:
                    raw = None
        except Exception as e:
            say("could not switch the laser: %s" % str(e).splitlines()[0])
        say("settings after streaming:")
        show_props(device)

    if raw is None:
        say("RESULT: the stream runs (%s tried) but every pixel reads zero" % ", ".join(tried))
        sys.exit(1)

    mm = raw.astype(np.float32) * (scale if scale else 1.0)
    valid = mm[mm > 0]
    h, w = mm.shape
    c = mm[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
    c = c[c > 0]
    say("distances: %.2f-%.2f m, centre of picture %s"
        % (valid.min() / 1000, valid.max() / 1000, ("%.2f m" % (np.median(c) / 1000)) if c.size else "no reading"))
    try:
        import cv2
        img = np.clip(mm / 4000.0 * 255, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(255 - img, cv2.COLORMAP_JET)
        img[mm == 0] = 0
        cv2.imwrite("/tmp/depth.png", img)
        say("saved /tmp/depth.png (near = red, far = blue, black = no reading)")
    except Exception as e:
        say("(could not save picture: %s)" % e)
    say("RESULT: depth works")
except Exception:
    traceback.print_exc()
    say("RESULT: failed")
    sys.exit(1)
