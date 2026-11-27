#!/usr/bin/env python3
# coding=utf-8
"""
Colour and depth from the robot's Orbbec Astra Pro Plus through Orbbec's SDK.

Why not OpenCV for the colour picture, as rosmaster_main.py does: once the
SDK has touched the camera it takes the colour stream away from the normal
webcam driver (/dev/video0 disappears until the camera is re-plugged). So
when the SDK is in use, everything has to come through it.

The depth picture is lined up with the colour picture by the SDK, so pixel
(x, y) means the same spot in both. Depth is in millimetres, 0 = no reading.
The camera cannot measure anything closer than about 0.6 m, shiny floors, or
very small objects.

Same interface as Rosmaster_Camera (isOpened / get_frame / reconnect), plus depth().
The SDK is expected in ~/pyorbbecsdk/install/lib (see tools/depth_setup.sh).
"""
import os
import sys
import threading
import time

import cv2 as cv
import numpy as np

SDK_DIR = os.path.expanduser("~/pyorbbecsdk/install/lib")


def sdk_available():
    return os.path.isdir(SDK_DIR) and any(n.startswith("pyorbbecsdk") for n in os.listdir(SDK_DIR))


class OrbbecCamera(object):
    def __init__(self, width=640, height=480, fps=30, with_depth=True, debug=False):
        if SDK_DIR not in sys.path:
            sys.path.insert(0, SDK_DIR)
        import pyorbbecsdk as ob
        self.ob = ob
        self.size = (width, height, fps)
        self.with_depth = with_depth
        self.debug = debug
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self._color = None
        self._depth = None
        self._depth_t = 0.0
        self._seq = 0
        self._taken = 0
        self._run = False
        self._thread = None
        self.pipeline = None
        self.aligned = False
        self.error = None
        if not debug:
            try:
                ob.Context().set_logger_level(ob.OBLogLevel.ERROR)
            except Exception:
                pass
        self._start()

    # ---- start / stop ----
    def _start(self):
        ob = self.ob
        w, h, fps = self.size
        try:
            self.pipeline = ob.Pipeline()
            color = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR) \
                .get_video_stream_profile(w, h, ob.OBFormat.MJPG, fps)
            attempts = [(False, False)]
            if self.with_depth:
                attempts = [(True, True), (True, False), (False, False)]   # (depth, aligned)
            for use_depth, align in attempts:
                try:
                    config = ob.Config()
                    config.enable_stream(color)
                    if use_depth:
                        depth = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR) \
                            .get_video_stream_profile(w, h, ob.OBFormat.Y11, fps)
                        config.enable_stream(depth)
                        if align:
                            config.set_align_mode(ob.OBAlignMode.SW_MODE)
                    self.pipeline.start(config)
                    self.with_depth, self.aligned = use_depth, align
                    break
                except Exception as e:
                    self.error = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
            else:
                return False
        except Exception as e:
            self.error = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
            return False
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def close(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None
        try:
            if self.pipeline:
                self.pipeline.stop()
        except Exception:
            pass
        self.pipeline = None

    def _loop(self):
        ob = self.ob
        while self._run:
            try:
                frames = self.pipeline.wait_for_frames(200)
            except Exception as e:
                self.error = str(e)
                time.sleep(0.1)
                continue
            if frames is None:
                continue
            color = depth = None
            c = frames.get_color_frame()
            if c is not None:
                color = cv.imdecode(np.frombuffer(c.get_data(), dtype=np.uint8), cv.IMREAD_COLOR)
            if self.with_depth:
                d = frames.get_depth_frame()
                if d is not None:
                    raw = np.frombuffer(d.get_data(), dtype=np.uint16)
                    w, h = d.get_width(), d.get_height()
                    if raw.size == w * h:
                        scale = d.get_depth_scale()
                        depth = raw.reshape((h, w))
                        depth = depth.copy() if scale == 1.0 else (depth.astype(np.float32) * scale).astype(np.uint16)
            with self.cond:
                if depth is not None:
                    self._depth, self._depth_t = depth, time.time()
                if color is not None:
                    self._color = color
                    self._seq += 1
                    self.cond.notify_all()

    # ---- Rosmaster_Camera-compatible interface ----
    def isOpened(self):
        return self._run

    def get_frame(self, timeout=0.6):
        """Wait for the next NEW colour frame. Returns (ok, bgr image)."""
        with self.cond:
            if self._seq == self._taken:
                self.cond.wait(timeout)
            if self._seq == self._taken or self._color is None:
                return False, None
            self._taken = self._seq
            return True, self._color

    def reconnect(self):
        self.close()
        time.sleep(0.5)
        return self._start()

    # ---- depth ----
    def depth(self, max_age=0.5):
        """Latest depth picture in millimetres (0 = no reading), or None."""
        with self.lock:
            if self._depth is None or time.time() - self._depth_t > max_age:
                return None
            return self._depth


def depth_ahead(depth, band=(0.30, 0.68), lane=(0.35, 0.65)):
    """Distance in metres to the nearest thing in the middle of the depth picture.

    Looks at a window ahead of the robot (fractions of height and width) and
    returns the 3rd-percentile distance of the pixels that have a reading, so
    a few stray pixels don't count. None if hardly anything there has a reading.
    """
    if depth is None:
        return None
    h, w = depth.shape
    win = depth[int(h * band[0]):int(h * band[1]), int(w * lane[0]):int(w * lane[1])]
    v = win[win > 0]
    if v.size < 0.05 * win.size:
        return None
    return float(np.percentile(v, 3)) / 1000.0
