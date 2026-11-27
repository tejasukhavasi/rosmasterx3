#!/usr/bin/env python3
# coding=utf-8
"""
Follow Bot for the Yahboom Rosmaster X3.

The robot follows a brightly coloured thing (a ball, a hat, a toy) that it
sees with its camera. It turns to keep the thing in the middle of the picture
and drives forward/back to keep it the same apparent size (= same distance).

Run on the Pi (stop the stock app first, both need the camera and serial port):

    bash rosmaster/kill_rosmaster.sh
    python3 rosmaster/follow_bot.py              # then open http://<pi-ip>:6501
    python3 rosmaster/follow_bot.py --no-motors  # camera/vision only, wheels never move

"Show what matches" on the page paints every pixel the robot counts as the
toy's colour bright green, and http://<pi-ip>:6501/snapshot.jpg is a clean
camera frame for tuning.

Stop with the STOP button on the page or Ctrl+C. Do NOT use `kill -9`:
the motors keep their last command if the program cannot say "stop".
"""
import argparse
import atexit
import logging
import math
import os
import signal
import threading
import time

import cv2 as cv
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
PROC_W, PROC_H = 640, 480     # processing size. Full resolution so small/far toys still show up

# Preset colours, OpenCV HSV (H 0-179). Each is a list of (low, high) ranges.
COLORS = {
    "red":    [((0, 135, 95), (9, 255, 255)), ((170, 135, 95), (179, 255, 255))],
    "orange": [((8, 130, 100), (21, 255, 255))],
    "yellow": [((22, 100, 100), (34, 255, 255))],
    "green":  [((40, 80, 60), (85, 255, 255))],
    "blue":   [((95, 120, 60), (125, 255, 255))],
    # Pink runs from magenta round to coral. Skin, leather and red-brown wood all
    # share this hue; what sets a toy apart is how much more saturated it is
    # (measured on the robot: ball about 140-180, skin and leather about 80-115).
    "pink":   [((140, 120, 95), (179, 255, 255))],
}

# Speed presets: (max forward m/s, max turn rad/s)
SPEEDS = {
    "turtle": (0.15, 1.0),
    "bunny":  (0.28, 1.6),
}

MAX_REVERSE = 0.10      # m/s. Backing up is blind, keep it gentle.
MIN_AREA_PX = 40        # ignore blobs smaller than this many pixels (about 7 px across)
MIN_FILL = 0.40         # blob area / enclosing-circle area. A ball is ~0.8; streaks and
                        # ragged patches of furniture are ~0.1-0.3
MIN_SAT_MARGIN = 15     # a blob's average saturation must beat the colour's floor by this
                        # much; a patch that only just scrapes over the floor is not the toy
MIN_FILL_EDGE = 0.20    # looser for a blob cut off by the edge of the picture
TURN_GAIN = 2.0         # rad/s per unit of horizontal error (-1..1)
DRIVE_GAIN = 0.45       # m/s per unit of relative size error
TURN_DEADBAND = 0.08    # fraction of half-frame
DRIVE_DEADBAND = 0.12   # relative size error
CONFIRM_FRAMES = 3      # a toy must be seen this many frames in a row before it counts,
                        # so a one-frame flicker of something pink-ish can't interrupt a search
LOST_GRACE_S = 0.4      # coast-free pause before deciding the target is gone
# The camera can't see the floor right in front of the robot, so a toy low in
# the picture is about to disappear underneath it.
NEAR_EDGE = 0.90        # toy's lower edge below this fraction of the picture: don't advance
CLIP_EDGE = 0.985       # toy cut off by the bottom of the picture: back away
# When the toy vanishes while close (or out of the top), back up to widen the view.
BACKUP_SPEED = 0.08     # m/s
BACKUP_S = 2.0          # at most this long (about 16 cm)
# Lidar: used automatically when /dev/rplidar is present (--no-lidar turns it off).
LIDAR_FRONT_DEG = 349.0  # lidar angle that points straight ahead (from tools/pi.sh calibrate)
RADAR_RANGE = 3.0        # m shown in the top-down lidar view on the video
# Lidar guard. Distances are measured from the lidar,
# which sits in the middle of the robot, so they include half the robot's length.
LIDAR_STOP = 0.40       # m. Something this close in the robot's lane: don't move that way
LIDAR_SLOW = 0.80       # m. Start slowing down from here
HALF_FOV = 0.55         # rad, roughly half the camera's horizontal field of view
PREDICT_S = 0.6         # how far ahead to extrapolate the toy's sideways motion
# Searching for a lost toy: spin on the spot, pausing the moment something
# toy-coloured shows up so it can be checked without motion blur.
SCAN_TURN = 1.5         # rad/s (about 85 degrees a second), whatever the speed setting
SCAN_SWEEP_S = 5.0      # one sweep = a bit more than a full circle
SCAN_REST_S = 1.0       # short rest; the next sweep goes the other way
SCAN_GIVE_UP_S = 30.0   # then wait quietly until the toy shows up again
# Following by shape (depth camera): the target is whatever blob of similar
# distance sits where the target was a moment ago.
FOLLOW_DIST = 1.2       # m to keep from a person/thing followed by shape
DIST_GAIN = 0.6         # m/s per metre of distance error
DIST_DEADBAND = 0.15    # m
DEPTH_SLAB_MM = 350     # the target's blob = pixels within this of its last distance
DEPTH_GROW = 0.35       # look this much beyond the last box (each side) for the blob
DEPTH_MIN_AREA = 150    # blob pixels at half resolution
DEPTH_MAX_WIDTH_M = 1.6 # a "blob" wider than this is a wall, not a target
FOCAL_PX = 600.0        # camera focal length in pixels at 640 wide (measured from the ball: ~630)
MOTION_MM = 300         # something this much nearer than a moment ago counts as "walked in"
MOTION_MIN_AREA = 500   # half-resolution pixels
LEARN_BOX = 0.12        # centre box (fraction of frame height) for "Learn colour"


# ----------------------------------------------------------------------------
# Vision
# ----------------------------------------------------------------------------
_K_OPEN = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
_K_CLOSE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))


def detect(frame, ranges, prev=None):
    """Find the best blob of the target colour.

    Returns (det, mask). det is (cx, cy, size) in pixels of `frame`, where size
    is the diameter of the blob's enclosing circle, or None if nothing is found.
    `prev` is the previous (cx, cy); blobs near it are preferred so the robot
    sticks with the toy it was already following.
    """
    hsv = cv.cvtColor(cv.GaussianBlur(frame, (5, 5), 0), cv.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        m = cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv.bitwise_or(mask, m)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, _K_OPEN)     # drop single-pixel noise
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, _K_CLOSE)   # fill shiny highlights
    found = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]   # OpenCV 3 vs 4
    h, w = mask.shape
    min_sat = (min(lo[1] for lo, _ in ranges) + MIN_SAT_MARGIN) / 255.0
    best, best_score = None, 0.0
    for c in contours:
        area = cv.contourArea(c)
        if area < MIN_AREA_PX:
            continue
        (x, y), r = cv.minEnclosingCircle(c)
        fill = area / (math.pi * r * r + 1e-6)
        bx, by, bw, bh = cv.boundingRect(c)
        on_edge = bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1
        if fill < (MIN_FILL_EDGE if on_edge else MIN_FILL):
            continue
        # How vivid is it? Toys are far more saturated than look-alike furniture.
        one = np.zeros((bh, bw), np.uint8)
        cv.drawContours(one, [c], -1, 255, -1, offset=(-bx, -by))
        sat = cv.mean(hsv[by:by + bh, bx:bx + bw], mask=cv.bitwise_and(one, mask[by:by + bh, bx:bx + bw]))[1] / 255.0
        if sat < min_sat:
            continue
        score = area * fill * fill * sat * sat
        if prev is not None and math.hypot(x - prev[0], y - prev[1]) < max(80.0, 4 * r):
            score *= 2.0
        if score > best_score:
            best, best_score = (x, y, 2.0 * r), score
    return best, mask


def learn_color(frame, at=None):
    """Sample a small box (the centre, or around the point `at`) and build an HSV
    range around its colour.

    Returns (ranges, message). ranges is None if the sample is too grey/dark.
    Only the colourful pixels in the box are used, so white highlights on a
    shiny ball and a bit of background around a small one don't spoil it.
    """
    h, w = frame.shape[:2]
    r = int(h * LEARN_BOX / 2) if at is None else 12
    cx, cy = (w // 2, h // 2) if at is None else (int(clamp(at[0], r, w - r)), int(clamp(at[1], r, h - r)))
    roi = cv.GaussianBlur(frame, (5, 5), 0)[cy - r:cy + r, cx - r:cx + r]
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    vivid = hsv[(hsv[:, 1] > 60) & (hsv[:, 2] > 50)]
    if len(vivid) < 0.3 * len(hsv):
        return None, "Too dull. Hold something bright right inside the box!"
    # Hue is circular (red wraps 179 -> 0). Histogram it and take the main peak.
    hist = np.bincount(vivid[:, 0].astype(int), minlength=180).astype(np.float32)
    hist = np.convolve(np.concatenate([hist[-4:], hist, hist[:4]]), np.ones(9), "valid")
    h_mid = int(np.argmax(hist))
    dh = np.abs(((vivid[:, 0] - h_mid + 90) % 180) - 90)
    core = vivid[dh <= 8]
    tol = int(clamp(np.percentile(dh[dh <= 8], 90) + 4, 6, 10))
    # Floors relative to the toy itself: a vivid toy gets a strict floor that
    # shuts out dull things of the same hue; a pastel toy gets a looser one.
    s_lo = int(max(60, 0.78 * np.median(core[:, 1])))
    v_lo = int(max(60, 0.45 * np.median(core[:, 2])))
    lo, hi = h_mid - tol, h_mid + tol
    if lo < 0:
        ranges = [((0, s_lo, v_lo), (hi, 255, 255)), ((180 + lo, s_lo, v_lo), (179, 255, 255))]
    elif hi > 179:
        ranges = [((lo, s_lo, v_lo), (179, 255, 255)), ((0, s_lo, v_lo), (hi - 180, 255, 255))]
    else:
        ranges = [((lo, s_lo, v_lo), (hi, 255, 255))]
    return ranges, "Got it!"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ----------------------------------------------------------------------------
# Control
# ----------------------------------------------------------------------------
def compute_cmd(det, frame_w, frame_h, target_size, max_fwd, max_turn):
    """Turn a detection into (vx m/s, vz rad/s). vz > 0 turns left (ROS style)."""
    cx, cy, size = det
    ex = (cx - frame_w / 2.0) / (frame_w / 2.0)        # -1 left .. +1 right
    vz = 0.0 if abs(ex) < TURN_DEADBAND else clamp(-TURN_GAIN * ex, -max_turn, max_turn)

    want = target_size * frame_w
    es = (want - size) / want                           # >0 too far, <0 too close
    vx = 0.0 if abs(es) < DRIVE_DEADBAND else clamp(DRIVE_GAIN * es, -MAX_REVERSE, max_fwd)
    if vx > 0:
        vx *= max(0.0, 1.0 - abs(ex))                   # face it first, then drive

    # A toy sliding out of the bottom of the picture looks SMALLER (it is cut
    # off), which would otherwise read as "far away, drive forward".
    low = (cy + size / 2.0) / frame_h
    if low >= CLIP_EDGE:
        vx = -BACKUP_SPEED
    elif low >= NEAR_EDGE:
        vx = min(vx, 0.0)
    return vx, vz


class DepthTarget(object):
    """Follows one thing through the depth picture.

    The target is the blob of pixels at about the same distance as last time,
    close to where it was last time. Works for anything the depth camera can
    see (people, pets, boxes, big toys), needs no training, and gives the
    distance directly. Internally works at half resolution; everything it
    hands out is in full-resolution pixels and metres.
    """

    def __init__(self):
        self.locked = False
        self.z = 0.0            # mm
        self.box = None         # half-res (x1, y1, x2, y2)
        self.cx = self.cy = 0.0
        self.area = 0
        self.label = "thing"

    def start_at(self, depth, x, y, label="thing"):
        """Lock onto whatever is at pixel (x, y). False if depth has no reading there."""
        d = depth[::2, ::2]
        h, w = d.shape
        x, y = int(clamp(x / 2, 0, w - 1)), int(clamp(y / 2, 0, h - 1))
        win = d[max(0, y - 8):y + 9, max(0, x - 8):x + 9]
        v = win[win > 0]
        if v.size < 20:
            return False
        self.z, self.cx, self.cy = float(np.median(v)), float(x), float(y)
        self.box = (0, 0, w, h)                       # first look: anywhere, nearest to the tap wins
        self.label = label
        self.locked = self._find(d, grow=False)
        return self.locked

    def start_box(self, depth, box, label="person"):
        """Lock onto the nearest thing inside a detector box (full-res pixels)."""
        d = depth[::2, ::2]
        h, w = d.shape
        x1, y1, x2, y2 = [int(v / 2) for v in box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return False
        mx, my = (x2 - x1) // 4, (y2 - y1) // 5
        mid = d[y1 + my:y2 - my, x1 + mx:x2 - mx]
        v = mid[mid > 0]
        if v.size < 0.1 * mid.size:
            return False
        self.z = float(np.percentile(v, 25))          # the near surface, not the wall seen between the legs
        self.box, self.cx, self.cy = (x1, y1, x2, y2), (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self.label = label
        self.locked = self._find(d, grow=False)
        return self.locked

    def update(self, depth):
        """Find the target in a new depth picture. False = not there this frame."""
        return self.locked and self._find(depth[::2, ::2], grow=True)

    def _find(self, d, grow):
        h, w = d.shape
        x1, y1, x2, y2 = self.box
        gw = int((x2 - x1) * DEPTH_GROW) + 6 if grow else 0
        gh = int((y2 - y1) * DEPTH_GROW) + 6 if grow else 0
        rx1, ry1, rx2, ry2 = max(0, x1 - gw), max(0, y1 - gh), min(w, x2 + gw), min(h, y2 + gh)
        roi = d[ry1:ry2, rx1:rx2]
        mask = ((roi > self.z - DEPTH_SLAB_MM) & (roi < self.z + DEPTH_SLAB_MM)).astype(np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, _K_OPEN)
        n, lab, stats, cents = cv.connectedComponentsWithStats(mask, 8)
        best, best_score = 0, None
        for i in range(1, n):
            area = stats[i, cv.CC_STAT_AREA]
            if area < DEPTH_MIN_AREA:
                continue
            dist = abs(cents[i][0] + rx1 - self.cx) + 0.5 * abs(cents[i][1] + ry1 - self.cy)
            score = area - 8.0 * dist                 # big and close to where it was
            if best_score is None or score > best_score:
                best, best_score = i, score
        if not best:
            return False
        bx, by, bw, bh, area = stats[best]
        z = float(np.median(roi[lab == best]))
        if bw * 2 * z / 1000.0 / FOCAL_PX > DEPTH_MAX_WIDTH_M:
            return False                              # slid onto a wall or big furniture
        self.box = (rx1 + bx, ry1 + by, rx1 + bx + bw, ry1 + by + bh)
        self.cx, self.cy, self.area, self.z = cents[best][0] + rx1, cents[best][1] + ry1, int(area), z
        return True

    # full-resolution views
    @property
    def dist(self):
        return self.z / 1000.0

    @property
    def center(self):
        return self.cx * 2.0, self.cy * 2.0

    @property
    def rect(self):
        return tuple(int(v * 2) for v in self.box)


def box_depth(depth, box):
    """Distance (m) of the nearest surface in the middle of a detector box, or None."""
    x1, y1, x2, y2 = [int(v) for v in box]
    mx, my = (x2 - x1) // 4, (y2 - y1) // 5
    mid = depth[max(0, y1 + my):max(0, y2 - my):2, max(0, x1 + mx):max(0, x2 - mx):2]
    v = mid[mid > 0]
    if mid.size == 0 or v.size < 0.1 * mid.size:
        return None
    return float(np.percentile(v, 25)) / 1000.0


def motion_blob(depth, ref):
    """Where did something walk in? Compares a depth picture with an earlier one
    taken while the robot stood still. Returns the (x, y) pixel of the biggest
    area that is now clearly nearer, or None."""
    d, r = depth[::2, ::2], ref[::2, ::2]
    nearer = (d > 0) & (d < 3500) & ((r == 0) | (d.astype(np.int32) < r.astype(np.int32) - MOTION_MM))
    mask = cv.morphologyEx(nearer.astype(np.uint8), cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    if stats[i, cv.CC_STAT_AREA] < MOTION_MIN_AREA:
        return None
    ys, xs = np.nonzero(lab == i)
    k = len(xs) // 2                                    # a pixel that really is on the blob
    order = np.argsort(xs)
    return float(xs[order[k]] * 2), float(ys[order[k]] * 2)


def depth_cmd(cx, dist, frame_w, target_dist, max_fwd, max_turn):
    """Steer toward a target at pixel column cx and keep target_dist metres away."""
    ex = (cx - frame_w / 2.0) / (frame_w / 2.0)
    vz = 0.0 if abs(ex) < TURN_DEADBAND else clamp(-TURN_GAIN * 0.8 * ex, -max_turn, max_turn)
    err = dist - target_dist
    vx = 0.0 if abs(err) < DIST_DEADBAND else clamp(DIST_GAIN * err, -MAX_REVERSE, max_fwd)
    if vx > 0:
        vx *= max(0.0, 1.0 - abs(ex))
    return float(vx), float(vz), float(ex)


def lidar_guard(vx, front, rear):
    """Limit forward/backward speed by what the lidar sees. Returns (vx, note).

    front/rear are free distances in metres, or None if the lidar has no data.
    Without data, driving forward is left to the camera but reversing (which
    the camera cannot see) is refused.
    """
    if vx > 0 and front is not None:
        if front <= LIDAR_STOP:
            return 0.0, "blocked ahead"
        if front < LIDAR_SLOW:
            return vx * (front - LIDAR_STOP) / (LIDAR_SLOW - LIDAR_STOP), ""
    if vx < 0 and (rear is None or rear <= LIDAR_STOP):
        return 0.0, "blocked behind"
    return vx, ""


def scan_cmd(t, direction):
    """Spin search, t seconds after it began. Returns (vz, status).

    direction is +1 to start turning left, -1 to start turning right.
    """
    if t > SCAN_GIVE_UP_S:
        return 0.0, "lost"
    sweep, tc = divmod(t, SCAN_SWEEP_S + SCAN_REST_S)
    if tc >= SCAN_SWEEP_S:
        return 0.0, "search"                               # resting
    return direction * (1.0 if int(sweep) % 2 == 0 else -1.0) * SCAN_TURN, "search"


class Recovery(object):
    """Remembers how the toy was moving and decides what to do once it is gone.

    - Toy was close, low in the picture, or left through the top: back up a
      little first, which widens what the camera sees.
    - Otherwise turn the way the toy was heading, then scan all around.
    """

    def __init__(self):
        self.t = 0.0          # when the toy was last seen (or GO was pressed)
        self.ex = 0.0         # last horizontal position, -1 left .. +1 right
        self.vex = 0.0        # sideways speed with the robot's own turning removed
        self.close = False
        self.plan = None
        self._have = False

    def idle(self, now):
        """Not following: forget everything, so GO starts with a plain scan."""
        self.__init__()
        self.t = now

    def seen(self, now, ex, close, vz_cmd):
        """ex: target's horizontal position (-1 left .. +1 right). close: it is so
        near that losing it probably means it slipped under/inside the camera's view."""
        dt = now - self.t
        if self._have and 0.005 < dt < 0.3:
            # Turning left (vz > 0) slides a still target to the right in the
            # picture; take that out so we measure the target, not ourselves.
            v = (ex - self.ex) / dt - vz_cmd / HALF_FOV
            self.vex = 0.6 * self.vex + 0.4 * clamp(v, -6.0, 6.0)
        else:
            self.vex = 0.0
        self.close = close
        self.t, self.ex, self.plan, self._have = now, ex, None, True

    def cmd(self, now):
        """Returns (vx, vz, status, note) for a frame with no toy in it."""
        if now - self.t < LOST_GRACE_S:
            return 0.0, 0.0, "follow", ""
        if self.plan is None:
            guess = self.ex + self.vex * PREDICT_S          # where it was heading
            if abs(guess) < 0.15:
                guess = self.ex if abs(self.ex) > 0.05 else -1.0
            self.plan = {"t0": now, "dir": -math.copysign(1.0, guess),
                         "backup": BACKUP_S if (self.close and self._have) else 0.0}
        t = now - self.plan["t0"]
        if t < self.plan["backup"]:
            return -BACKUP_SPEED, 0.0, "backup", "too close"
        vz, status = scan_cmd(t - self.plan["backup"], self.plan["dir"])
        return 0.0, vz, status, "left" if self.plan["dir"] > 0 else "right"


class Drive(object):
    """Thin wrapper over Rosmaster_Lib so the rest can run without the robot."""

    LIGHTS = {"idle": (0, 0, 60), "follow": (0, 120, 0), "search": (120, 90, 0), "backup": (120, 40, 0), "lost": (120, 0, 0)}

    def __init__(self, enabled, invert_turn=False, debug=False):
        self.bot = None
        self.sign = -1.0 if invert_turn else 1.0
        self._last = (None, None)
        self._last_t = 0.0
        self._light = None
        self._beep_t = 0.0
        if enabled:
            from Rosmaster_Lib import Rosmaster   # only exists on the Pi image
            self.bot = Rosmaster(debug=debug)
            self.bot.create_receive_threading()

    def move(self, vx, vz):
        cmd = (round(vx, 3), round(vz * self.sign, 3))
        now = time.time()
        # Re-send unchanged commands only occasionally to keep the serial quiet.
        if cmd == self._last and now - self._last_t < 0.3:
            return
        self._last, self._last_t = cmd, now
        if self.bot:
            self.bot.set_car_motion(cmd[0], 0, cmd[1])

    def stop(self):
        self._last = (0.0, 0.0)
        self._last_t = time.time()
        if self.bot:
            self.bot.set_car_motion(0, 0, 0)

    def light(self, state):
        if state == self._light:
            return
        self._light = state
        if self.bot:
            try:
                r, g, b = self.LIGHTS.get(state, (0, 0, 0))
                self.bot.set_colorful_effect(0, 255, 255)
                self.bot.set_colorful_lamps(0xFF, r, g, b)
            except Exception:
                pass

    def beep(self, ms=60):
        if self.bot and time.time() - self._beep_t > 2.0:
            self._beep_t = time.time()
            try:
                self.bot.set_beep(ms)
            except Exception:
                pass

    def battery(self):
        try:
            return round(float(self.bot.get_battery_voltage()), 1) if self.bot else None
        except Exception:
            return None

    def shutdown(self):
        self.stop()
        if self.bot:
            try:
                self.bot.set_colorful_lamps(0xFF, 0, 0, 0)
                self.bot.set_beep(0)
            except Exception:
                pass
            self.bot.set_car_motion(0, 0, 0)


class FollowBot(object):
    def __init__(self, camera, drive, target_size=0.10, search=True, lidar=None):
        self.cam = camera
        self.drive = drive
        self.lidar = lidar
        self.frames_dir = None          # set to a folder to save a picture each time a toy is spotted
        self._frame_n = 0
        self._frame_t = 0.0
        self.search = search
        self.lock = threading.Lock()
        self.running = True
        # --- shared state (guarded by self.lock) ---
        self.following = False
        self.color_name = "red"
        self.ranges = COLORS["red"]
        self.speed_name = "turtle"
        self.target_size = target_size
        self.status = "idle"
        self.message = ""
        self.jpeg = None
        self._want_learn = False
        self._want_distance = False
        self.show_mask = False
        self.raw = None
        # "shape" = follow a person/thing through the depth camera; "colour" = follow a colour
        self.mode = "shape" if hasattr(camera, "depth") else "colour"
        self.target = DepthTarget()
        self.target_dist = FOLLOW_DIST
        self.detector = None            # set later if the object finder loads
        self.detector_note = "off"
        self._tap = None

    # ---- called from the web threads ----
    def set_following(self, on):
        with self.lock:
            self.following = bool(on)
        if not on:
            self.drive.stop()          # don't wait for the next camera frame

    def set_color(self, name):
        with self.lock:
            self.color_name, self.ranges, self.message = name, COLORS[name], ""
            self.mode = "colour"

    def set_mode(self, mode):
        with self.lock:
            if mode == "shape" and not hasattr(self.cam, "depth"):
                self.message = "No depth camera, so I can only follow colours."
                return
            self.mode, self.message = mode, ""
            self.target.locked = False

    def tap(self, fx, fy):
        """The picture was tapped at (fx, fy), each 0..1."""
        with self.lock:
            self._tap = (clamp(fx, 0.0, 1.0) * PROC_W, clamp(fy, 0.0, 1.0) * PROC_H)

    def set_speed(self, name):
        with self.lock:
            self.speed_name = name

    def request_learn(self):
        with self.lock:
            self._want_learn = True

    def toggle_mask(self):
        with self.lock:
            self.show_mask = not self.show_mask

    def raw_jpeg(self):
        with self.lock:
            raw = self.raw
        if raw is None:
            return None
        ok, jpg = cv.imencode(".jpg", raw, [cv.IMWRITE_JPEG_QUALITY, 92])
        return jpg.tobytes() if ok else None

    def request_distance(self):
        with self.lock:
            self._want_distance = True

    def snapshot(self):
        with self.lock:
            return {"following": self.following, "color": self.color_name if self.mode == "colour" else "",
                    "mode": self.mode, "target": self.target.label if self.target.locked else "",
                    "finder": self.detector_note,
                    "speed": self.speed_name, "status": self.status,
                    "message": self.message, "mask": self.show_mask, "battery": self.drive.battery()}

    # ---- control loop (own thread) ----
    def loop(self):
        try:
            self._loop()
        finally:
            self.drive.stop()          # any crash here must leave the wheels stopped

    def _loop(self):
        rec = Recovery()
        was_visible, prev, shown, streak = False, None, None, 0
        vx_s, vz_s = 0.0, 0.0
        fails = 0
        last_ok = 0.0            # shape mode: when the target was last found
        still_since = None       # when the robot last came to rest
        ref, ref_t = None, 0.0   # depth picture to compare against for "something walked in"
        det_seq = -1
        while self.running:
            ok, raw = self.cam.get_frame()
            if not ok:
                self.drive.stop()
                vx_s = vz_s = 0.0
                with self.lock:
                    self.status = "no camera"
                fails += 1
                if fails % 20 == 0:
                    try:
                        self.cam.reconnect()
                    except Exception:
                        pass
                time.sleep(0.05)
                continue
            fails = 0
            frame = raw if raw.shape[:2] == (PROC_H, PROC_W) else cv.resize(raw, (PROC_W, PROC_H))
            depth = self.cam.depth() if hasattr(self.cam, "depth") else None
            if depth is not None and depth.shape != (PROC_H, PROC_W):
                depth = None
            if self.detector is not None:
                self.detector.submit(frame)

            with self.lock:
                if self._want_learn:
                    self._want_learn = False
                    ranges, msg = learn_color(frame)
                    self.message = msg
                    if ranges:
                        self.ranges, self.color_name, self.mode = ranges, "learned", "colour"
                tap, self._tap = self._tap, None
                if tap is not None:
                    # Tap = "follow this". By shape if the depth camera can see it, else by its colour.
                    if depth is not None and self.target.start_at(depth, tap[0], tap[1]):
                        self.mode, self.message = "shape", "OK! Following that (%.1f m away)." % self.target.dist
                        last_ok = time.time()
                    else:
                        ranges, msg = learn_color(frame, at=tap)
                        if ranges:
                            self.ranges, self.color_name, self.mode = ranges, "learned", "colour"
                            self.message = "OK! Following that colour."
                        else:
                            self.message = "I can't make that out. Tap something closer or more colourful."
                following, ranges, show_mask = self.following, self.ranges, self.show_mask
                mode = self.mode
                self.raw = frame
                max_fwd, max_turn = SPEEDS[self.speed_name]
                target, target_dist = self.target_size, self.target_dist

            now = time.time()
            frame = frame.copy()                      # keep self.raw free of drawings
            vx = vz = 0.0
            note = ""
            det = blip = None                         # (cx, cy, size) of what is being followed
            people = []

            if mode == "shape":
                tgt = self.target
                if self.detector is not None:
                    dets, dets_t, seq = self.detector.results()
                    if now - dets_t < 0.6:
                        people = [d for d in dets if d["cls"] == "person"]
                found = tgt.locked and depth is not None and tgt.update(depth)
                if found:
                    last_ok = now
                elif tgt.locked and now - last_ok > LOST_GRACE_S:
                    tgt.locked = False
                    print("[follow_bot] lost the %s" % tgt.label, flush=True)

                # Nothing locked: try to pick a target up.
                if following and not tgt.locked and depth is not None:
                    if people and self.detector is not None and seq != det_seq:
                        det_seq = seq
                        near = [(box_depth(depth, d["box"]), d) for d in people]
                        near = sorted([n for n in near if n[0] is not None and n[0] < 4.5], key=lambda n: n[0])
                        if near and tgt.start_box(depth, near[0][1]["box"], "person"):
                            found, last_ok = True, now
                    if not tgt.locked and still_since is not None and now - still_since > 0.7:
                        if ref is None or now - ref_t > 3.0:
                            ref, ref_t = depth, now       # fresh "before" picture
                        else:
                            spot = motion_blob(depth, ref)
                            if spot and tgt.start_at(depth, spot[0], spot[1], "visitor"):
                                found, last_ok = True, now
                    if found:
                        self.drive.beep()
                        print("[follow_bot] picked up a %s at %.2f m" % (tgt.label, tgt.dist), flush=True)

                if not following:
                    status = "idle"
                    rec.idle(now)
                elif found:
                    cx, cy = tgt.center
                    x1, y1, x2, y2 = tgt.rect
                    det = (cx, cy, x2 - x1)
                    vx, vz, ex = depth_cmd(cx, tgt.dist, PROC_W, target_dist, max_fwd, max_turn)
                    rec.seen(now, ex, tgt.dist < 0.9, vz_s)
                    status = "follow"
                    note = "%s %.1fm" % (tgt.label, tgt.dist)
                    if not was_visible:
                        self._save_frame(frame, det, now)
                elif tgt.locked:
                    status = "follow"                 # missed a frame or two; hold still
                elif self.search and (rec._have or self.detector is not None):
                    # Lost it, or GO with nobody in view and a person finder to look with: turn and look.
                    vx, vz, status, note = rec.cmd(now)
                else:
                    status = "waiting"
                if status == "lost":                  # gave up looking: stand still and watch for a visitor
                    status, note = "waiting", ""
            else:
                det, mask = detect(frame, ranges, prev)
                if det:
                    prev = (det[0], det[1])
                elif now - rec.t > 1.0:
                    prev = None
                if show_mask:
                    frame[mask > 0] = (0, 255, 0)
                streak = streak + 1 if det else 0
                blip = det if (det and not was_visible and streak < CONFIRM_FRAMES) else None
                if blip:
                    det = None                        # not convinced yet; carry on as if unseen
                if not following:
                    status = "idle"
                    rec.idle(now)                     # pressing GO with no toy in view starts a scan
                elif det:
                    vx, vz = compute_cmd(det, PROC_W, PROC_H, target, max_fwd, max_turn)
                    cx, cy, size = det
                    low, top = (cy + size / 2.0) / PROC_H, (cy - size / 2.0) / PROC_H
                    rec.seen(now, (cx - PROC_W / 2.0) / (PROC_W / 2.0),
                             low >= NEAR_EDGE or top <= 0.02 or size > 1.5 * target * PROC_W, vz_s)
                    status = "follow"
                    if not was_visible:
                        self.drive.beep()
                        print("[follow_bot] saw toy at x=%d y=%d, %d px wide" % det, flush=True)
                        self._save_frame(frame, det, now)
                elif self.search:
                    vx, vz, status, note = rec.cmd(now)
                    if blip:
                        vz = 0.0                      # hold still for a proper look
                else:
                    status = "follow" if now - rec.t < LOST_GRACE_S else "lost"

            if self.lidar is not None:
                front, rear = self.lidar.clearance()
                vx, guard = lidar_guard(vx, front, rear)
                if guard:
                    note = (note + " " + guard).strip()
            key = (status, note.split(" ")[0] if status == "follow" else note)
            if key != shown:                          # one terminal line per change, for debugging
                shown = key
                print("[follow_bot] %s %s" % (status, note), flush=True)
            was_visible = det is not None

            # Ease into motion, but stop instantly.
            vx_s = 0.0 if vx == 0 else 0.6 * vx_s + 0.4 * vx
            vz_s = 0.0 if vz == 0 else 0.5 * vz_s + 0.5 * vz
            if vx_s == 0 and vz_s == 0:
                still_since = still_since or now
            else:
                still_since, ref = None, None

            with self.lock:
                still_following = self.following      # STOP may have arrived mid-frame
                if self._want_distance:
                    self._want_distance = False
                    if mode == "shape" and self.target.locked:
                        self.target_dist = clamp(self.target.dist, 0.8, 3.0)
                        self.message = "OK, I'll stay %.1f m away." % self.target_dist
                    elif mode == "colour" and det:
                        self.target_size = clamp(det[2] / PROC_W, 0.03, 0.6)
                        self.message = "OK, I'll stay this far away."
                    else:
                        self.message = "I can't see it. Show me what to follow first."
                self.status = status
            if still_following:
                self.drive.move(vx_s, vz_s)
            else:
                self.drive.stop()
                vx_s = vz_s = 0.0
            self.drive.light("lost" if status == "waiting" else status)

            for d in people:
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
            if mode == "shape" and self.target.locked:
                x1, y1, x2, y2 = self.target.rect
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self._draw(frame, None, (status + " " + note).strip(), True)
            else:
                self._draw(frame, det or blip, (status + " " + note).strip(), following or mode == "shape")
            if self.lidar is not None:
                self._draw_radar(frame)
            if depth is not None or hasattr(self.cam, "depth"):
                self._draw_depth(frame)
            ok, jpg = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self.lock:
                    self.jpeg = jpg.tobytes()

    def _save_frame(self, frame, det, now, keep=40):
        """Debug aid: keep pictures of the last `keep` things taken for the toy."""
        if not self.frames_dir or now - self._frame_t < 0.7:
            return
        self._frame_t = now
        pic = frame.copy()
        cv.circle(pic, (int(det[0]), int(det[1])), int(det[2] / 2) + 4, (255, 255, 255), 2)
        name = "spot-%02d.jpg" % (self._frame_n % keep)
        self._frame_n += 1
        try:
            cv.imwrite(os.path.join(self.frames_dir, name), pic, [cv.IMWRITE_JPEG_QUALITY, 80])
            print("[follow_bot]   saved %s" % name, flush=True)
        except Exception:
            pass

    def _draw_depth(self, frame, w=160, h=120):
        """Small depth picture in the bottom-right corner: near = red, far = blue,
        black = no reading. The number is the nearest thing straight ahead."""
        depth = self.cam.depth()
        x0, y0 = frame.shape[1] - w - 8, frame.shape[0] - h - 8
        if depth is None:
            cv.putText(frame, "no depth", (x0 + 30, y0 + h // 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return
        small = cv.resize(depth, (w, h), interpolation=cv.INTER_NEAREST)
        img = cv.applyColorMap(255 - np.clip(small / 4000.0 * 255, 0, 255).astype(np.uint8), cv.COLORMAP_JET)
        img[small == 0] = 0
        frame[y0:y0 + h, x0:x0 + w] = img
        cv.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
        from orbbec_camera import depth_ahead
        ahead = depth_ahead(depth)
        cv.putText(frame, "ahead %.2f m" % ahead if ahead else "ahead ?", (x0 + 4, y0 + 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _draw_radar(self, frame, size=150):
        """Top-down view of the lidar points in the corner of the video.

        Robot in the middle facing up; the box ahead/behind is the lane the
        guard watches; points inside the stop distance are drawn red.
        """
        pts = self.lidar.points()
        x0, y0 = frame.shape[1] - size - 8, 8
        cx, cy, sc = x0 + size // 2, y0 + size // 2, (size / 2.0) / RADAR_RANGE
        frame[y0:y0 + size, x0:x0 + size] = frame[y0:y0 + size, x0:x0 + size] // 3
        cv.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (255, 255, 255), 1)
        lane, stop = int(0.18 * sc), int(LIDAR_STOP * sc)
        cv.rectangle(frame, (cx - lane, cy - stop), (cx + lane, cy + stop), (120, 120, 120), 1)
        for x, y in pts:
            px, py = int(cx - y * sc), int(cy - x * sc)
            if x0 < px < x0 + size - 1 and y0 < py < y0 + size - 1:
                near = abs(y) <= 0.18 and abs(x) <= LIDAR_STOP
                cv.circle(frame, (px, py), 1, (0, 0, 255) if near else (0, 255, 255), -1)
        cv.drawMarker(frame, (cx, cy), (255, 255, 255), cv.MARKER_TRIANGLE_UP, 8, 1)
        if not pts:
            cv.putText(frame, "no lidar data", (x0 + 8, cy + 30), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    @staticmethod
    def _draw(frame, det, status, following):
        h, w = frame.shape[:2]
        if det:
            c = (int(det[0]), int(det[1]))
            cv.circle(frame, c, int(det[2] / 2) + 3, (255, 255, 255), 2)
            cv.drawMarker(frame, c, (255, 255, 255), cv.MARKER_CROSS, 12, 1)
        if not following:
            r = int(h * LEARN_BOX / 2)
            cv.rectangle(frame, (w // 2 - r, h // 2 - r), (w // 2 + r, h // 2 + r), (255, 255, 255), 1)
        cv.putText(frame, status.upper(), (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv.putText(frame, status.upper(), (12, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# ----------------------------------------------------------------------------
# Web page (phone friendly)
# ----------------------------------------------------------------------------
PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Follow Bot</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;padding:12px;background:#14213d;color:#fff;text-align:center}
  h1{margin:4px 0 10px;font-size:1.5rem}
  img{width:100%;max-width:480px;border-radius:14px;border:3px solid #fff3}
  .row{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:10px auto;max-width:480px}
  button{border:0;border-radius:14px;padding:14px 10px;font-size:1.05rem;font-weight:700;color:#fff;
         background:#3a4a6b;flex:1 1 28%;min-width:88px}
  button.on{outline:4px solid #fff}
  #go{background:#2a9d4a;font-size:1.5rem;flex-basis:45%}
  #stop{background:#d62828;font-size:1.5rem;flex-basis:45%}
  #status{font-size:1.15rem;min-height:1.6em;margin:8px 0}
  small{opacity:.7}
</style></head><body>
<h1>&#129302; Follow Bot</h1>
<img id="cam" src="/video_feed" alt="robot camera">
<small>Tap anything in the picture to follow it</small>
<div id="status">&nbsp;</div>
<div class="row">
  <button id="go" onclick="post('start')">GO!</button>
  <button id="stop" onclick="post('stop')">STOP</button>
</div>
<div class="row">
  <button id="shape" onclick="post('mode/shape')">&#128694; Follow people and things</button>
</div>
<div class="row" id="colors">
  <button data-c="red" style="background:#d62828">Red</button>
  <button data-c="orange" style="background:#e76f00">Orange</button>
  <button data-c="yellow" style="background:#c79a00">Yellow</button>
  <button data-c="green" style="background:#2a9d4a">Green</button>
  <button data-c="blue" style="background:#1d6fd6">Blue</button>
  <button data-c="pink" style="background:#d6409f">Pink</button>
</div>
<div class="row">
  <button data-c="learned" onclick="post('learn')">Learn my toy's colour</button>
  <button onclick="post('distance')">Stay this far away</button>
</div>
<div class="row" id="speeds">
  <button data-s="turtle">&#128034; Turtle</button>
  <button data-s="bunny">&#128007; Bunny</button>
</div>
<div class="row">
  <button id="mask" onclick="post('mask')">&#128064; Show what matches</button>
</div>
<small id="batt"></small>
<p><a href="/lidar" style="color:#9cc3ff">Lidar view</a></p>
<script>
const words = {idle:"Press GO, then tap what to follow or walk in front of me!", follow:"I see it! Following...",
               waiting:"Tap what to follow, or walk in front of me!",
               search:"Where did it go? Looking...", backup:"Oops, too close! Backing up...", lost:"I can't find it. Show me the toy!",
               "no camera":"My camera is not working."};
function post(p){ fetch('/api/'+p,{method:'POST'}).then(r=>r.json()).then(show); }
function show(s){
  let t = s.message || words[s.status] || s.status;
  if (!s.message && s.status=='follow' && s.target) t = "Following the "+s.target+"!";
  document.getElementById('status').textContent = t;
  document.getElementById('shape').classList.toggle('on', s.mode=='shape');
  document.querySelectorAll('[data-c]').forEach(b=>b.classList.toggle('on', b.dataset.c==s.color));
  document.querySelectorAll('[data-s]').forEach(b=>b.classList.toggle('on', b.dataset.s==s.speed));
  document.getElementById('go').classList.toggle('on', s.following);
  document.getElementById('mask').classList.toggle('on', s.mask);
  document.getElementById('batt').textContent = (s.battery ? 'Battery '+s.battery+' V   ' : '') + 'Person finder: '+s.finder;
}
document.getElementById('cam').addEventListener('click', e=>{
  const r=e.target.getBoundingClientRect();
  post('tap/'+((e.clientX-r.left)/r.width).toFixed(4)+'/'+((e.clientY-r.top)/r.height).toFixed(4));
});
document.querySelectorAll('#colors [data-c]').forEach(b=>b.onclick=()=>post('color/'+b.dataset.c));
document.querySelectorAll('[data-s]').forEach(b=>b.onclick=()=>post('speed/'+b.dataset.s));
setInterval(()=>fetch('/api/status').then(r=>r.json()).then(show).catch(()=>{}), 700);
</script></body></html>
"""


LIDAR_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lidar</title>
<style>body{margin:0;background:#101623;color:#dfe6f3;font-family:system-ui,sans-serif;text-align:center}
canvas{width:min(96vw,82vh);height:min(96vw,82vh);background:#0a0e17;border-radius:12px;margin-top:8px}
p{margin:8px}</style></head><body>
<p id="info">waiting for the lidar...</p>
<canvas id="c" width="900" height="900"></canvas>
<p>Robot in the middle, facing up. Rings every 1 m. Range <select id="r"><option>2</option><option selected>4</option><option>8</option><option>12</option></select> m
&nbsp; <a href="/" style="color:#8fb8ff">back to Follow Bot</a></p>
<script>
const c=document.getElementById('c'),g=c.getContext('2d'),W=c.width;
function draw(d){
  const R=+document.getElementById('r').value, sc=W/2/R, m=W/2;
  g.clearRect(0,0,W,W); g.strokeStyle='#26324a'; g.lineWidth=1;
  for(let i=1;i<=R;i++){g.beginPath();g.arc(m,m,i*sc,0,7);g.stroke();}
  g.beginPath();g.moveTo(m,0);g.lineTo(m,W);g.moveTo(0,m);g.lineTo(W,m);g.stroke();
  g.strokeStyle='#6b7a99';g.strokeRect(m-0.18*sc,m-d.stop*sc,0.36*sc,2*d.stop*sc);
  for(const [x,y] of d.points){
    const near=Math.abs(y)<=0.18&&Math.abs(x)<=d.stop;
    g.fillStyle=near?'#ff5a5a':'#ffe45c'; g.beginPath(); g.arc(m-y*sc,m-x*sc,3,0,7); g.fill();
  }
  g.fillStyle='#fff'; g.beginPath(); g.moveTo(m,m-12); g.lineTo(m-8,m+8); g.lineTo(m+8,m+8); g.fill();
  const f=v=>v==null?'clear':v.toFixed(2)+' m';
  document.getElementById('info').textContent = d.on ? d.points.length+' points this moment (of 360 directions) | ahead: '+f(d.front)+' | behind: '+f(d.rear) : 'lidar is off';
}
setInterval(()=>fetch('/api/lidar').then(r=>r.json()).then(draw).catch(()=>{}),200);
</script></body></html>
"""


def make_app(fb):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(PAGE)

    @app.route("/video_feed")
    def video_feed():
        def gen():
            while fb.running:
                with fb.lock:
                    jpg = fb.jpeg
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.07)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def status():
        return jsonify(fb.snapshot())

    @app.route("/api/start", methods=["POST"])
    def start():
        fb.set_following(True)
        return jsonify(fb.snapshot())

    @app.route("/api/stop", methods=["POST"])
    def stop():
        fb.set_following(False)
        return jsonify(fb.snapshot())

    @app.route("/api/color/<name>", methods=["POST"])
    def color(name):
        if name in COLORS:
            fb.set_color(name)
        return jsonify(fb.snapshot())

    @app.route("/api/mode/<name>", methods=["POST"])
    def mode(name):
        if name in ("shape", "colour"):
            fb.set_mode(name)
        return jsonify(fb.snapshot())

    @app.route("/api/tap/<float:fx>/<float:fy>", methods=["POST"])
    def tap(fx, fy):
        fb.tap(fx, fy)
        time.sleep(0.2)                # let the camera loop act on it
        return jsonify(fb.snapshot())

    @app.route("/api/speed/<name>", methods=["POST"])
    def speed(name):
        if name in SPEEDS:
            fb.set_speed(name)
        return jsonify(fb.snapshot())

    @app.route("/api/learn", methods=["POST"])
    def learn():
        fb.set_following(False)        # hold still while the toy is in the box
        fb.request_learn()
        time.sleep(0.25)               # let the camera loop pick it up
        return jsonify(fb.snapshot())

    @app.route("/api/lidar")
    def lidar_points():
        if fb.lidar is None:
            return jsonify({"on": False, "points": []})
        front, rear = fb.lidar.clearance()
        fin = lambda v: None if v is None or v == float("inf") else round(v, 2)
        return jsonify({"on": True, "front": fin(front), "rear": fin(rear), "stop": LIDAR_STOP,
                        "points": [[round(x, 3), round(y, 3)] for x, y in fb.lidar.points()]})

    @app.route("/lidar")
    def lidar_page():
        return render_template_string(LIDAR_PAGE)

    @app.route("/api/mask", methods=["POST"])
    def mask():
        fb.toggle_mask()
        return jsonify(fb.snapshot())

    @app.route("/snapshot.jpg")
    def snapshot():
        jpg = fb.raw_jpeg()             # clean camera frame, handy for tuning colours
        return Response(jpg, mimetype="image/jpeg") if jpg else ("no frame yet", 503)

    @app.route("/api/distance", methods=["POST"])
    def distance():
        fb.request_distance()
        time.sleep(0.25)
        return jsonify(fb.snapshot())

    return app


def open_camera(which, debug):
    """auto: Orbbec SDK (colour + depth) if it is installed, otherwise OpenCV.
    opencv / usb / <number>: plain OpenCV capture, as rosmaster_main.py does."""
    if which == "auto":
        try:
            import orbbec_camera
            if orbbec_camera.sdk_available():
                cam = orbbec_camera.OrbbecCamera(debug=debug)
                if cam.isOpened():
                    print("[follow_bot] camera: Orbbec SDK, colour + %s"
                          % ("depth lined up with colour" if cam.aligned else
                             "depth (not lined up)" if cam.with_depth else "no depth"), flush=True)
                    return cam
                print("[follow_bot] Orbbec SDK camera failed (%s); trying OpenCV" % cam.error, flush=True)
        except Exception as e:
            print("[follow_bot] Orbbec SDK not usable (%s); trying OpenCV" % e, flush=True)
    from camera_rosmaster import Rosmaster_Camera
    ids = {"auto": 0x50, "opencv": 0x50, "depth": 0x50, "usb": 0x51}
    vid = ids[which] if which in ids else int(which)
    print("[follow_bot] camera: OpenCV", flush=True)
    return Rosmaster_Camera(video_id=vid, debug=debug)


def main():
    ap = argparse.ArgumentParser(description="Rosmaster X3 follow bot")
    ap.add_argument("--camera", default="auto",
                    help="auto (default: Orbbec SDK if installed, else OpenCV), opencv, usb, or a /dev/video index")
    ap.add_argument("--port", type=int, default=6501)
    ap.add_argument("--no-motors", action="store_true", help="vision only; never move the wheels")
    ap.add_argument("--invert-turn", action="store_true", help="use if the robot turns AWAY from the toy")
    ap.add_argument("--no-search", action="store_true", help="don't turn around looking for a lost toy")
    ap.add_argument("--target-size", type=float, default=0.10,
                    help="how big the toy should look, as a fraction of picture width (bigger = closer)")
    ap.add_argument("--lidar-front", type=float, default=LIDAR_FRONT_DEG, metavar="DEG",
                    help="lidar angle that points straight ahead of the robot (tools/pi.sh calibrate)")
    ap.add_argument("--no-lidar", action="store_true", help="ignore the lidar even if it is plugged in")
    ap.add_argument("--lidar-port", default="/dev/rplidar")
    ap.add_argument("--no-finder", action="store_true",
                    help="don't load the person finder (YOLO); people are then picked up by tapping or walking in")
    ap.add_argument("--save-frames", default=None, metavar="DIR",
                    help="save a picture to DIR each time something is taken for the toy (last 40 kept)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cam = open_camera(args.camera, args.debug)
    if not cam.isOpened():
        raise SystemExit("Camera '%s' did not open. Is rosmaster_main.py still running? "
                         "(bash rosmaster/kill_rosmaster.sh)" % args.camera)
    drive = Drive(enabled=not args.no_motors, invert_turn=args.invert_turn, debug=args.debug)
    lidar = None
    if args.no_lidar or not os.path.exists(args.lidar_port):
        print("[follow_bot] lidar off (%s)" % ("--no-lidar" if args.no_lidar else args.lidar_port + " not found"), flush=True)
    else:
        from lidar import Lidar
        try:
            lidar = Lidar(args.lidar_port, front_deg=args.lidar_front).start()
            print("[follow_bot] lidar on, front at %.0f deg" % args.lidar_front, flush=True)
        except Exception as e:
            print("[follow_bot] lidar NOT available (%s); it will refuse to reverse" % e, flush=True)
            lidar = Lidar(args.lidar_port, front_deg=args.lidar_front)     # never started: reports no data
    fb = FollowBot(cam, drive, target_size=args.target_size, search=not args.no_search, lidar=lidar)

    def load_finder():
        # torch + YOLO take a while to import, so load them while the bot is already usable.
        try:
            fb.detector_note = "loading"
            from detector import YoloDetector
            fb.detector = YoloDetector()
            fb.detector_note = "on"
            print("[follow_bot] person finder ready", flush=True)
        except Exception as e:
            fb.detector_note = "not available"
            print("[follow_bot] person finder not available: %s" % (str(e).splitlines() or [repr(e)])[0], flush=True)

    if not args.no_finder and fb.mode == "shape":
        threading.Thread(target=load_finder, daemon=True).start()

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        fb.frames_dir = args.save_frames
    if not args.debug:
        logging.getLogger("werkzeug").setLevel(logging.WARNING)    # no line per web request

    def shutdown(*_):
        fb.running = False
        drive.shutdown()
        if lidar is not None:
            lidar.stop()
        if hasattr(cam, "close"):
            cam.close()

    atexit.register(shutdown)
    signal.signal(signal.SIGTERM, lambda *_: (shutdown(), exit(0)))

    t = threading.Thread(target=fb.loop, daemon=True)
    t.start()
    print("Follow Bot ready: open http://<this-pi's-ip>:%d on a phone%s"
          % (args.port, "  [NO MOTORS]" if args.no_motors else ""))
    try:
        make_app(fb).run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
    finally:
        shutdown()
        t.join(timeout=1.0)


if __name__ == "__main__":
    main()
