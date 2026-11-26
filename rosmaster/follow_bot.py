#!/usr/bin/env python3
# coding=utf-8
"""
Follow Bot for the Yahboom Rosmaster X3.

The robot follows a brightly coloured thing (a ball, a hat, a toy) that it
sees with its camera. It turns to keep the thing in the middle of the picture
and drives forward/back to keep it the same apparent size (= same distance).

Run on the Pi (stop the stock app first, both need the camera and serial port):

    bash rosmaster/kill_rosmaster.sh
    sudo python3 rosmaster/follow_bot.py              # then open http://<pi-ip>:6501
    sudo python3 rosmaster/follow_bot.py --no-motors  # camera/vision only, wheels never move

Stop with the STOP button on the page or Ctrl+C. Do NOT use `kill -9`:
the motors keep their last command if the program cannot say "stop".
"""
import argparse
import atexit
import math
import signal
import threading
import time

import cv2 as cv
import numpy as np
from flask import Flask, Response, jsonify, render_template_string

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
PROC_W, PROC_H = 320, 240     # frames are shrunk to this before processing

# Preset colours, OpenCV HSV (H 0-179). Each is a list of (low, high) ranges.
COLORS = {
    "red":    [((0, 120, 70), (9, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "orange": [((8, 130, 100), (21, 255, 255))],
    "yellow": [((22, 100, 100), (34, 255, 255))],
    "green":  [((40, 80, 60), (85, 255, 255))],
    "blue":   [((95, 120, 60), (125, 255, 255))],
    "pink":   [((140, 70, 100), (169, 255, 255))],
}

# Speed presets: (max forward m/s, max turn rad/s)
SPEEDS = {
    "turtle": (0.15, 1.0),
    "bunny":  (0.28, 1.6),
}

MAX_REVERSE = 0.10      # m/s. Backing up is blind, keep it gentle.
MIN_RADIUS_PX = 7       # ignore blobs smaller than this (at 320x240)
TURN_GAIN = 2.0         # rad/s per unit of horizontal error (-1..1)
DRIVE_GAIN = 0.45       # m/s per unit of relative size error
TURN_DEADBAND = 0.08    # fraction of half-frame
DRIVE_DEADBAND = 0.12   # relative size error
LOST_GRACE_S = 0.3      # coast-free pause before deciding the target is gone
SEARCH_S = 3.0          # spin slowly toward where it was last seen, then stop
SEARCH_TURN = 0.7       # rad/s
LEARN_BOX = 0.18        # centre box (fraction of frame height) for "Learn colour"


# ----------------------------------------------------------------------------
# Vision
# ----------------------------------------------------------------------------
_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))


def detect(frame, ranges):
    """Find the biggest blob of the target colour.

    Returns (cx, cy, size) in pixels of `frame`, where size is the diameter of
    a circle with the same area as the blob, or None if nothing is found.
    """
    hsv = cv.cvtColor(cv.GaussianBlur(frame, (5, 5), 0), cv.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        m = cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv.bitwise_or(mask, m)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, _KERNEL)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, _KERNEL)
    found = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]   # OpenCV 3 vs 4
    if not contours:
        return None
    c = max(contours, key=cv.contourArea)
    area = cv.contourArea(c)
    size = 2.0 * math.sqrt(area / math.pi)
    if size < 2 * MIN_RADIUS_PX:
        return None
    m = cv.moments(c)
    return (m["m10"] / m["m00"], m["m01"] / m["m00"], size)


def learn_color(frame):
    """Sample the centre box and build an HSV range around its colour.

    Returns (ranges, message). ranges is None if the sample is too grey/dark.
    """
    h, w = frame.shape[:2]
    r = int(h * LEARN_BOX / 2)
    roi = frame[h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r]
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV).reshape(-1, 3)
    s_med, v_med = int(np.median(hsv[:, 1])), int(np.median(hsv[:, 2]))
    if s_med < 70 or v_med < 50:
        return None, "Too dull. Pick something bright and colourful!"
    # Hue is circular (red wraps 179 -> 0), so take a circular mean.
    ang = hsv[:, 0].astype(np.float32) * (math.pi / 90.0)
    h_mean = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
    h_mid = int(round((h_mean * 90.0 / math.pi) % 180))
    tol = 11
    s_lo, v_lo = max(60, s_med - 80), max(50, v_med - 90)
    lo, hi = h_mid - tol, h_mid + tol
    if lo < 0:
        ranges = [((0, s_lo, v_lo), (hi, 255, 255)), ((180 + lo, s_lo, v_lo), (179, 255, 255))]
    elif hi > 179:
        ranges = [((lo, s_lo, v_lo), (179, 255, 255)), ((0, s_lo, v_lo), (hi - 180, 255, 255))]
    else:
        ranges = [((lo, s_lo, v_lo), (hi, 255, 255))]
    return ranges, "Got it!"


# ----------------------------------------------------------------------------
# Control
# ----------------------------------------------------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def compute_cmd(det, frame_w, target_size, max_fwd, max_turn):
    """Turn a detection into (vx m/s, vz rad/s). vz > 0 turns left (ROS style)."""
    cx, _, size = det
    ex = (cx - frame_w / 2.0) / (frame_w / 2.0)        # -1 left .. +1 right
    vz = 0.0 if abs(ex) < TURN_DEADBAND else clamp(-TURN_GAIN * ex, -max_turn, max_turn)

    want = target_size * frame_w
    es = (want - size) / want                           # >0 too far, <0 too close
    vx = 0.0 if abs(es) < DRIVE_DEADBAND else clamp(DRIVE_GAIN * es, -MAX_REVERSE, max_fwd)
    if vx > 0:
        vx *= max(0.0, 1.0 - abs(ex))                   # face it first, then drive
    return vx, vz


class Drive(object):
    """Thin wrapper over Rosmaster_Lib so the rest can run without the robot."""

    LIGHTS = {"idle": (0, 0, 60), "follow": (0, 120, 0), "search": (120, 90, 0), "lost": (120, 0, 0)}

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
    def __init__(self, camera, drive, target_size=0.22, search=True):
        self.cam = camera
        self.drive = drive
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
        self._last_size = None

    # ---- called from the web threads ----
    def set_following(self, on):
        with self.lock:
            self.following = bool(on)
        if not on:
            self.drive.stop()          # don't wait for the next camera frame

    def set_color(self, name):
        with self.lock:
            self.color_name, self.ranges, self.message = name, COLORS[name], ""

    def set_speed(self, name):
        with self.lock:
            self.speed_name = name

    def request_learn(self):
        with self.lock:
            self._want_learn = True

    def request_distance(self):
        with self.lock:
            self._want_distance = True

    def snapshot(self):
        with self.lock:
            return {"following": self.following, "color": self.color_name,
                    "speed": self.speed_name, "status": self.status,
                    "message": self.message, "battery": self.drive.battery()}

    # ---- control loop (own thread) ----
    def loop(self):
        try:
            self._loop()
        finally:
            self.drive.stop()          # any crash here must leave the wheels stopped

    def _loop(self):
        last_seen, last_ex, was_visible = 0.0, 0.0, False
        vx_s, vz_s = 0.0, 0.0
        fails = 0
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
            frame = cv.resize(raw, (PROC_W, PROC_H))

            with self.lock:
                if self._want_learn:
                    self._want_learn = False
                    ranges, msg = learn_color(frame)
                    self.message = msg
                    if ranges:
                        self.ranges, self.color_name = ranges, "learned"
                following, ranges = self.following, self.ranges
                max_fwd, max_turn = SPEEDS[self.speed_name]
                target = self.target_size

            det = detect(frame, ranges)
            now = time.time()
            vx = vz = 0.0
            if det:
                last_seen = now
                last_ex = (det[0] - PROC_W / 2.0) / (PROC_W / 2.0)
                if following:
                    vx, vz = compute_cmd(det, PROC_W, target, max_fwd, max_turn)
                    if not was_visible:
                        self.drive.beep()
                status = "follow" if following else "idle"
            elif not following:
                status = "idle"
            elif now - last_seen < LOST_GRACE_S:
                status = "follow"
            elif self.search and last_seen and now - last_seen < SEARCH_S:
                vz = -math.copysign(SEARCH_TURN, last_ex)
                status = "search"
            else:
                status = "lost"
            was_visible = det is not None

            # Ease into motion, but stop instantly.
            vx_s = 0.0 if vx == 0 else 0.6 * vx_s + 0.4 * vx
            vz_s = 0.0 if vz == 0 else 0.5 * vz_s + 0.5 * vz

            with self.lock:
                still_following = self.following      # STOP may have arrived mid-frame
                if self._want_distance:
                    self._want_distance = False
                    if det:
                        self.target_size = clamp(det[2] / PROC_W, 0.08, 0.6)
                        self.message = "OK, I'll stay this far away."
                    else:
                        self.message = "I can't see it. Show me the toy first."
                self.status = status
            if still_following:
                self.drive.move(vx_s, vz_s)
            else:
                self.drive.stop()
                vx_s = vz_s = 0.0
            self.drive.light(status)

            self._draw(frame, det, status, following)
            ok, jpg = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self.lock:
                    self.jpeg = jpg.tobytes()

    @staticmethod
    def _draw(frame, det, status, following):
        h, w = frame.shape[:2]
        if det:
            c = (int(det[0]), int(det[1]))
            cv.circle(frame, c, int(det[2] / 2), (0, 255, 0), 2)
            cv.circle(frame, c, 3, (0, 255, 0), -1)
        if not following:
            r = int(h * LEARN_BOX / 2)
            cv.rectangle(frame, (w // 2 - r, h // 2 - r), (w // 2 + r, h // 2 + r), (255, 255, 255), 1)
        cv.putText(frame, status.upper(), (8, 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv.putText(frame, status.upper(), (8, 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


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
<img src="/video_feed" alt="robot camera">
<div id="status">&nbsp;</div>
<div class="row">
  <button id="go" onclick="post('start')">GO!</button>
  <button id="stop" onclick="post('stop')">STOP</button>
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
<small id="batt"></small>
<script>
const words = {idle:"Press GO and show me the toy!", follow:"I see it! Following...",
               search:"Where did it go? Looking...", lost:"I lost it. Show me again!",
               "no camera":"My camera is not working."};
function post(p){ fetch('/api/'+p,{method:'POST'}).then(r=>r.json()).then(show); }
function show(s){
  document.getElementById('status').textContent = s.message || words[s.status] || s.status;
  document.querySelectorAll('[data-c]').forEach(b=>b.classList.toggle('on', b.dataset.c==s.color));
  document.querySelectorAll('[data-s]').forEach(b=>b.classList.toggle('on', b.dataset.s==s.speed));
  document.getElementById('go').classList.toggle('on', s.following);
  document.getElementById('batt').textContent = s.battery ? 'Battery '+s.battery+' V' : '';
}
document.querySelectorAll('#colors [data-c]').forEach(b=>b.onclick=()=>post('color/'+b.dataset.c));
document.querySelectorAll('[data-s]').forEach(b=>b.onclick=()=>post('speed/'+b.dataset.s));
setInterval(()=>fetch('/api/status').then(r=>r.json()).then(show).catch(()=>{}), 700);
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

    @app.route("/api/distance", methods=["POST"])
    def distance():
        fb.request_distance()
        time.sleep(0.25)
        return jsonify(fb.snapshot())

    return app


def open_camera(which, debug):
    from camera_rosmaster import Rosmaster_Camera
    ids = {"depth": 0x50, "usb": 0x51}
    vid = ids[which] if which in ids else int(which)
    return Rosmaster_Camera(video_id=vid, debug=debug)


def main():
    ap = argparse.ArgumentParser(description="Rosmaster X3 colour follow bot")
    ap.add_argument("--camera", default="depth", help="depth (default), usb, or a /dev/video index")
    ap.add_argument("--port", type=int, default=6501)
    ap.add_argument("--no-motors", action="store_true", help="vision only; never move the wheels")
    ap.add_argument("--invert-turn", action="store_true", help="use if the robot turns AWAY from the toy")
    ap.add_argument("--no-search", action="store_true", help="don't spin to look for a lost toy")
    ap.add_argument("--target-size", type=float, default=0.22,
                    help="how big the toy should look, as a fraction of picture width (bigger = closer)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cam = open_camera(args.camera, args.debug)
    if not cam.isOpened():
        raise SystemExit("Camera '%s' did not open. Is rosmaster_main.py still running? "
                         "(bash rosmaster/kill_rosmaster.sh)" % args.camera)
    drive = Drive(enabled=not args.no_motors, invert_turn=args.invert_turn, debug=args.debug)
    fb = FollowBot(cam, drive, target_size=args.target_size, search=not args.no_search)

    def shutdown(*_):
        fb.running = False
        drive.shutdown()

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
