#!/usr/bin/env python3
# coding=utf-8
"""
Minimal reader for the robot's RPLIDAR A1 (115200 baud on /dev/rplidar).

The lidar measures distance in one flat slice at its own height, all the way
round. It will not see a small toy on the floor, but it does see walls,
furniture and legs, so the follow bot uses it to avoid driving into things.

Angles handed out by this module are in the ROBOT's frame: 0 = straight
ahead, positive = to the left (counter-clockwise seen from above). The lidar's
own zero depends on how it is bolted on; `front_deg` is the lidar angle that
points straight ahead of the robot.

Run directly to see what the lidar sees (used for calibration):

    python3 lidar.py            # 4 seconds of scanning, summary per 15 degrees
"""
import math
import sys
import threading
import time

import serial

CMD_STOP = b"\xa5\x25"
CMD_SCAN = b"\xa5\x20"
SCAN_DESCRIPTOR = b"\xa5\x5a\x05\x00\x00\x40\x81"

MIN_RANGE = 0.15        # m. Closer than this is the robot's own body / sensor noise.
MAX_AGE = 0.6           # s. Readings older than this are ignored.


def parse_samples(buf):
    """Pull complete 5-byte samples out of buf.

    Returns (samples, leftover) where samples is a list of
    (angle_deg, distance_m, quality, new_rotation). Bad bytes are skipped
    one at a time until the stream lines up again.
    """
    out = []
    i, n = 0, len(buf)
    while i + 5 <= n:
        b0, b1 = buf[i], buf[i + 1]
        start = b0 & 1
        if start == ((b0 >> 1) & 1) or not (b1 & 1):      # check bits wrong: out of step
            i += 1
            continue
        angle = (((buf[i + 2] << 8) | b1) >> 1) / 64.0
        dist = ((buf[i + 4] << 8) | buf[i + 3]) / 4000.0
        out.append((angle % 360.0, dist, b0 >> 2, bool(start)))
        i += 5
    return out, buf[i:]


class Lidar(object):
    def __init__(self, port="/dev/rplidar", baud=115200, front_deg=0.0, flip=False):
        self.port, self.baud = port, baud
        self.front_deg = front_deg
        self.flip = flip                # True if the lidar is mounted upside down
        self.lock = threading.Lock()
        self.dist = [0.0] * 360         # metres per whole LIDAR degree, 0 = nothing seen
        self.stamp = [0.0] * 360
        self.rotations = 0
        self.samples = 0
        self.error = None
        self._run = False
        self._thread = None
        self._serial = None

    # ---- lifecycle ----
    def start(self):
        self._serial = serial.Serial(self.port, self.baud, timeout=0.2)
        self._serial.dtr = False        # DTR low lets the A1's motor spin
        self._serial.write(CMD_STOP)
        time.sleep(0.1)
        self._serial.reset_input_buffer()
        self._serial.write(CMD_SCAN)
        head = self._serial.read(len(SCAN_DESCRIPTOR))
        if head != SCAN_DESCRIPTOR:
            self._serial.close()
            raise IOError("lidar did not start scanning (got %r)" % head.hex())
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            if self._serial:
                self._serial.write(CMD_STOP)
                self._serial.close()
        except Exception:
            pass

    def _loop(self):
        buf = b""
        try:
            while self._run:
                buf += self._serial.read(max(5, min(4096, self._serial.in_waiting)))
                samples, buf = parse_samples(buf)
                if not samples:
                    continue
                now = time.time()
                with self.lock:
                    for angle, dist, quality, new_rot in samples:
                        if new_rot:
                            self.rotations += 1
                        self.samples += 1
                        k = int(angle + 0.5) % 360
                        # A zero distance means "no echo": remember that the
                        # direction was looked at and found empty.
                        self.dist[k] = dist if quality > 0 else 0.0
                        self.stamp[k] = now
        except Exception as e:                          # unplugged, etc.
            self.error = str(e)
            self._run = False

    # ---- queries ----
    def alive(self):
        with self.lock:
            return self._run and time.time() - max(self.stamp) < MAX_AGE

    def points(self):
        """Fresh readings as (x, y) metres in the robot frame: x ahead, y left."""
        now = time.time()
        pts = []
        with self.lock:
            for k in range(360):
                d = self.dist[k]
                if d < MIN_RANGE or now - self.stamp[k] > MAX_AGE:
                    continue
                # Lidar angles run clockwise; robot angles counter-clockwise.
                a = math.radians(self.front_deg - k) if not self.flip else math.radians(k - self.front_deg)
                pts.append((d * math.cos(a), d * math.sin(a)))
        return pts

    def clearance(self, half_width=0.18):
        """Free distance (m, from the lidar) straight ahead and straight behind,
        within a lane as wide as the robot. Returns (front, rear); each is None
        when the lidar has no fresh data and inf when the lane is empty."""
        if not self.alive():
            return None, None
        front = rear = float("inf")
        for x, y in self.points():
            if abs(y) <= half_width:
                if x > 0:
                    front = min(front, x)
                else:
                    rear = min(rear, -x)
        return front, rear

    def raw(self):
        """Snapshot of fresh readings by LIDAR degree (for calibration)."""
        now = time.time()
        with self.lock:
            return {k: self.dist[k] for k in range(360)
                    if self.dist[k] >= MIN_RANGE and now - self.stamp[k] <= MAX_AGE}


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/rplidar"
    lidar = Lidar(port).start()
    t0 = time.time()
    time.sleep(4.0)
    raw = lidar.raw()
    secs = time.time() - t0
    print("rotations/s %.1f, samples/s %d, directions with an echo %d/360"
          % (lidar.rotations / secs, lidar.samples / secs, len(raw)))
    print("\nLIDAR angle (its own frame, clockwise)   nearest   median   echoes")
    for a0 in range(0, 360, 15):
        ds = sorted(raw[k] for k in range(a0, a0 + 15) if k in raw)
        if ds:
            bar = "#" * int(min(ds[0], 3.0) * 10)
            print("  %3d-%3d deg   %5.2f m  %5.2f m   %2d   %s" % (a0, a0 + 14, ds[0], ds[len(ds) // 2], len(ds), bar))
        else:
            print("  %3d-%3d deg   (no echo)" % (a0, a0 + 14))
    if raw:
        near = sorted(raw.items(), key=lambda kv: kv[1])[:12]
        print("\n12 nearest echoes (deg: m):", ", ".join("%d: %.2f" % kv for kv in sorted(near)))
    print("\nfull scan (deg:cm):", " ".join("%d:%d" % (k, raw[k] * 100) for k in sorted(raw)))
    lidar.stop()
