#!/usr/bin/env python3
# coding=utf-8
"""
Find which lidar angle points straight ahead of the robot.

The robot scans, drives forward about 12 cm, scans again and drives back.
Things ahead got closer, things behind got farther, things to the side
barely changed; fitting that pattern gives the lidar angle of "forward".

Needs about 50 cm of clear floor in front of the robot. Run from the folder
that holds lidar.py:

    python3 lidar_calibrate.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.getcwd())

SPEED = 0.10        # m/s
DRIVE_S = 1.2       # seconds forward (about 12 cm), then the same back
MIN_CLEAR = 0.35    # m. Refuse to move if anything is closer than this, in any direction


def fit_front(before, after):
    """before/after: {lidar_deg: metres}. Returns (front_deg, moved_m, used, rms_m).

    Moving s metres toward lidar angle F changes the range at angle k by about
    -s*cos(k - F) = -(a*cos k + b*sin k). Solve for a, b by least squares,
    drop the worst-fitting points (edges of objects jump about) and solve again.
    """
    pts = [(k, after[k] - before[k]) for k in before
           if k in after and before[k] < 4.0 and abs(after[k] - before[k]) < 0.3]

    def solve(points):
        scc = scs = sss = scd = ssd = 0.0
        for k, d in points:
            c, s = math.cos(math.radians(k)), math.sin(math.radians(k))
            scc += c * c
            scs += c * s
            sss += s * s
            scd += c * -d
            ssd += s * -d
        det = scc * sss - scs * scs
        if abs(det) < 1e-9:
            return None
        return (scd * sss - ssd * scs) / det, (ssd * scc - scd * scs) / det

    def resid(ab, k, d):
        return d + ab[0] * math.cos(math.radians(k)) + ab[1] * math.sin(math.radians(k))

    ab = solve(pts)
    if ab is None or len(pts) < 20:
        return None
    for _ in range(2):
        keep = [(k, d) for k, d in pts if abs(resid(ab, k, d)) < 0.04]
        if len(keep) < 20:
            break
        pts, ab = keep, solve(keep) or ab
    rms = math.sqrt(sum(resid(ab, k, d) ** 2 for k, d in pts) / len(pts))
    return math.degrees(math.atan2(ab[1], ab[0])) % 360.0, math.hypot(*ab), len(pts), rms


def steady_scan(lidar, seconds=1.5):
    time.sleep(seconds)
    return lidar.raw()


def main():
    from lidar import Lidar
    from Rosmaster_Lib import Rosmaster

    lidar = Lidar(sys.argv[1] if len(sys.argv) > 1 else "/dev/rplidar").start()
    bot = Rosmaster()
    try:
        before = steady_scan(lidar)
        if len(before) < 60:
            print("RESULT: failed, only %d directions returned an echo" % len(before))
            return
        k, d = min(before.items(), key=lambda kv: kv[1])
        if d < MIN_CLEAR:
            print("RESULT: not moving, something is %.2f m away at lidar angle %d. "
                  "Give the robot 50 cm of space all round and try again." % (d, k))
            return
        print("driving forward %.0f cm..." % (SPEED * DRIVE_S * 100), flush=True)
        t0 = time.time()
        while time.time() - t0 < DRIVE_S:
            bot.set_car_motion(SPEED, 0, 0)
            time.sleep(0.05)
        bot.set_car_motion(0, 0, 0)
        after = steady_scan(lidar)
        print("driving back...", flush=True)
        t0 = time.time()
        while time.time() - t0 < DRIVE_S:
            bot.set_car_motion(-SPEED, 0, 0)
            time.sleep(0.05)
        bot.set_car_motion(0, 0, 0)

        fit = fit_front(before, after)
        if fit is None:
            print("RESULT: failed, too few matching directions between the two scans")
        else:
            front, moved, used, rms = fit
            ok = 0.04 <= moved <= 0.30 and rms < 0.035
            print("RESULT: front_deg=%.0f moved=%.2f m (expected ~%.2f) points=%d fit_error=%.3f m %s"
                  % (front, moved, SPEED * DRIVE_S, used, rms, "OK" if ok else "DOUBTFUL"))
        print("before (deg:cm):", " ".join("%d:%d" % (a, before[a] * 100) for a in sorted(before)))
        print("after  (deg:cm):", " ".join("%d:%d" % (a, after[a] * 100) for a in sorted(after)))
    finally:
        bot.set_car_motion(0, 0, 0)
        lidar.stop()


if __name__ == "__main__":
    main()
