#!/usr/bin/env python3
"""Runs ON THE ROBOT, from the folder holding orbbec_camera.py and lidar.py.

Records what the robot's sensors see for a few seconds, robot standing still:
colour picture, depth picture (16-bit PNG, millimetres, lined up with the
colour) and a lidar scan, several times a second. Used to develop and test
the following logic on real data.

    python3 record.py [seconds=12] [frames_per_second=6]
"""
import json
import os
import shutil
import sys
import time

import cv2

sys.path.insert(0, os.getcwd())
from orbbec_camera import OrbbecCamera          # noqa: E402

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
rate = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
OUT = "/tmp/rec"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

cam = OrbbecCamera()
if not cam.isOpened():
    print("## RESULT: camera failed:", cam.error)
    sys.exit(1)
print("## camera ok, depth %s, lined up %s" % (cam.with_depth, cam.aligned), flush=True)

lidar = None
try:
    if os.path.exists("/dev/rplidar"):
        from lidar import Lidar
        lidar = Lidar("/dev/rplidar").start()
except Exception as e:
    print("## lidar not available:", e, flush=True)

time.sleep(1.5)                                  # exposure and depth settle
print("## RECORDING for %.0f s - go!" % secs, flush=True)
t0, n, meta = time.time(), 0, []
while time.time() - t0 < secs:
    ok, color = cam.get_frame()
    if not ok:
        continue
    depth = cam.depth()
    t = time.time() - t0
    if t < n / rate:
        continue
    cv2.imwrite("%s/%03d-color.jpg" % (OUT, n), color, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if depth is not None:
        cv2.imwrite("%s/%03d-depth.png" % (OUT, n), depth)
    meta.append({"n": n, "t": round(t, 3), "depth": depth is not None,
                 "lidar": {str(k): round(v, 3) for k, v in lidar.raw().items()} if lidar else None})
    n += 1
json.dump(meta, open(OUT + "/meta.json", "w"))
cam.close()
if lidar:
    lidar.stop()
print("## RESULT: recorded %d frames (%d with depth)" % (n, sum(m["depth"] for m in meta)))
