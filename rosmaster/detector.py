#!/usr/bin/env python3
# coding=utf-8
"""
Object finder for the follow bot: a small YOLO model (ultralytics, already on
Yahboom's Pi image) that picks out people and about 80 kinds of everyday
things (dog, cat, ball, teddy bear, chair, ...) in the camera picture.

It runs in its own thread on the newest frame, a few times a second, so the
follow bot's control loop never waits for it.

The model file yolov8n.pt (6 MB) is downloaded by ultralytics the first time,
so the robot needs internet once.
"""
import os
import threading
import time


class YoloDetector(object):
    def __init__(self, weights=None, imgsz=416, conf=0.30, threads=3):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(threads)
        self.model = YOLO(weights or os.path.expanduser("~/yolov8n.pt"))
        self.names = self.model.names
        self.imgsz, self.conf = imgsz, conf
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self._frame = None
        self._frame_seq = 0
        self._done_seq = 0
        self._dets = []
        self._dets_t = 0.0
        self.seq = 0                 # goes up by one with every finished result
        self.fps = 0.0
        self.error = None
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Hand over the newest camera frame (it is not modified)."""
        with self.cond:
            self._frame = frame
            self._frame_seq += 1
            self.cond.notify()

    def results(self):
        """(detections, time they were made, sequence number).

        Each detection is {"cls": name, "conf": 0..1, "box": (x1, y1, x2, y2)} in
        pixels of the frame that was submitted."""
        with self.lock:
            return self._dets, self._dets_t, self.seq

    def stop(self):
        self._run = False
        with self.cond:
            self.cond.notify()

    def _loop(self):
        while self._run:
            with self.cond:
                while self._run and self._frame_seq == self._done_seq:
                    self.cond.wait(0.5)
                if not self._run:
                    return
                frame, seq = self._frame, self._frame_seq
            t0 = time.time()
            try:
                r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                dets = [{"cls": self.names[int(c)], "conf": float(p), "box": tuple(float(v) for v in b)}
                        for b, c, p in zip(xyxy, cls, conf)]
            except Exception as e:                      # keep going; the bot shows the error
                self.error = str(e).splitlines()[0] if str(e) else repr(e)
                dets = []
                time.sleep(0.5)
            dt = time.time() - t0
            with self.lock:
                self._dets, self._dets_t, self._done_seq = dets, t0, seq
                self.seq += 1
                self.fps = 0.8 * self.fps + 0.2 / max(dt, 1e-3) if self.fps else 1.0 / max(dt, 1e-3)
