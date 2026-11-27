#!/usr/bin/env python3
"""Work out which lidar is on /dev/rplidar and how to talk to it.

Phase 1 only listens at each common baud rate (some lidars stream all the
time). Phase 2 sends the standard "who are you" request of the RPLIDAR and
YDLIDAR families. Nothing here changes any setting on the lidar.

    python3 lidar_probe.py [port]
"""
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/rplidar"
BAUDS = [115200, 230400, 460800, 1000000, 256000, 512000, 128000, 153600]


def open_port(baud):
    s = serial.Serial(PORT, baud, timeout=0.4)
    s.dtr = False          # on RPLIDAR A-series adapters, DTR low lets the motor run
    return s


def spacing(data, marker):
    """Most common distance between repeats of marker, and how often it occurs."""
    pos, i = [], data.find(marker)
    while i >= 0:
        pos.append(i)
        i = data.find(marker, i + 1)
    gaps = [b - a for a, b in zip(pos, pos[1:])]
    if not gaps:
        return len(pos), None, 0
    best = max(set(gaps), key=gaps.count)
    return len(pos), best, gaps.count(best)


print("port:", PORT)
print("\n== Phase 1: listening only")
for baud in BAUDS:
    try:
        s = open_port(baud)
    except Exception as e:
        print("%8d baud: cannot open (%s)" % (baud, e))
        continue
    try:
        s.reset_input_buffer()
        time.sleep(0.5)
        data = s.read(6000)
        line = "%8d baud: %5d bytes" % (baud, len(data))
        for name, marker in (("LD/MS200-style 54 2C", b"\x54\x2c"), ("YDLIDAR-style AA 55", b"\xaa\x55")):
            n, gap, hits = spacing(data, marker)
            if n >= 5:
                line += " | %s x%d, usual gap %s bytes (%d times)" % (name, n, gap, hits)
        print(line)
        if data:
            print("           first bytes:", data[:40].hex(" "))
    finally:
        s.close()

print("\n== Phase 2: asking for device info")
for baud in BAUDS:
    try:
        s = open_port(baud)
    except Exception:
        continue
    try:
        for family, stop, ask in (("RPLIDAR", b"\xa5\x25", b"\xa5\x50"), ("YDLIDAR", b"\xa5\x65", b"\xa5\x90")):
            s.write(stop)
            time.sleep(0.15)
            s.reset_input_buffer()
            s.write(ask)
            r = s.read(200)
            i = r.find(b"\xa5\x5a")
            if i >= 0 and len(r) - i >= 27:
                body = r[i + 7:i + 27]
                print("%8d baud: %s info reply! descriptor %s" % (baud, family, r[i:i + 7].hex(" ")))
                print("           model byte 0x%02x, firmware %d.%d, hardware %d, serial %s"
                      % (body[0], body[2], body[1], body[3], body[4:].hex()))
                if family == "RPLIDAR":
                    s.write(b"\xa5\x52")                     # health
                    print("           health reply:", s.read(10).hex(" "))
                else:
                    s.write(b"\xa5\x60")                     # resume scanning
            elif r:
                print("%8d baud: %s request -> %d bytes, no info header: %s" % (baud, family, len(r), r[:24].hex(" ")))
            else:
                print("%8d baud: %s request -> silence" % (baud, family))
    finally:
        s.close()
print("\ndone")
