#!/bin/bash
# Helper for working on the robot from the Mac. Run from anywhere:
#
#   tools/pi.sh deploy   copy follow_bot.py and lidar.py to the robot
#   tools/pi.sh run      deploy, then run the follow bot; its log is saved in pi_reports/follow.log
#   tools/pi.sh snap     save a camera frame from the running follow bot to pi_reports/
#   tools/pi.sh diag     collect hardware info from the robot into pi_reports/diag.txt
#   tools/pi.sh lidar    identify the lidar on /dev/rplidar into pi_reports/lidar.txt
#   tools/pi.sh calibrate  find the lidar's forward direction: the robot drives 12 cm forward and back,
#                          so give it 50 cm of clear floor all round. Saved in pi_reports/calibrate.txt
#   tools/pi.sh scan     4 seconds of lidar readings into pi_reports/scan.txt (stop the follow bot first)
#
# Extra arguments after "run" go to follow_bot.py, e.g.  tools/pi.sh run --no-motors
# Set PI=pi@<ip> if yahboom.local does not resolve.
# Tip: run  ssh-copy-id pi@yahboom.local  once so it stops asking for the password.

PI="${PI:-pi@yahboom.local}"
REMOTE_DIR="/home/pi/Rosmaster/rosmaster"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/pi_reports"
mkdir -p "$OUT"

deploy() {
    scp -q "$REPO/rosmaster/follow_bot.py" "$REPO/rosmaster/lidar.py" "$PI:$REMOTE_DIR/" && echo "Copied follow_bot.py and lidar.py to $PI"
}

case "$1" in
deploy)
    deploy
    ;;
run)
    shift
    deploy || exit 1
    # An earlier copy still running (e.g. in another SSH window) holds the camera.
    # Ask it to stop politely (SIGINT) so it can stop the wheels; never kill -9.
    ssh "$PI" 'pkill -INT -f "python3.*[f]ollow_bot" && echo "Stopped a follow bot that was already running" && sleep 2'
    echo "Running on the robot. Ctrl+C here stops it (and the wheels). Log: pi_reports/follow.log"
    # -tt so Ctrl+C reaches the robot and follow_bot.py can stop the motors cleanly.
    ssh -tt "$PI" "cd $REMOTE_DIR && bash kill_rosmaster.sh >/dev/null 2>&1; rm -rf /tmp/follow_frames; python3 -u follow_bot.py --save-frames /tmp/follow_frames $*" 2>&1 | tee "$OUT/follow.log"
    # Bring back the pictures of whatever it took for the toy during this run.
    mkdir -p "$OUT/frames" && scp -q "$PI:/tmp/follow_frames/*.jpg" "$OUT/frames/" 2>/dev/null && echo "Saved pictures to pi_reports/frames/"
    ;;
snap)
    f="$OUT/snap-$(date +%H%M%S).jpg"
    curl -s -m 8 -o "$f" "http://${PI#*@}:6501/snapshot.jpg" && echo "Saved $f" || echo "Could not fetch a frame. Is follow_bot.py running?"
    ;;
calibrate)
    deploy || exit 1
    scp -q "$REPO/tools/lidar_calibrate.py" "$PI:/tmp/" || exit 1
    ssh "$PI" 'pkill -INT -f "python3.*[f]ollow_bot" && sleep 2'
    ssh -tt "$PI" "cd $REMOTE_DIR && bash kill_rosmaster.sh >/dev/null 2>&1; python3 -u /tmp/lidar_calibrate.py" 2>&1 | tee "$OUT/calibrate.txt"
    echo "Saved pi_reports/calibrate.txt"
    ;;
scan)
    deploy || exit 1
    ssh "$PI" "cd $REMOTE_DIR && python3 -u lidar.py" 2>&1 | tee "$OUT/scan.txt"
    echo "Saved pi_reports/scan.txt"
    ;;
lidar)
    scp -q "$REPO/tools/lidar_probe.py" "$PI:/tmp/" && ssh "$PI" "python3 -u /tmp/lidar_probe.py" 2>&1 | tee "$OUT/lidar.txt"
    echo "Saved pi_reports/lidar.txt"
    ;;
diag)
    ssh "$PI" 'bash -s' > "$OUT/diag.txt" 2>&1 <<'REMOTE'
echo "== serial devices"; ls -l /dev/serial/by-id/ /dev/myserial /dev/rplidar /dev/ydlidar /dev/ttyUSB* /dev/ttyACM* /dev/ttyAMA* 2>&1
echo "== lsusb"; lsusb
echo "== udev rules"; ls /etc/udev/rules.d/; grep -h -iE "lidar|serial|ttyUSB" /etc/udev/rules.d/* 2>/dev/null
LIB=$(python3 -c "import Rosmaster_Lib; print(Rosmaster_Lib.__file__)" 2>&1)
echo "== Rosmaster_Lib: $LIB"   # may live inside a zipped .egg, so ask Python instead of grep
python3 -c "
import inspect, Rosmaster_Lib
R = Rosmaster_Lib.Rosmaster
print('init:', inspect.signature(R.__init__))
print('lidar-like methods:', [n for n in dir(R) if any(k in n.lower() for k in ('lidar', 'laser', 'scan'))] or '(none)')
print('all methods:', [n for n in dir(R) if not n.startswith('_')])
" 2>&1
echo "== python"; python3 --version; python3 -c "import cv2; print('cv2', cv2.__version__, cv2.__file__)" 2>&1
echo "== lidar-related files in home"; ls ~ ; find ~ -maxdepth 4 -iname "*lidar*" 2>/dev/null | head -20
echo "== recent tty kernel messages"; dmesg 2>/dev/null | grep -iE "tty|cp210|ch34" | tail -15
echo "== system"; uname -a; cat /etc/os-release | head -3
REMOTE
    echo "Saved pi_reports/diag.txt"
    ;;
*)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
