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
#   tools/pi.sh depth-setup  build Orbbec's depth-camera SDK on the robot and test it (10+ minutes,
#                          needs internet on the robot). Log in pi_reports/depth_setup.txt
#   tools/pi.sh backup   copy the robot's own files (Yahboom app, motor library, device rules) to pi_backup/
#   tools/pi.sh repair   step 1 of the SD-card repair: report the damage, schedule a disk check, reboot
#   tools/pi.sh repair2  step 2, after the reboot: fix the package database and the compiler
#   tools/pi.sh health   check the robot's SD card for corrupted system files (several minutes)
#   tools/pi.sh depth-snap   save a colour picture and a depth picture taken together (pi_reports/depth_snap/)
#   tools/pi.sh depth-test   run just the depth test (seconds). Summary in pi_reports/depth_test.txt
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
    scp -q "$REPO/rosmaster/follow_bot.py" "$REPO/rosmaster/lidar.py" "$REPO/rosmaster/orbbec_camera.py" "$PI:$REMOTE_DIR/" && echo "Copied follow_bot.py, lidar.py and orbbec_camera.py to $PI"
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
backup)
    # Copy the robot-specific files to the Mac (read-only on the robot). Not a full disk image.
    B="$REPO/pi_backup"; mkdir -p "$B"
    ssh "$PI" 'cd / && tar czf - --ignore-failed-read --warning=no-file-changed \
        home/pi/Rosmaster etc/udev/rules.d boot/firmware/config.txt boot/firmware/cmdline.txt \
        usr/local/lib/python3.11/dist-packages \
        $(cd / && ls -d home/pi/*.sh home/pi/*.py home/pi/Version.txt home/pi/.bashrc home/pi/.config/autostart 2>/dev/null) 2>/tmp/backup_errors.txt; \
        echo "--- tar errors:" >&2; cat /tmp/backup_errors.txt >&2' > "$B/robot-files.tgz"
    ssh "$PI" 'pip list 2>/dev/null; echo ---; dpkg --get-selections 2>&1 | head -3000; echo ---; ls -la ~ ; du -sh ~/* 2>/dev/null' > "$B/inventory.txt"
    ls -lh "$B"; echo "Files in archive: $(tar tzf "$B/robot-files.tgz" | wc -l)"
    ;;
repair|repair2)
    n=1; [ "$1" = "repair2" ] && n=2
    scp -q "$REPO/tools/pi_repair.sh" "$PI:/tmp/" || exit 1
    ssh "$PI" 'pkill -INT -f "python3.*[f]ollow_bot" && sleep 2'
    ssh -tt "$PI" "bash /tmp/pi_repair.sh $n" 2>&1 | tee "$OUT/repair$n.txt"
    echo "Saved pi_reports/repair$n.txt"
    ;;
health)
    # Looks for a failing SD card / power problems: corrupted system files, disk and voltage errors.
    ssh -tt "$PI" 'echo "== power/throttle (0x0 is good)"; vcgencmd get_throttled; vcgencmd measure_temp
echo "== disk"; df -h / | tail -1; mount | grep " / "
echo "== kernel errors"; sudo dmesg | grep -iE "mmc|ext4|i/o error|under-?voltage|segfault" | tail -25
echo "== corrupted packaged files (this takes several minutes)"
dpkg -s debsums >/dev/null 2>&1 || sudo apt-get install -y -qq debsums
sudo debsums -s 2>&1 | head -200; echo "== done"' 2>&1 | tee "$OUT/health.txt"
    echo "Saved pi_reports/health.txt"
    ;;
depth-snap)
    scp -q "$REPO/tools/depth_snap.py" "$PI:/tmp/" || exit 1
    ssh "$PI" 'pkill -INT -f "python3.*[f]ollow_bot" && sleep 2'
    ssh "$PI" 'PYTHONPATH=$HOME/pyorbbecsdk/install/lib python3 -u /tmp/depth_snap.py 2>&1' | grep "^##\|Traceback\|Error" | cut -c1-200
    mkdir -p "$OUT/depth_snap" && scp -q "$PI:/tmp/depth_snap/*" "$OUT/depth_snap/" && echo "Saved to pi_reports/depth_snap/"
    ;;
depth-test)
    scp -q "$REPO/tools/depth_test.py" "$PI:/tmp/" || exit 1
    ssh "$PI" 'pkill -INT -f "python3.*[f]ollow_bot" && sleep 2'
    ssh "$PI" 'rm -f /tmp/depth.png; PYTHONPATH=$HOME/pyorbbecsdk/install/lib python3 -u /tmp/depth_test.py 2>&1' > "$OUT/depth_test_full.txt"
    grep "^##\|Traceback\|Error\|error\]" "$OUT/depth_test_full.txt" | grep -v usb_request_cancel | cut -c1-200 | tee "$OUT/depth_test.txt"
    scp -q "$PI:/tmp/depth.png" "$OUT/" 2>/dev/null && echo "Saved pi_reports/depth.png"
    ;;
depth-setup)
    scp -q "$REPO/tools/depth_setup.sh" "$REPO/tools/depth_test.py" "$PI:/tmp/" || exit 1
    ssh -tt "$PI" "bash /tmp/depth_setup.sh" 2>&1 | tee "$OUT/depth_setup.txt"
    scp -q "$PI:/tmp/depth.png" "$OUT/" 2>/dev/null && echo "Saved pi_reports/depth.png"
    echo "Saved pi_reports/depth_setup.txt"
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
    sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
