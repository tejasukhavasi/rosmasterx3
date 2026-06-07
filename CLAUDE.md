# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Onboard control software for the Yahboom Rosmaster X3 robot (mecanum-wheeled rover, optional 6-DOF arm on X3 PLUS, Ackermann on R2). Targets a Raspberry Pi running Raspberry Pi OS — most of the code **will not run on a dev macOS host** because it depends on:

- `Rosmaster_Lib` — vendor Python library that talks to the robot's MCU over a USB serial port. Not on PyPI; comes preinstalled on Yahboom's Pi image.
- `/dev/camera_depth` and `/dev/camera_usb` — udev symlinks created by `rosmaster/video.rules` (must be copied to `/etc/udev/rules.d/`).
- `gevent`, `flask`, `opencv-python`, `pyzbar` — system Python packages on the Pi image.

There is no `requirements.txt`, no test suite, and no lint config in this repo. Don't invent one.

## Running

On the Pi:

```bash
sudo python3 rosmaster/rosmaster_main.py          # full control + video
sudo python3 rosmaster/rosmaster_main.py debug    # verbose logging
sudo python3 rosmaster/rosmaster_no_control.py    # camera streaming only (no Rosmaster_Lib)
bash rosmaster/start_app.sh                       # the autostart wrapper (sleeps 5s, then sudo-runs main)
bash rosmaster/kill_rosmaster.sh                  # pkill rosmaster_main.py
```

`sudo` is required for GPIO/serial access. The desktop autostart unit is `rosmaster.desktop`.

The `samples/` directory holds standalone Jupyter notebooks (`*.ipynb`) that exercise `Rosmaster_Lib` directly (motors, servos, RGB, arm). They are documentation/demos, not part of the app.

## Architecture

**`rosmaster_main.py` is the whole app** (~860 lines, single file). It runs three concurrent things:

1. **Flask + gevent WSGI on `0.0.0.0:6500`** — serves `/` (HTML viewer), `/video_feed` (MJPEG stream), `/init` (kicks off TCP init). Templates in `templates/`.
2. **Raw TCP server on `<host-ip>:6000`** (`start_tcp_server`) — accepts a single client (the companion mobile app) and parses an ASCII framed protocol. This is the actual control channel; HTTP is just for video.
3. **Background threads** — `create_receive_threading()` (serial RX inside `Rosmaster_Lib`), `task_mecanum` (pushes per-wheel speed back to client every 350 ms while in `MecanumWheel` mode), and an arm "dance" thread spawned ad-hoc.

The HTTP server and TCP server bind to different ports and live in parallel; the mobile app uses both simultaneously.

### TCP protocol (the thing to read before touching `parse_data`)

Frames are ASCII text: `$<carType><cmd><len><payload...><checksum>#`, each field two hex digits. `parse_data()` (`rosmaster_main.py:309`) validates length and an 8-bit additive checksum (sum of all hex-pair bytes between `$` and the checksum), then dispatches on `cmd`. The same encoding is used for outgoing `return_*` helpers (`rosmaster_main.py:81-223`).

Command bytes you'll touch most often:
- `0F` enter UI page (sets `g_mode` to `Home` / `Standard` / `MecanumWheel`)
- `10` joystick (x,y) → `set_car_motion`
- `15` directional button → `ctrl_car_<type>(state)` dispatcher
- `11`/`12` PWM / serial-bus servo, `13` beep, `30`–`32` RGB strip
- `20`/`21` per-wheel mecanum control
- `40`–`43` arm calibration / pose, `50`–`53` Ackermann front-wheel angle

### Car-type dispatch

`g_car_type` is set from the incoming frame and forwarded to `Rosmaster_Lib` via `set_car_type`. Four type-specific control wrappers (`ctrl_car_x3`, `ctrl_car_x3plus`, `ctrl_car_r2`, `ctrl_car_x1`, `rosmaster_main.py:232-281`) translate the same button events into different speed/angle commands. R2 (Ackermann) is the odd one — it needs auto-recenter on stop and steering-angle mapping; the others are differential/mecanum and share shape.

When the type switches to X3 PLUS, a *second* camera is opened on the fly: `g_camera` is the depth camera (`/dev/camera_depth`, id `0x50`), `g_camera_usb` the USB cam (`/dev/camera_usb`, id `0x51`). `cmd 18` toggles which one feeds `/video_feed` via `g_camera_type`. The `0x50`/`0x51` magic numbers double as both `video_id` arguments and `TYPE_*` constants in `camera_rosmaster.py` — keep them in sync if you touch either.

### State machine

`g_mode` ∈ {`Home`, `Standard`, `MecanumWheel`} gates what `mode_handle()` (the video generator) yields and whether `thread_mecanum` pushes telemetry. It's set by `cmd 0F` and reset to `Home` on TCP disconnect.

### `rosmaster_no_control.py`

A stripped-down variant that imports only the camera class — useful for camera bring-up when `Rosmaster_Lib` / serial isn't available. Same Flask routes, same TCP scaffolding, but `parse_data` is commented out. Keep it in sync with `rosmaster_main.py` if you change the video pipeline.

### `wifi_rosmaster.py`

QR-code WiFi onboarding (scan a phone-generated QR with the camera, then `nmcli dev wifi connect`). Most of the GPIO/key-scan code is **commented out** in this checkout — the live code path is just `get_ip_address()` (used by `init_tcp_socket` in main). Don't assume the key/LED logic runs.

## Conventions worth knowing

- Comments and `print()` strings are bilingual Chinese/English. Preserve both when editing existing lines.
- Globals everywhere (`g_*`). Don't refactor away unless the user asks — the mobile-app protocol depends on the exact dispatch shape.
- Angles for arm joints 2/3/4 are mirrored: the wire protocol uses `180 - angle` vs. what `Rosmaster_Lib` expects. See `return_arm_angle` and `cmd 12` handler.
- The checksum is an 8-bit sum (mod 256) of byte values, *not* XOR. Easy to get wrong.
