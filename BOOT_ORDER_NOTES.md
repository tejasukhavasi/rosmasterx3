# Raspberry Pi 5 boot order — note for this robot

**Date:** 2026-11-26

## What changed

This robot's Raspberry Pi 5 had its bootloader **boot order** changed from the
factory default to **USB first, then SD card**:

| | BOOT_ORDER | Meaning (read right-to-left) |
|---|---|---|
| Factory (Yahboom) | `0xf461` | SD card → NVMe → USB |
| **Now** | `0xf614` | **USB → SD card → NVMe** |

The setting is stored in the Pi's bootloader EEPROM (on the board itself), **not**
on the SD card or any drive. The robot's Yahboom SD card image is unchanged.

## Why

So the same Pi can boot a USB SSD that runs a separate project (a local Arlo
camera hub) when that SSD is attached, without removing the robot's SD card
(which is hard to reach in the assembly).

## What this means in practice

- **SSD unplugged:** the Pi finds nothing on USB and falls back to the SD card,
  so the robot boots the Yahboom image normally (a few seconds slower at most).
- **SSD plugged in:** the Pi boots from the SSD (the Arlo hub), **not** Yahboom.

So: **SSD out = robot. SSD in = Arlo hub.** Keep the SSD unplugged for normal
robot use.

## Check / change it

```bash
# View current boot order
rpi-eeprom-config | grep BOOT_ORDER

# To revert to the factory SD-first order (optional; not required):
sudo rpi-eeprom-config --edit        # set BOOT_ORDER=0xf461, save, then:
sudo reboot
```

Keeping `0xf614` is intentional and harmless for the robot.
