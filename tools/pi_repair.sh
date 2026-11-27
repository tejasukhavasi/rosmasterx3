#!/bin/bash
# Runs ON THE ROBOT (tools/pi.sh repair / repair2 send it there).
#
#   step 1: find out what is damaged, schedule a full disk check, reboot.
#   step 2: (after the reboot) repair the package database, reinstall the
#           compiler, confirm it works.
#
# Every change is described first and waits for "y".

ROOTDEV=$(findmnt -n -o SOURCE /)
CMDLINE=/boot/firmware/cmdline.txt
say() { echo; echo "===== $* ====="; }
ask() { read -r -p ">>> $* [y/N] " a; [ "$a" = "y" ] || [ "$a" = "Y" ]; }

bad_lists() {
    # dpkg's per-package file lists: every line must be an absolute path and the
    # file must end with a newline. Anything else is damage.
    for f in /var/lib/dpkg/info/*.list; do
        [ -s "$f" ] || continue
        if [ "$(tail -c1 "$f" | od -An -c | tr -d ' ')" != '\n' ] || LC_ALL=C grep -qav '^/' "$f"; then
            echo "$f"
        fi
    done
}

if [ "$1" = "1" ]; then
    say "what is damaged (nothing is changed yet)"
    echo "root filesystem: $ROOTDEV"
    echo "disk errors logged since boot: $(sudo dmesg | grep -c 'EXT4-fs error')"
    for ino in $(sudo dmesg | grep -o 'EXT4-fs error.*inode #[0-9]*' | grep -o '#[0-9]*' | tr -d '#' | sort -u); do
        echo "damaged entry (inode $ino) is the folder/file:"
        sudo debugfs -R "ncheck $ino" "$ROOTDEV" 2>/dev/null | tail -n +2
    done
    echo "damaged package records:"; bad_lists
    sudo tune2fs -l "$ROOTDEV" 2>/dev/null | grep -iE "state|error|mount count|last checked"

    say "schedule a full disk check"
    echo "Current boot line:"; cat "$CMDLINE"
    echo "Change: add 'fsck.mode=force' to that line (a copy is kept as cmdline.txt.before-repair),"
    echo "so the next boot checks and repairs the whole disk. It already has fsck.repair=yes."
    echo "The check may remove files it cannot repair; step 2 reinstalls the ones that belong to packages."
    if ask "Make this change and reboot the robot now?"; then
        sudo cp -n "$CMDLINE" /boot/firmware/cmdline.txt.before-repair
        grep -q "fsck.mode=force" "$CMDLINE" || sudo sed -i '1 s/$/ fsck.mode=force/' "$CMDLINE"
        echo "New boot line:"; cat "$CMDLINE"
        echo "Rebooting. The disk check can take several minutes; then run: tools/pi.sh repair2"
        sudo reboot
    else
        echo "Nothing changed."
    fi
    exit 0
fi

if [ "$1" = "2" ]; then
    say "result of the disk check"
    if grep -q "fsck.mode=force" "$CMDLINE"; then
        sudo sed -i 's/ fsck.mode=force//' "$CMDLINE"
        echo "removed fsck.mode=force from the boot line again:"; cat "$CMDLINE"
    fi
    sudo journalctl -b --no-pager 2>/dev/null | grep -iE "fsck|e2fsck" | tail -15
    cat /run/initramfs/fsck.log 2>/dev/null | tail -15
    sudo tune2fs -l "$ROOTDEV" 2>/dev/null | grep -iE "state|last checked"
    echo "disk errors logged since this boot: $(sudo dmesg | grep -c 'EXT4-fs error')"
    echo "lost+found entries: $(sudo ls /lost+found 2>/dev/null | wc -l)"

    say "package database"
    BAD=$(bad_lists)
    if [ -z "$BAD" ]; then
        echo "no damaged package records"
    else
        echo "Damaged records:"; echo "$BAD"
        PKGS=$(for f in $BAD; do basename "$f" .list; done | tr '\n' ' ')
        echo "Change: move these files to /var/backups/dpkg-bad-lists/ and reinstall: $PKGS"
        echo "(reinstalling a package writes a fresh record)"
        if ask "Do it?"; then
            sudo mkdir -p /var/backups/dpkg-bad-lists && sudo mv $BAD /var/backups/dpkg-bad-lists/
            sudo dpkg --configure -a 2>&1 | tail -5
            sudo apt-get update -qq
            sudo apt-get install -y --reinstall $PKGS 2>&1 | tail -6
        else
            echo "Skipped; package installs will keep failing."; exit 1
        fi
    fi
    sudo dpkg --audit 2>&1 | head -20

    say "compiler"
    test_cc() {
        echo 'int main(void){return 0;}' > /tmp/cc_test.c && cc /tmp/cc_test.c -o /tmp/cc_test_c >/tmp/cc_test.log 2>&1 &&
        printf '#include <vector>\nint main(){std::vector<int> v(3); return (int)v.size()-3;}\n' > /tmp/cc_test.cpp &&
        c++ /tmp/cc_test.cpp -o /tmp/cc_test_cpp >>/tmp/cc_test.log 2>&1 && /tmp/cc_test_cpp
    }
    if test_cc; then
        echo "C and C++ compilers work"
    else
        tail -4 /tmp/cc_test.log
        echo "Change: reinstall the compiler packages (cpp-12 gcc-12 g++-12 and their libraries)."
        if ask "Do it?"; then
            sudo apt-get install -y --reinstall cpp-12 gcc-12 g++-12 libgcc-12-dev libstdc++-12-dev libc6-dev \
                binutils binutils-aarch64-linux-gnu libisl23 libmpc3 libmpfr6 libgmp10 2>&1 | tail -6
        fi
        if test_cc; then echo "compilers work now"; else echo "RESULT: compiler still broken"; tail -6 /tmp/cc_test.log; fi
    fi

    say "other damage"
    python3 -c "import cv2; print('user OpenCV ok', cv2.__version__)" 2>&1 | tail -1
    ldd /lib/aarch64-linux-gnu/libopencv_text.so.406 >/dev/null 2>&1 && echo "system OpenCV library ok" || echo "system OpenCV library still damaged (only matters with sudo)"
    echo "disk errors logged since this boot: $(sudo dmesg | grep -c 'EXT4-fs error')"
    echo "RESULT: step 2 finished"
    exit 0
fi
echo "usage: pi_repair.sh 1|2"
