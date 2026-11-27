#!/bin/bash
# Runs ON THE ROBOT (tools/pi.sh depth-setup sends it there).
# Builds Orbbec's Python SDK (v1.x "main" branch, the one that still supports
# the Astra Pro Plus) into ~/pyorbbecsdk, then runs a depth test.
# Safe to run again: finished steps are skipped.

SDK="$HOME/pyorbbecsdk"
step() { echo; echo "===== $* ====="; }

step "system info"
uname -m; python3 --version; cmake --version 2>/dev/null | head -1; g++ --version 2>/dev/null | head -1
lsusb | grep -i orbbec

step "build tools (apt)"
need=""
for p in cmake g++ git python3-dev python3-venv; do dpkg -s "$p" >/dev/null 2>&1 || need="$need $p"; done
if [ -n "$need" ]; then
    echo "installing:$need"
    sudo apt-get update -qq && sudo apt-get install -y $need || { echo "FAILED: apt install"; exit 1; }
else
    echo "all present"
fi

step "compiler check"
# The robot's SD card has shown corrupted system files before (OpenCV, and the C
# compiler crashing on a one-line program). Test the compilers and reinstall
# their packages if they are broken.
compilers_ok() {
    echo 'int main(void){return 0;}' > /tmp/cc_test.c && cc /tmp/cc_test.c -o /tmp/cc_test_c >/tmp/cc_test.log 2>&1 &&
    printf '#include <vector>\nint main(){std::vector<int> v(3); return (int)v.size()-3;}\n' > /tmp/cc_test.cpp &&
    c++ /tmp/cc_test.cpp -o /tmp/cc_test_cpp >>/tmp/cc_test.log 2>&1 && /tmp/cc_test_cpp
}
if compilers_ok; then
    echo "C and C++ compilers work"
else
    echo "compiler is broken:"; tail -5 /tmp/cc_test.log
    echo "reinstalling the compiler packages..."
    sudo apt-get update -qq
    sudo apt-get install -y --reinstall cpp-12 gcc-12 g++-12 libgcc-12-dev libstdc++-12-dev libc6-dev binutils binutils-aarch64-linux-gnu libisl23 libmpc3 libmpfr6 libgmp10 2>&1 | tail -8
    if compilers_ok; then
        echo "compilers work after reinstall"
        rm -rf "$SDK/build"
    else
        echo "FAILED: compiler still broken after reinstall:"; tail -8 /tmp/cc_test.log
        exit 1
    fi
fi

step "get the SDK source"
if [ ! -d "$SDK/.git" ]; then
    git clone --depth 1 -b main https://github.com/orbbec/pyorbbecsdk.git "$SDK" || { echo "FAILED: git clone"; exit 1; }
fi
cd "$SDK" || exit 1
git log --oneline -1; ls; echo "--- bundled SDK libraries:"; ls sdk/lib 2>/dev/null; ls sdk/lib/* 2>/dev/null | head -20

step "build"
if ls "$SDK"/install/lib/pyorbbecsdk*.so >/dev/null 2>&1; then
    echo "already built: $(ls "$SDK"/install/lib/pyorbbecsdk*.so)"
else
    python3 -m venv venv && . venv/bin/activate || { echo "FAILED: venv"; exit 1; }
    pip install -q pybind11==2.11.0 pybind11-global==2.11.0 wheel || { echo "FAILED: pip pybind11"; exit 1; }
    rm -rf build && mkdir -p build && cd build
    cmake -Dpybind11_DIR="$(pybind11-config --cmakedir)" .. || { echo "FAILED: cmake"; exit 1; }
    make -j4 2>&1 | tail -40
    make install 2>&1 | tail -15
    deactivate
    cd "$SDK"
    ls "$SDK"/install/lib/pyorbbecsdk*.so >/dev/null 2>&1 || { echo "FAILED: build produced no module"; ls -R install 2>/dev/null | head -30; exit 1; }
fi
ls -l "$SDK/install/lib" | head -20

step "usb permissions"
ls /etc/udev/rules.d/ | grep -i orbbec || {
    r=$(find "$SDK/scripts" -name "install_udev_rules.sh" | head -1)
    [ -n "$r" ] && sudo bash "$r" && sudo udevadm control --reload-rules && sudo udevadm trigger
}

step "depth test"
PYTHONPATH="$SDK/install/lib" python3 -u /tmp/depth_test.py
