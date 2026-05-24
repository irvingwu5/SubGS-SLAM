#!/bin/sh
set -eu

# Build Simple-RGBD-Odometry pybind module for FVO-GS-SLAM.
# Usage:  sh scripts/build_simple_rgbd_odom_bridge.sh

PROJECT_ROOT="$(pwd)"
UPSTREAM_DIR="$PROJECT_ROOT/third_party/Simple-RGBD-Odometry"
PYTHON_DIR="$UPSTREAM_DIR/python"

echo "=== Simple-RGBD-Odometry Bridge Build ==="
echo "Project root:  $PROJECT_ROOT"
echo "Upstream dir:  $UPSTREAM_DIR"

# Step 1: prerequisites
echo ""
echo "--- Step 1: Checking prerequisites ---"

if [ ! -d "$UPSTREAM_DIR" ]; then
    echo "ERROR: Upstream not found at $UPSTREAM_DIR"
    exit 1
fi
echo "OK: upstream directory found"

# pybind11
if python -c "import pybind11" 2>/dev/null; then
    echo "OK: pybind11 importable"
else
    echo "WARNING: pybind11 not importable (pip install pybind11)"
fi

# Eigen3
if pkg-config --exists eigen3 2>/dev/null || [ -d "/usr/include/eigen3" ] || [ -d "/usr/local/include/eigen3" ]; then
    echo "OK: Eigen3 found"
else
    echo "ERROR: Eigen3 not found. Install: sudo apt install libeigen3-dev"
    exit 1
fi

echo "--- Prerequisites done ---"

# Step 2: Build
echo ""
echo "--- Step 2: Building pybind module ---"

cd "$PYTHON_DIR"
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
nproc_count="$(nproc 2>/dev/null || echo 4)"
cmake --build . -j"$nproc_count"

echo ""
echo "--- Build done ---"

# Step 3: Verify
echo ""
echo "--- Step 3: Verification ---"
cd "$PROJECT_ROOT"

PYTHONPATH="$PYTHON_DIR:$PYTHONPATH" python -c "
from utils.rgbd_vo_prior import version, is_available
print('Version:', version())
print('Available:', is_available())
"

echo ""
echo "=== Complete ==="
echo ""
echo "Add to your env before running slam.py:"
echo "  export PYTHONPATH=\"$PYTHON_DIR:\$PYTHONPATH\""
