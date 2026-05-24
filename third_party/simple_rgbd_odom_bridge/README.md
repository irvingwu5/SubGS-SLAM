# Simple-RGBD-Odometry Bridge

C++ pybind bridge for integrating PRBonn/Simple-RGBD-Odometry into FVO-GS-SLAM.

## Prerequisites

- Python >= 3.7
- pybind11: `pip install pybind11`
- Eigen3: `sudo apt install libeigen3-dev`
- Sophus: included as submodule under `third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/3rdparty/sophus/`
- OpenCV (for Python): `pip install opencv-python`

## Build

```bash
# From project root:
bash scripts/build_simple_rgbd_odom_bridge.sh
```

## Manual Build (if script fails)

```bash
cd third_party/Simple-RGBD-Odometry/python
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

The built .so file will be in `build/rgbd_odom/pybind/`.

## Import

```bash
export PYTHONPATH="third_party/Simple-RGBD-Odometry/python:$PYTHONPATH"
python -c "from utils.rgbd_vo_prior import version; print(version())"
```

## Pure Python Fallback

If pybind compilation fails, the `rgbd_odom` Python package provides a pure Python
implementation of `RGBDOdom.register_frame()` and `Ransac.ransac_kabsch_umeyama()`.
The pybind module only accelerates the C++ core operations (Frame, VoxelMap, RANSAC).

To use pure Python without pybind:
```python
# The rgbd_odom package detects missing pybind and falls back to numpy/scipy
from rgbd_odom import RGBDOdom
```
