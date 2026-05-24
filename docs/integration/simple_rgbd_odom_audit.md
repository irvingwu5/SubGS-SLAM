# Simple-RGBD-Odometry Integration Audit

审计日期：2026-05-12

## 1. Upstream Code Facts

### 1.1 仓库结构

Simple-RGBD-Odometry 有两套平行实现：

| 层 | 路径 | 状态 |
|---|---|---|
| C++ 核心库 | `cpp/rgbd_odom/src/` (Frame, VoxelMap, PNP, Ransac) | 完整，可编译 |
| C++ pipeline | `cpp/rgbd_odom/pipeline/` (RgbdOdom, RGBDConfig) | **不完整** — 依赖缺失的 `Registration.hpp` |
| Python 实现 | `python/rgbd_odom/` (RGBDOdom, Frame, Ransac, VoxelMap) | **完整，可直接使用** |
| PyBind 桥接 | `python/rgbd_odom/pybind/rgbd_odom_pybind.cpp` | 完整，包装 C++ 核心类 |
| CMake 构建 | `python/CMakeLists.txt` | 完整，产出 `rgbd_odom_pybind` 模块 |

### 1.2 Frame 类

**C++ 定义** (`cpp/rgbd_odom/src/Frame.hpp`):
```cpp
struct Frame {
    explicit Frame(const Vector2dVector &keypoints,       // 2D像素坐标
                   const Vector32iVector &descriptors,     // ORB描述子 (32字节/个)
                   const std::vector<double> &depth_values, // 米制深度
                   const Eigen::Matrix3d &intrinsics);     // 3x3内参矩阵K

    Vector3dVector TransformAndUnprojectKeypoints(const Sophus::SE3d &pose) const;
    // 实现: depth * K_inv * [u,v,1]^T → point_camera, 然后 pose * point_camera → world

    Vector2dVector keypoints_;
    Vector32iVector descriptors_;  // Eigen::Matrix<int, 32, 1>
    std::vector<double> depth_values_;
    Eigen::Matrix3d intrinsics_;
};
```

**Python 包装** (`python/rgbd_odom/frame.py`):
- 封装 C++ `_Frame`（通过 pybind），暴露 `point_cloud(pose)` 方法
- 输入: numpy keypoints (N,2), numpy descriptors (N,32) uint8, depth_values 列表, intrinsics (3,3)

### 1.3 RgbdOdom 类（Python 实现，可直接使用）

**关键方法** `register_frame(rgb_img, depth_img)` (文件: `python/rgbd_odom/rgbd_odom.py`):

1. `cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)` — RGB→灰度
2. `cv2.ORB_create(nfeatures).detectAndCompute(gray_img)` — ORB 特征提取
3. `keypoints_filter(config, keypoints, descriptors, depth_img)` — 深度范围过滤（需 `min_range < depth < max_range`）
4. `Frame(keypoints, descriptors, depth_values, intrinsics)` — 构造 Frame
5. `const_velocity_model = inv(poses[-2]) @ poses[-1]` — 恒速运动预测
6. `initial_guess = last_pose @ const_velocity_model` — 预测 C2W
7. `voxel_map.get_points_descriptors(initial_guess, search_radius)` — 局部地图查询
8. `cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch` + ratio test (0.7) — 描述子匹配
9. `ransac.ransac_kabsch_umeyama(frame, map_pts, initial_guess, intrinsics, max_corr_dist)` — C++ pybind RANSAC + Kabsch-Umeyama + PNP refine
10. `voxel_map.update(frame, new_pose)` — 更新体素地图
11. 返回 `(pose_camera_in_world_frame, corresp_tuple)`

**质量指标可获得性**:
- RANSAC 返回 `(pose_matrix, inlier_indices)` → 可计算 `num_inliers`, `inlier_ratio`
- 匹配数可通过包装层记录 `len(good_matches)`
- `num_keypoints` = ORB 输出数量
- `num_valid_depth_keypoints` = 深度过滤后数量

### 1.4 依赖

C++ 核心库依赖: Eigen3, Sophus, OpenCV (仅 pipeline 层需要)
Python 实现依赖: numpy, opencv-python (cv2), pybind11 (编译时)
PyBind 包装依赖: pybind11, Eigen3, Sophus

### 1.5 CMake 目标

- `cpp/rgbd_odom/CMakeLists.txt`: 只构建 `src/` 和 `metrics/`，**不构建 `pipeline/`**（因缺失 Registration.hpp）
- `python/CMakeLists.txt`: 构建 `rgbd_odom_pybind` 模块，依赖 `cpp/rgbd_odom/` 子目录

---

## 2. Pose Convention Analysis

### 2.1 证据链：pose 是 global C2W

| 证据 | 来源 | 强度 |
|---|---|---|
| Python 变量命名 `pose_camera_in_world_frame` | `rgbd_odom.py:101,120` | 强 |
| `VoxelMap::Update` 调用 `TransformAndUnprojectKeypoints(pose)` 其中 `pose * point_camera` → world | `VoxelMap.cpp:59`, `Frame.cpp:37` | 强 |
| Python `initial_guess = last_pose @ const_velocity_model` 其中 `const_velocity_model = inv(poses[-2]) @ poses[-1]` | `rgbd_odom.py:85,101` | 强：`C2W_{n-1} @ (C2W_{n-2}^{-1} @ C2W_{n-1})` = 预测 C2W |
| `GetPointsDescriptors(pose, radius)` 用 `pose.inverse() * voxel_center` 判断体素是否在相机前方 | `VoxelMap.cpp:47-49` | 中：pose 的逆把世界体素中心转到相机坐标系 |
| `RemovePointsFarFromLocation(pose)` 用 `pose.inverse() * voxel_center` | `VoxelMap.cpp:82` | 中：同上 |
| `pybind _get_points_descriptors` 接收 `Eigen::Matrix4d T` 直接用 `Sophus::SE3d pose(T)` | `rgbd_odom_pybind.cpp:111-114` | 强：pybind 层不做任何 inverse |

### 2.2 C++ pipeline 层的位姿方向 bug

`RGBDOdom.cpp:25`:
```cpp
Sophus::SE3d initial_guess = (poses_.back() * GetPredictionModel()).inverse();
```
这里对预测 C2W 取了 `.inverse()`，变成了 W2C。但 `RansacKabschUmeyama` 内部调用 `frame.TransformAndUnprojectKeypoints(T)` 期望 T 是 C2W。**这是 C++ pipeline 的 bug，不过不影响我们，因为我们会用 Python 实现。**

### 2.3 FVO-GS-SLAM 的位姿约定

| 变量 | 语义 | 数据类型 |
|---|---|---|
| `viewpoint.T` | global W2C | `torch.float32` (4,4) on GPU |
| `torch.linalg.inv(viewpoint.T)` | global C2W | torch.float32 (4,4) |
| `est_c2w` (FFTVO 返回) | global C2W | `np.float64` (4,4) |
| `prev_c2w` (candidate 用) | global C2W | `np.float64` (4,4) |
| `viewpoint.T = inv(selected["c2w"])` | C2W → W2C 写入 | GPU torch |

**集成时需要**: SimpleRGBDOdom 返回 `est_c2w` (C2W)，然后 `viewpoint.T = np.linalg.inv(est_c2w)` 写入。

---

## 3. FVO Current VO Path (FFTVO)

### 3.1 初始化链

```
slam.py:161 → FrontEnd.__init__(config) → 读取 config["FFTEdgeVO"] → 设置 self.use_fft_edge_vo
tracking() 首次调用 → _init_fft_edge_vo() → FFTEdgeVO(config, W, H, fx, fy, cx, cy)
```

### 3.2 每帧 tracking() 调用链

```
Step 1: FFTVO.track(rgb_bgr, depth_np, init_c2w) → (success, est_c2w, info)
  - 如 VO 未初始化：用上一帧/当前帧 set_reference()
  - 如 VO 已初始化：直接 track()
Step 2: Pose initialization
  - candidate_selection_enable=true: _build_candidates() → _select_candidate() → 设置 viewpoint.T
    候选: previous, constant_velocity, fft_vo
  - candidate_selection_enable=false: VO成功→直接用，失败→上一帧pose
Step 2b: VO render gate (vo_render_gate_enable=true)
  - 检查 VO candidate 的渲染质量（color loss, depth loss, opacity, score ratio）
  - 决定 vo_render_accepted 状态
Step 3: Refinement iter count → refine_iters (短/长)
Step 4: Render refinement (Adam, 可微渲染, get_loss_tracking)
Step 5: 保存 best_T → viewpoint.T
Cache update:
  - last_vo_ref_* 缓存 (供VO参考帧初始化)
  - last_c2ws 缓存 (供 constant velocity)
  - last_tracking_diag (供 submap cut gate)
```

### 3.3 Candidate Selection 候选名映射

| 当前候选名 | 含义 | 转换为 viewpoint.T |
|---|---|---|
| `"previous"` | 上一帧 C2W | `inv(prev_c2w)` |
| `"constant_velocity"` | 恒速预测 C2W | `inv(cv_c2w)` |
| `"fft_vo"` | FFTVO 估计 C2W | `inv(vo_c2w)` |

### 3.4 子图切图与 FFTVO 关系

- `perform_submap_cut()` (line 893): `self.fft_edge_vo_initialized = False`
- 切图后在主循环 (line 1232-1238): 用 seed 帧重建 VO 参考帧
- Backend sync (line 990-1020): VO 参考帧位姿与后端优化同步

### 3.5 FFTVO 不影响以下模块

- Keyframe selection: 完全独立，只看运动/时间阈值
- Backend queue: 完全独立，只传 keyframe 数据
- Mapping: 完全独立，只依赖 keyframe + Gaussian

---

## 4. Integration Risks

### 4.1 Depth Unit

| 数据集 | depth_scale | 原始格式 | 需转换 |
|---|---|---|---|
| TUM | 5000.0 | uint16 毫米 | `depth_m = raw / 5000.0` |
| Replica | 1000.0 | uint16 毫米 | `depth_m = raw / 1000.0` |
| ScanNet++ | 1.0 | float32 米 | 无需转换 |

**风险**: Simple-RGBD-Odometry 的 `keypoints_filter` 期望 depth_img 已经是以米为单位的 float。FVO 中 `viewpoint.depth` 需要在传入前确认单位。TUM 数据集在 `dataset.py` 中做了 `/depth_scale` 转换，所以 `viewpoint.depth` 应该已经是米制。

**验证方法**: 打印 `viewpoint.depth.min(), max()` 确认值范围。

### 4.2 RGB vs BGR

- Simple-RGBD-Odometry 的 `RGBDOdom.register_frame(rgb_img, depth_img)` 内部调用 `cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)` — 输入必须是 **RGB**
- FVO 的 `_camera_rgb_to_bgr()` 将 Camera 的 RGB 转为 BGR 给 FFTVO
- **SimpleRGBDOdom 应接收 RGB**，不需要 BGR 转换

### 4.3 Pose Direction

- **低风险**: 代码证据充分（见 §2），但仍需 identity test 验证
- Identity test: 输入连续相同图像，C2W 应保持 identity

### 4.4 PyBind 编译

- 需要: Eigen3, Sophus, pybind11, Python development headers
- **高风险**: FVO 的 conda 环境中这些库可能版本不对
- **备选方案**: 纯 Python 移植 `RGBDOdom.register_frame()` 逻辑（~150 行），仅依赖 numpy + opencv（已在 conda 环境中）

### 4.5 Multiprocessing Compatibility

- FVO 使用 `mp.set_start_method("spawn")` (slam.py:633)
- PyBind 模块需在 spawn 子进程中可导入
- **中等风险**: pybind + CUDA 在 spawn 模式下可能冲突
- **备选方案**: 纯 Python 移植无此风险

### 4.6 ORB Descriptor 类型

- C++ `Vector32i` = `Eigen::Matrix<int, 32, 1>`
- Python cv2.ORB 输出 `descriptors` 为 `np.ndarray` shape (N, 32) dtype uint8
- PyBind 包装 `py::py_array_to_vectors_int<Vector32i>` 处理转换
- **低风险**: 纯 Python 移植直接用 numpy uint8，不需要类型转换

### 4.7 Quality Metrics

- `num_inliers`: RANSAC 返回 `inlier_indices`，`len(inlier_indices)`
- `inlier_ratio`: `num_inliers / num_matches`
- `num_matches`: 需在包装层记录 BFMatcher ratio test 后匹配数
- `num_keypoints`: ORB 输出 keypoints 数量
- `runtime_ms`: Python `time.perf_counter()` 计时

### 4.8 Map Reset 策略

Simple-RGBD-Odometry 的 voxel map 是累积式的。子图切图时需要:
- 调用 `voxel_map.clear()` 清空旧地图
- 重置 `poses` 列表
- 用 seed 帧重新初始化

---

## 5. Proposed Patch Plan

### Phase 1: 构建 pybind 模块（或备选纯 Python）
- 尝试编译 `python/CMakeLists.txt` → `rgbd_odom_pybind`
- 备选: 纯 Python 移植 `SimpleRGBDOdom.register_frame()` 逻辑

### Phase 2: Python Provider
- 创建 `utils/rgbd_vo_prior/` 包
  - `simple_rgbd_odom.py`: SimpleRGBDVOProvider 类
  - `pose_convention.py`: C2W/W2C 转换 + 验证工具
- 接口: `track(rgb_img, depth_np, init_c2w) → (success, est_c2w, info)`

### Phase 3: 位姿约定测试
- Identity test
- Stationary test
- TUM fr1_desk 前 50 帧测试

### Phase 4: 前端集成
- 新增 `VOPrior` 配置段
- FrontEnd 支持 `vo_prior_type` 切换
- Candidate selection 新增 `"simple_rgbd_odom"` 候选
- Fallback 链保持现有逻辑

### Phase 5: 评估与消融
- 完整序列 ATE 对比

---

## 6. Commands Inspected

以下为审计过程中实际读取和检查的文件：

```bash
# 上游 Simple-RGBD-Odometry
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/src/Frame.hpp
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/src/Frame.cpp
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/src/VoxelMap.hpp
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/src/VoxelMap.cpp
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/src/PNP.hpp
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/src/PNP.cpp
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/src/Ransac.hpp
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/src/Ransac.cpp
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/pipeline/RGBDOdom.hpp
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/pipeline/RGBDOdom.cpp
third_party/Simple-RGBD-Odometry/cpp/rgbd_odom/CMakeLists.txt
third_party/Simple-RGBD-Odometry/python/rgbd_odom/rgbd_odom.py
third_party/Simple-RGBD-Odometry/python/rgbd_odom/pipeline.py
third_party/Simple-RGBD-Odometry/python/rgbd_odom/frame.py
third_party/Simple-RGBD-Odometry/python/rgbd_odom/ransac.py
third_party/Simple-RGBD-Odometry/python/rgbd_odom/mapping.py
third_party/Simple-RGBD-Odometry/python/rgbd_odom/preprocess.py
third_party/Simple-RGBD-Odometry/python/rgbd_odom/config/config.py
third_party/Simple-RGBD-Odometry/python/rgbd_odom/pybind/rgbd_odom_pybind.cpp
third_party/Simple-RGBD-Odometry/python/CMakeLists.txt

# FVO-GS-SLAM
utils/slam_frontend.py (全文)
utils/fft_edge_vo.py (全文)
utils/pose_utils.py
utils/camera_utils.py
utils/config_utils.py
configs/rgbd/tum/base_config.yaml
configs/rgbd/replica/base_config.yaml
configs/rgbd/scannetpp/base_config.yaml
configs/rgbd/tum/fr1_desk.yaml
```
