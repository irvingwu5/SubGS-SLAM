# Simple-RGBD-Odometry 接入 FVO-GS-SLAM 详细分析

## 1. 总体架构

Simple-RGBD-Odometry 作为 **VOPrior（视觉里程计先验）** 接入 FVO-GS-SLAM，为每帧的 render-based tracking 提供初始位姿估计。它**不替代** 2DGS 渲染精化，而是作为候选初值之一参与多候选渲染仲裁，减少渲染精化所需迭代次数。

整体分为四层：

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 3: FrontEnd.tracking() — 消费者                        │
│  slam_frontend.py:363-591                                     │
│  ├── Step 1: 调用 VO 获取 est_c2w                               │
│  ├── Step 2: 候选选择 (_build_candidates → _render_precheck     │
│  │              → _select_candidate)                            │
│  ├── Step 3: VO Render Gate 二次验证                            │
│  ├── Step 4: 决定精化迭代次数 (refine vs fallback)              │
│  └── Step 5: Adam 渲染精化 (get_loss_tracking)                  │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: utils/rgbd_vo_prior/__init__.py — 桥接层            │
│  SimpleRGBDVOProvider                                         │
│  ├── track(rgb, depth, init_c2w) → (success, est_c2w, info)  │
│  ├── reset(initial_c2w) → 清除 VO 内部状态                     │
│  └── 质量门控 + 运动门控 + 帧到帧防漂移                         │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: third_party/Simple-RGBD-Odometry — 上游 C++ pybind  │
│  rgbd_odom.RGBDOdom (python/rgbd_odom/rgbd_odom.py)           │
│  ├── ORB 特征检测 + 描述子 (cv2.ORB)                           │
│  ├── 深度滤波 (keypoints_filter)                               │
│  ├── 恒定速度预测 (get_prediction_model)                       │
│  ├── KNN 匹配 + ratio test (knnmatch_desc_and_ratio_test)      │
│  ├── RANSAC Kabsch-Umeyama (C++ pybind _Ransac)               │
│  └── VoxelMap 局部地图 (C++ pybind _VoxelMap)                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1: 上游 Simple-RGBD-Odometry（C++ pybind 加速）

### 2.1 文件结构

| 文件 | 职责 |
|------|------|
| `third_party/Simple-RGBD-Odometry/python/rgbd_odom/rgbd_odom.py` | `RGBDOdom` 类：ORB + 匹配 + RANSAC + VoxelMap 管线 |
| `third_party/Simple-RGBD-Odometry/python/rgbd_odom/frame.py` | `Frame` 类：包装 C++ pybind `_Frame`（关键点、描述子、深度值、内参、点云投影） |
| `third_party/Simple-RGBD-Odometry/python/rgbd_odom/ransac.py` | `Ransac` 类：包装 C++ pybind `_Ransac`（Kabsch-Umeyama 刚体变换估计） |
| `third_party/Simple-RGBD-Odometry/python/rgbd_odom/mapping.py` | `VoxelMap` 类：包装 C++ pybind `_VoxelMap`（体素滤波局部地图） |
| `third_party/Simple-RGBD-Odometry/python/rgbd_odom/preprocess.py` | `keypoints_filter`：深度范围滤波，剔除无效深度关键点 |
| `third_party/Simple-RGBD-Odometry/python/rgbd_odom/config/config.py` | `RGBDConfig`：Pydantic 配置模式 |
| `third_party/Simple-RGBD-Odometry/python/rgbd_odom/pybind/rgbd_odom_pybind.cpp` | C++ pybind11 wrapper，暴露 `_Frame`、`_Ransac`、`_VoxelMap` |

### 2.2 RGBDOdom 核心管线

`RGBDOdom.register_frame(rgb_img, depth_img)` 方法（`rgbd_odom.py:52-80`）执行完整的帧到局部地图配准：

```
RGBDOdom.register_frame(rgb_img, depth_img)
  │
  ├── 1. cv2.cvtColor(rgb_img, COLOR_RGB2GRAY)
  ├── 2. orb.detectAndCompute(gray) → keypoints, descriptors
  ├── 3. keypoints_filter(config, kps, descs, depth_img)
  │      └── 深度范围滤波 (min_range ~ max_range)，剔除无效深度关键点
  ├── 4. Frame(keypoints, descriptors, depth_values, intrinsics)
  │      └── 创建 Frame 对象（C++ pybind _Frame）
  ├── 5. get_prediction_model()
  │      └── 恒定速度模型: inv(poses[-2]) @ poses[-1]
  │      └── 初始猜测: last_pose @ const_velocity_model
  ├── 6. register_frame_to_map(frame, initial_guess)
  │      ├── local_map.get_points_descriptors(initial_guess, search_radius)
  │      │      └── 从 VoxelMap 中搜索半径内的地图点 + 描述子
  │      ├── knnmatch_desc_and_ratio_test(frame, map_kps, map_descs)
  │      │      ├── BFMatcher.knnMatch(k=2) → Hamming 距离匹配
  │      │      └── ratio_test(ratio=0.7): 第一近邻/第二近邻 < 0.7
  │      └── ransac.ransac_kabsch_umeyama(filtered_frame, filtered_map, init, K, max_corr_dist)
  │             └── C++ pybind RANSAC + Kabsch-Umeyama 刚体变换估计
  ├── 7. local_map.update(frame, new_pose)
  │      └── 将当前帧点云插入 VoxelMap
  └── 8. 返回 (frame.point_cloud(initial_guess), corresp_tuple)
```

**关键点**：
- VO 维护内部局部地图 (`local_map`)，帧间通过 ORB 特征匹配 + RANSAC PnP 估计位姿
- 恒定速度模型提供初始猜测，加速 RANSAC 收敛
- C++ pybind 加速了 Frame 构造、RANSAC、VoxelMap 查找/更新

---

## 3. Layer 2: 桥接层 — SimpleRGBDVOProvider

文件：`utils/rgbd_vo_prior/__init__.py`（55-187 行，共 194 行）

### 3.1 模块初始化

```python
# 自动注入上游 Python 路径
_upstream_python = ".../third_party/Simple-RGBD-Odometry/python"
sys.path.insert(0, _upstream_python)

# 延迟导入，不阻塞模块加载
from rgbd_odom.config import RGBDConfig
from rgbd_odom.rgbd_odom import RGBDOdom as _UpstreamRGBDOdom
```

导入失败时不抛异常，而是设置 `_IMPORT_ERROR`，由 `is_available()` 和 `version()` 检查。

### 3.2 SimpleRGBDVOProvider.__init__

```python
def __init__(self, config: dict, W: int, H: int, fx: float, fy: float, cx: float, cy: float):
```

**配置来源**：`config["SimpleRGBDOdom"]` 字典（来自 YAML 配置文件）

**初始化步骤**：

1. **构建 `RGBDConfig`**（上游 Pydantic 配置）：
   - `data.min_range` / `max_range`：深度有效范围（默认 0.05m ~ 5.0m）
   - `mapping.voxel_size`：体素地图分辨率（默认 0.5m）
   - `mapping.max_points_per_voxel`：每体素最大点数（默认 20）
   - `registration.max_correspondence_distance`：RANSAC 最大对应距离（默认 0.20m）
   - `registration.search_radius`：地图点搜索半径（默认 5.0m）
   - `descriptor.num_descriptors`：ORB 特征数（默认 1000）

2. **创建上游 `RGBDOdom` 实例**：`self._odom = _UpstreamRGBDOdom(intrinsics, config=rgbd_cfg)`

3. **设置质量门控阈值**：
   - `min_valid_keypoints = 80`：最少有效深度关键点数
   - `min_inliers = 20`：最少 RANSAC 内点数
   - `min_inlier_ratio = 0.15`：最低内点比率
   - `max_motion_trans = 0.50m`：最大帧间平移
   - `max_motion_rot_deg = 30.0°`：最大帧间旋转

### 3.3 SimpleRGBDVOProvider.track — 核心方法

```python
def track(self, rgb_img: np.ndarray, depth_np: np.ndarray,
          init_c2w: Optional[np.ndarray] = None
) -> Tuple[bool, np.ndarray, dict]:
```

**输入**：
- `rgb_img`：`(H, W, 3)` uint8 numpy RGB 图像
- `depth_np`：`(H, W)` float32 深度图（米）
- `init_c2w`：上一帧的精炼全局 C2W 矩阵（`inv(prev_cam.T)`）

**输出**：`(success: bool, est_c2w: np.ndarray, info: dict)`

**内部流程**：

```
track(rgb_img, depth_np, init_c2w)
  │
  ├── 1. 灰度转换: cv2.cvtColor(rgb_img, COLOR_RGB2GRAY)
  ├── 2. 关键点计数: self._odom.orb.detect(gray) → num_keypoints
  ├── 3. 调用上游 VO:
  │      frame_pcd, corresp_tuple = self._odom.register_frame(rgb_img, depth_np)
  │      (stdout 重定向到空缓冲区，抑制匹配不足时的上游 print 警告)
  │
  ├── 4. 帧到帧 delta 防漂移 (核心设计):
  │      vo_pose = self._odom.poses[-1]     # VO 内部累积位姿
  │      if len(poses) >= 2 and init_c2w:
  │          delta = inv(vo_prev) @ vo_pose  # 帧间相对运动 (VO 坐标系)
  │          est_c2w = init_c2w @ delta      # 应用到上一帧精炼位姿
  │      else:
  │          est_c2w = init_c2w               # 第一帧直接用上一帧位姿
  │
  ├── 5. 质量门控 (三层):
  │      num_valid_depth < min_valid_keypoints  → fail
  │      num_inliers < min_inliers              → fail
  │      inlier_ratio < min_inlier_ratio        → fail
  │
  ├── 6. 运动门控:
  │      motion_trans > max_motion_trans  → fail
  │      motion_rot_deg > max_motion_rot_deg → fail
  │
  └── 7. 返回 (success, est_c2w, info_dict)
```

**帧到帧 delta 防漂移机制**：
- 上游 VO 的 `poses` 列表存储累积位姿（从 VO 初始化的恒等位姿开始），长期运行会产生漂移
- 桥接层计算 `delta = inv(vo_prev) @ vo_pose`（VO 坐标系下的帧间相对运动）
- 然后 `est_c2w = init_c2w @ delta`，将增量应用于上一帧**经渲染精化**的全局 C2W
- 这样每帧都从精炼后的精确位姿重新出发，VO 只提供帧间增量，漂移不累积

### 3.4 SimpleRGBDVOProvider.reset

```python
def reset(self, initial_c2w: Optional[np.ndarray] = None):
    self._odom.local_map.clear()   # 清空体素地图
    self._odom.poses = []           # 清空位姿历史
    self.frame_id = 0
```

在子图切图时调用，确保新子图从干净的 VO 状态开始。

---

## 4. Layer 3: FrontEnd 集成（VO 与 Render-based Tracking 互动）

### 4.1 初始化阶段

`FrontEnd.__init__`（`slam_frontend.py:70-101`）从配置读取 `VOPrior` 段：

| 配置参数 | 默认值 | 作用 |
|---------|--------|------|
| `type` | `"none"` | VO 类型：`"none"` / `"simple_rgbd_odom"` |
| `full_render_warmup_frames` | 0 | 前 N 帧跳过 VO，全量迭代 warmup |
| `tracking_refine_iters` | 40 | VO render-accepted 后精化迭代数 |
| `tracking_fallback_iters` | 100 | VO 失败/未选中时精化迭代数 |
| `candidate_selection_enable` | false | 是否启用多候选渲染仲裁 |
| `vo_render_gate_enable` | false | 是否启用 VO render 二次验证 |
| `vo_candidate_interval` | 1 | VO 候选间隔（每 N 帧选一次） |

`_init_vo_prior()`（107-118 行）惰性初始化：
```python
def _init_vo_prior(self):
    if self.vo_prior is not None:
        return
    if self.vo_prior_type == "simple_rgbd_odom":
        from utils.rgbd_vo_prior import SimpleRGBDVOProvider
        self.vo_prior = SimpleRGBDVOProvider(
            self.config,
            W=self.dataset.width, H=self.dataset.height,
            fx=self.dataset.fx, fy=self.dataset.fy,
            cx=self.dataset.cx, cy=self.dataset.cy,
        )
```

### 4.2 tracking() — 主跟踪管线（5 步互动）

`FrontEnd.tracking(cur_frame_idx, viewpoint)`（363-591 行）是 VO 与 render-based tracking 互动的核心。

#### Step 1: 外部 VO 先验估计（376-415 行）

```
├── if vo_prior_type == "simple_rgbd_odom":
│   ├── _init_vo_prior()              # 惰性初始化
│   ├── rgb_img = _camera_rgb(viewpoint)  # Camera → (H,W,3) uint8 numpy
│   ├── depth_np = viewpoint.depth     # (H,W) float32 numpy (米)
│   ├── init_c2w = inv(prev_cam.T)    # 上一帧精炼位姿作为 base
│   └── vo_success, est_c2w, vo_info = self.vo_prior.track(rgb, depth, init_c2w)
│
├── 跳过条件:
│   ├── in_warmup (cur_frame_idx < warmup_frames) → vo_success = False
│   ├── vo_candidate_interval 跳帧 (每 N 帧选一次) → vo_success = False
│   └── VO 输出 ≈ 上一帧位姿 (delta < 1e-4m, <1e-5 rad) → vo_success = False
│
└── debug_log 输出: [VOPrior] frame N: success=T/F, dt=ms, inliers=N, motion_t/r, reason
```

**关键设计**：
- **warmup 跳过**：前 `warmup_frames` 帧跳过 VO 候选，用全量迭代温暖启动 IMAP
- **间隔跳帧**：`vo_candidate_interval > 1` 时每隔 N 帧才使用 VO，节省计算
- **静止跳过**：VO 输出与上一帧几乎相同时跳过，避免无意义的候选

#### Step 2: 候选选择与渲染仲裁（417-465 行）

```
if candidate_selection_enable and gaussians is not None:
│
├── _build_candidates(prev_cam, vo_success, est_c2w):
│   ├── 候选1: "previous"        — inv(prev_cam.T) 上一帧位姿
│   ├── 候选2: "constant_velocity" — last_c2ws[-1] @ inv(last_c2ws[0]) @ last_c2ws[-1]
│   └── 候选3: "external_vo"     — est_c2w (VO 输出，仅当 vo_success=true)
│
├── _select_candidate(candidates, viewpoint):
│   ├── 对每个候选执行 _render_precheck(c2w, viewpoint):
│   │   ├── 无梯度渲染当前地图到候选位姿
│   │   ├── 计算 l1_rgb (opacity-weighted L1 颜色误差)
│   │   ├── 计算 l1_depth (opacity-masked L1 深度误差)
│   │   ├── 计算 opacity_ratio (不透明度覆盖率)
│   │   └── 恢复 viewpoint.T 原始值
│   ├── 拒绝 opacity_ratio < candidate_min_opacity_ratio 的候选
│   ├── 得分: score = l1_rgb + lambda_depth * l1_depth + lambda_coverage * penalty
│   └── 选择得分最低的候选，设置 viewpoint.T = inv(selected_c2w)
│
├── VO Render Gate 二次验证 (vo_render_gate_enable=true):
│   ├── 找到 "external_vo" 候选的渲染指标
│   ├── 逐项检查:
│   │   ├── vo_l1_rgb > vo_max_color_loss     → REJECT
│   │   ├── vo_l1_depth > vo_max_depth_loss    → REJECT
│   │   ├── vo_opacity < vo_min_opacity_ratio  → REJECT
│   │   └── vo_score/best_score > vo_max_score_ratio_to_best → REJECT
│   └── 全部通过 → vo_render_accepted = True
│
└── 若 vo_render_gate_enable=false, vo_render_accepted = (selected == "external_vo")
```

**候选渲染仲裁流程**：

```
   ┌──────────────┐
   │  候选位姿列表  │
   │ previous     │
   │ const_vel    │
   │ external_vo  │
   └──────┬───────┘
          ▼
   ┌──────────────────┐
   │ _render_precheck  │  对每个候选：无梯度渲染 → L1_rgb + L1_depth + opacity
   └──────┬───────────┘
          ▼
   ┌──────────────────┐
   │ _select_candidate │  按综合得分排序 → 选择最优候选
   └──────┬───────────┘
          ▼
   ┌──────────────────┐
   │  VO Render Gate   │  VO 候选被选中时：二次验证渲染质量
   └──────┬───────────┘
          ▼
   ┌──────────────────┐
   │ viewpoint.T 赋值  │  设置渲染精化的初始位姿
   └──────────────────┘
```

#### Step 3: 精化迭代次数决策（467-480 行）

```
if in_warmup:
    refine_iters = tracking_itr_num          # 全量迭代
elif candidate_selection_enable and vo_prior_type != "none":
    if vo_render_accepted:
        refine_iters = tracking_refine_iters  # 40-60 次 (VO 可信)
    else:
        refine_iters = tracking_fallback_iters # 100-120 次 (VO 不可信)
elif vo_prior_type != "none":
    refine_iters = tracking_refine_iters if vo_success
                   else tracking_fallback_iters  # 无候选选择时的简化逻辑
else:
    refine_iters = tracking_itr_num           # 无 VO 时的默认迭代数
```

**VO 的核心价值**：当 VO 输出被渲染验证接受时，精化迭代从 100-120 次降至 40-60 次，节省约 50% 的计算量。

#### Step 4: 渲染精化（482-541 行）

```
for tracking_itr in range(refine_iters):
    render_pkg = render(viewpoint, render_model, pipeline_params, background)
    loss = get_loss_tracking(config, image, depth, opacity, viewpoint)
    loss.backward()
    pose_optimizer.step()
    converged = update_pose(viewpoint)
    if converged: break
```

Adam 优化器在 `cam_rot_delta` 和 `cam_trans_delta` 上运行。这是 2DGS 可微渲染的标准位姿精化过程。**VO 不参与此步骤**——VO 的输出仅影响初始位姿和迭代次数。

#### Step 5: PAR RSKM 元数据保存（564-589 行）

当 `rskm_mode == "par"` 时，保存 VO 与渲染精化的一致性度量：

```python
viewpoint.vo_init_c2w = est_c2w                          # VO 初始估计
viewpoint.render_opt_c2w = inv(viewpoint.T)              # 渲染精化后位姿
trans_err, rot_err = compute_pose_delta_metrics(vo_init_c2w, render_opt_c2w)
# 计算 PAR reliability:
pose_error = trans_err + rot_err / 30.0
viewpoint.par_reliability = exp(-beta_pose * pose_error)
```

VO 与渲染精化的一致性越高 → `par_reliability` 越高 → 该帧在 RSKM 历史重放中被采样的权重越大。

### 4.3 子图切图时的 VO 重置

`perform_submap_cut()`（810-869 行）和主循环（1148-1150 行）中，子图切图时重置 VO：

```python
if self.vo_prior is not None and hasattr(self.vo_prior, "reset"):
    self.vo_prior.reset()
```

清空 VO 的局部地图和位姿历史，避免旧子图的 VoxelMap 数据污染新子图。

---

## 5. 完整数据流：一帧的 VO → Render 交互时间线

```
Frame t 到达
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: VO 先验估计                                        │
│   rgb_img (H,W,3) + depth_np (H,W) + init_c2w (prev)     │
│     → SimpleRGBDVOProvider.track()                        │
│       → 上游 ORB+RANSAC+Kabsch-Umeyama                    │
│       → delta = inv(vo_prev) @ vo_pose                    │
│       → est_c2w = init_c2w @ delta                        │
│       → 质量门控 (keypoints/inliers/ratio)                 │
│       → 运动门控 (trans/rot)                               │
│     → (vo_success, est_c2w, info_dict)                    │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: 候选选择 + 渲染仲裁                                  │
│   candidates = [previous, constant_velocity, external_vo]│
│                                                          │
│   对每个候选执行 _render_precheck(c2w, viewpoint):        │
│     viewpoint.T ← inv(c2w)                               │
│     render(viewpoint, model) → image, depth, opacity     │
│     计算: l1_rgb (颜色L1) + l1_depth (深度L1) + opacity   │
│     viewpoint.T ← 恢复                                    │
│                                                          │
│   _select_candidate: 按得分选最佳候选                       │
│   VO Render Gate: 若 VO 被选中，二次验证渲染质量             │
│                                                          │
│   → viewpoint.T = inv(selected_c2w)                      │
│   → vo_render_accepted = True/False                      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: 迭代次数决策                                       │
│   vo_render_accepted → refine_iters = 40-60 (少迭代)     │
│   !vo_render_accepted → refine_iters = 100-120 (多迭代)  │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: 2DGS Render Refinement                           │
│   Adam(cam_rot_delta, cam_trans_delta)                   │
│   for i in range(refine_iters):                          │
│     render → loss = L1_rgb + L1_depth + DSSIM            │
│     loss.backward → optimizer.step → update_pose         │
│   → viewpoint.T = best_T (精炼后位姿)                     │
│   → best_render_pkg                                      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Step 5: PAR RSKM 元数据                                   │
│   viewpoint.vo_init_c2w = est_c2w                        │
│   viewpoint.render_opt_c2w = inv(viewpoint.T)            │
│   trans_err, rot_err = compute_pose_delta_metrics(...)   │
│   viewpoint.par_reliability = exp(-beta * pose_error)    │
│   → VO-渲染一致性越高, reliability 越高                    │
└─────────────────────────────────────────────────────────┘
```

---

## 6. 配置文件结构

### 6.1 VOPrior 配置段（`base_config.yaml` 中）

控制 VO 在 FrontEnd 中的行为：

```yaml
VOPrior:
  type: simple_rgbd_odom            # "none" | "simple_rgbd_odom"
  debug_log: false
  full_render_warmup_frames: 100    # 前 N 帧全量迭代 warmup
  tracking_refine_iters: 60         # VO render-accepted 后精化迭代
  tracking_fallback_iters: 120      # VO 失败时精化迭代
  candidate_selection_enable: true  # 多候选渲染仲裁开关
  candidate_lambda_depth: 1.0
  candidate_lambda_coverage: 1.0
  candidate_min_opacity_ratio: 0.05
  vo_render_gate_enable: true       # VO render 二次验证开关
  vo_max_score_ratio_to_best: 1.25  # VO/最优得分比上限
  vo_max_score_ratio_to_previous: 1.10
  vo_min_opacity_ratio: 0.05
  vo_max_depth_loss:                # null=不限制
  vo_max_color_loss:                # null=不限制
  # 子图切图质量门控
  submap_cut_gate_enable: false
  submap_cut_min_opacity: 0.05
  submap_cut_max_delay: 3
```

### 6.2 SimpleRGBDOdom 配置段（`base_config.yaml` 中）

控制上游 VO 引擎本身的参数：

```yaml
SimpleRGBDOdom:
  enable: true
  min_range: 0.05                   # 深度有效范围下限 (m)
  max_range: 5.0                    # 深度有效范围上限 (m)
  voxel_size: 0.50                  # 体素地图分辨率 (m)
  max_points_per_voxel: 20          # 每体素最大点数
  max_correspondence_distance: 0.20 # RANSAC 对应距离阈值 (m)
  search_radius: 5.0                # 地图点搜索半径 (m)
  orb_nfeatures: 1000               # ORB 特征点数量
  min_valid_keypoints: 80           # 最少有效深度关键点数
  min_inliers: 20                   # 最少 RANSAC 内点数
  min_inlier_ratio: 0.15            # 最低内点比率
  max_motion_trans: 0.50            # 最大允许帧间平移 (m)
  max_motion_rot_deg: 30.0          # 最大允许帧间旋转 (deg)
```

---

## 7. 关键设计决策

### 7.1 VO 是"先验"不是"替代品"

VO 只提供 tracking 初始位姿，**不替代** 2DGS 渲染精化。无论 VO 是否成功，渲染精化始终运行。VO 的价值在于减少精化迭代次数（从 ~100 降至 ~40-60），而非替代精化。

### 7.2 帧到帧 delta 防漂移

VO 维护内部累积位姿，长期会漂移。桥接层不直接信任 VO 的累积位姿，而是：
```
est_c2w = init_c2w @ (inv(vo_poses[-2]) @ vo_poses[-1])
```
每帧从上一帧**渲染精化**后的精确位姿重新出发，只使用 VO 的帧间相对增量。

### 7.3 多候选渲染仲裁

VO 不是唯一的初始位姿来源。每帧构建 2-3 个候选（previous / constant_velocity / external_vo），通过**无梯度渲染预检查**选出最优候选。这避免了在 VO 质量差时因强制使用 VO 而损害 tracking。

### 7.4 三层质量门控递进

1. **VO 自身质量门控**（Layer 2）：关键点数、内点数、内点比率、运动幅度
2. **渲染预检查**（Layer 3 Step 2）：opacity 覆盖率、RGB/深度损失
3. **VO Render Gate**（Layer 3 Step 2）：VO 候选相对于其他候选的渲染质量比

三层递进确保只有真正高质量的 VO 输出才被信任。

### 7.5 子图切图 VO 重置

每次子图切图时调用 `vo_prior.reset()`，清空 VoxelMap 和位姿历史。每个子图从独立的 VO 状态开始，避免旧场景的 3D 地图点污染新场景的位姿估计。

### 7.6 PAR RSKM 位姿一致性利用

VO 初始位姿与渲染精化后位姿的偏差量化为 `par_reliability`。高一致性 → 高 reliability → RSKM 采样时该帧权重更高 → 后端 mapping 更倾向于重放"VO 与渲染一致"的帧。

### 7.7 无 C++ 代码侵入

集成完全在 Python 侧完成，通过直接导入上游 `RGBDOdom` 类。FVO-GS-SLAM 没有添加任何 C++ 代码。构建只需运行 pybind 编译脚本。

---

## 8. 关键文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `utils/rgbd_vo_prior/__init__.py` | 55-91 | `SimpleRGBDVOProvider.__init__`：配置 + 上游实例化 |
| `utils/rgbd_vo_prior/__init__.py` | 92-97 | `SimpleRGBDVOProvider.reset`：清除内部状态 |
| `utils/rgbd_vo_prior/__init__.py` | 100-187 | `SimpleRGBDVOProvider.track`：核心 VO 估计 + 门控 |
| `utils/slam_frontend.py` | 70-101 | `FrontEnd.__init__`：VOPrior 配置读取 |
| `utils/slam_frontend.py` | 107-118 | `FrontEnd._init_vo_prior`：惰性初始化 |
| `utils/slam_frontend.py` | 120-133 | `FrontEnd._camera_rgb`：Camera → numpy RGB |
| `utils/slam_frontend.py` | 145-168 | `FrontEnd._build_candidates`：构建候选列表 |
| `utils/slam_frontend.py` | 170-216 | `FrontEnd._render_precheck`：无梯度渲染预检查 |
| `utils/slam_frontend.py` | 218-257 | `FrontEnd._select_candidate`：候选评分与选择 |
| `utils/slam_frontend.py` | 376-415 | `tracking` Step 1：VO 先验估计 |
| `utils/slam_frontend.py` | 417-465 | `tracking` Step 2：候选选择 + VO render gate |
| `utils/slam_frontend.py` | 467-480 | `tracking` Step 3：迭代次数决策 |
| `utils/slam_frontend.py` | 482-541 | `tracking` Step 4：渲染精化 |
| `utils/slam_frontend.py` | 564-589 | `tracking` Step 5：PAR RSKM 元数据保存 |
| `utils/slam_frontend.py` | 858-860 | 子图切图 VO 重置 (perform_submap_cut) |
| `utils/slam_frontend.py` | 1148-1150 | 子图切图 VO 重置 (主循环) |
| `configs/rgbd/replica/base_config.yaml` | 142-167 | VOPrior 配置 |
| `configs/rgbd/replica/base_config.yaml` | 169-192 | SimpleRGBDOdom 配置 |
| `third_party/Simple-RGBD-Odometry/python/rgbd_odom/rgbd_odom.py` | 38-80 | 上游 `RGBDOdom` 类 |
| `third_party/Simple-RGBD-Odometry/python/rgbd_odom/rgbd_odom.py` | 87-120 | `register_frame_to_map`：匹配+RANSAC |
