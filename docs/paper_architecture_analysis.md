# FVO-GS-SLAM 论文模块架构分析

本文分析论文拟包含模块的作用、模块间数据流及整体 Pipeline。

## 1. 论文模块清单

| 模块 | 对应文件 | 状态 |
|---|---|---|
| 2DGS Model | `gaussian_splatting/scene/gaussian_model.py` | 核心表示 |
| VOPrior (Simple RGBD Odometry) | `utils/rgbd_vo_prior/__init__.py` | Tracking 初值 |
| RSKM (Random Sampling Keyframe Mapping) | `utils/slam_backend.py:_select_rskm_keyframes` | 建图采样策略 |
| FFT Mask + `use_freq_sampling_density` | `utils/fft_filter.py` + `gaussian_model.py` | 频率感知播种 |
| Error Mask + `use_rgb_error_mask` | `utils/slam_backend.py:add_next_kf` + `gaussian_model.py` | 渲染误差补点 |
| Expected Depth | `gaussian_splatting/gaussian_renderer/__init__.py` | CUDA 输出 allmap[0] |
| Median Depth | `gaussian_splatting/gaussian_renderer/__init__.py` | CUDA 输出 allmap[5] |
| Surfel-Aware (SA) Depth | `submodules/diff-surfel-rasterization/.../forward.cu` | 置信度加权期望深度 |
| Depth Distortion Loss (`use_dist`) | `utils/slam_utils.py:get_loss_mapping_rgbd` | 深度压实正则 |
| Color Refinement | 后端 mapping 始终优化 RGB 参数 | 颜色持续优化 |

### 不出现在论文中的模块

子图策略、RAP2DGS Lite、Gaussian Inheritance、Loop Closure、PGO、Reloc3R、FDN Normal (`use_fdn`)

---

## 2. 模块作用详解

### 2.1 2DGS Model — 地图表示核心

**文件**: `gaussian_splatting/scene/gaussian_model.py`

2D Gaussian Splatting 将场景表示为可微的 2D surfel 集合。每个 surfel 由以下参数定义：

| 参数 | 维度 | 含义 |
|---|---|---|
| `_xyz` | (N, 3) | 世界坐标系 3D 位置 |
| `_features_dc` | (N, 1, 3) | 球谐系数 DC 分量（基础颜色） |
| `_features_rest` | (N, 15, 3) | 球谐系数高阶分量（视角相关颜色） |
| `_opacity` | (N, 1) | sigmoid-activated 不透明度 |
| `_scaling` | (N, 3) | 局部切空间缩放（前两维=surfel半径，第三维≈0） |
| `_rotation` | (N, 4) | 四元数旋转（局部切空间→世界空间） |
| `_normal` | (N, 3) | 世界坐标系 surfel 法线（由 rotation 推导） |

**关键操作**:

- **`extend_from_pcd_seq()`** (line 462): 从 RGBD 关键帧反投影点云 → FFT/Error mask 过滤 → KNN-based 尺度初始化 → 扩展 Gaussian 集合
- **`densify_and_prune()`**: 梯度累积驱动的 densify（克隆/分裂）+ 不透明度/尺度/观测次数驱动的 prune
- **`training_setup()`**: Adam 优化器配置，位置 LR 使用指数衰减调度
- **`capture_dict()`**: 导出所有参数用于 checkpoint

**2DGS 与 3DGS 的关键区别**: 2DGS 使用 2D surfel（圆盘），`_scaling` 第三维固定为小值，渲染时使用 ray-disk intersection 而非 3D ellipsoid projection。

### 2.2 VOPrior (Simple RGBD Odometry) — Tracking 初值提供

**文件**: `utils/rgbd_vo_prior/__init__.py`

封装上游 Simple-RGBD-Odometry（C++ pybind 加速）为 tracking 初值提供者。不替代可微渲染位姿精化，仅提供初始 C2W 估计。

**核心流程** (`track` 方法):

```
RGB + Depth → ORB 特征提取 → RANSAC PnP → 帧间相对位姿
    → delta = inv(VO_prev) @ VO_curr
    → est_c2w = init_c2w @ delta   （帧到帧 delta 对齐，防止 VO 漂移）
```

**质量门控**:

| 门控 | 作用 |
|---|---|
| `min_valid_keypoints` (80) | 有效深度特征点数量下限 |
| `min_inliers` (20) | RANSAC 内点数量下限 |
| `min_inlier_ratio` (0.15) | 内点比例下限 |
| `max_motion_trans` (0.50m) | 帧间最大平移 |
| `max_motion_rot_deg` (30°) | 帧间最大旋转 |

**在 Frontend 中的位置** (Frontend.tracking, line 363):

```
Step 1: VOPrior.track() → vo_success, est_c2w
Step 2: Candidate Selection → 从 {previous, constant_velocity, external_vo} 中选择最佳初始位姿
Step 3: Render Refinement → Adam 优化 cam_rot_delta / cam_trans_delta
```

### 2.3 RSKM — 随机关键帧重放建图

**文件**: `utils/slam_backend.py:_select_rskm_keyframes` (line 703)

在 mapping 阶段，从 active keyframe 池中随机采样监督帧，避免最近关键帧对 Gaussian 优化的过强支配，提升旧视角渲染质量。

**采样策略**:

```python
每 N 个采样中（rskm_current_frame_interval=4），有 1 次强制使用当前帧
其余采样从全体 active keyframes 中均匀随机选取
```

**配置路径**: `Training.use_rskm`, `Training.rskm_current_frame_interval`

**对比不使用 RSKM**: 默认只从 `current_window` 中采样监督帧，模型优化过度偏向最新视角，旧视角渲染质量退化。

### 2.4 FFT Mask + `use_freq_sampling_density` — 频率感知播种

**文件**: `utils/fft_filter.py` → `utils/slam_backend.py:add_next_kf` → `gaussian_model.py:create_pcd_from_image`

**FFT Mask 生成** (`fft_filter.py`):

```
RGB → CLAHE (局部对比度增强) → FFT → Gaussian HPF (高通滤波)
    → IFFT → Triangle 阈值二值化 → torch.bool mask
    高频区 = True（纹理丰富），低频区 = False（平坦区域）
```

**两个作用**:

#### 作用 1: 频率感知采样密度 (`use_freq_sampling_density`)

在 `gaussian_model.py:create_pcd_from_image` (line 185):

```python
high_stride = 2 (高频区 stride，密集采样)
low_stride  = 4 (低频区 stride，稀疏采样)
freq_sampling = (高频区 & 2-stride) | (低频区 & 4-stride)
```

**效果**: 纹理丰富区域播更多 Gaussian，平坦区域播更少，采样效率更高。

#### 作用 2: 初始尺度控制

在 `gaussian_model.py:create_pcd_from_image_and_depth` (line 393):

```python
scale_multiplier[低频区] = low_freq_scale_multiplier (默认 1.05)
# 低频区高斯初始尺度略大，补偿稀疏采样
```

**配置路径**: `Ablation.use_fft_mask`, `Training.use_freq_sampling_density`, `Training.low_freq_scale_multiplier`

### 2.5 Error Mask + `use_rgb_error_mask` — 渲染误差补点

**文件**: `utils/slam_backend.py:add_next_kf` (line 231)

在非初始化帧，通过渲染当前地图到新关键帧视角，检测渲染不足区域，生成 error_mask 指导新 Gaussian 播种。

**Error Mask 组成** (三个来源的 OR):

#### 来源 1: Alpha Mask (空洞检测)
```
alpha_mask = (render_opacity < 0.98)  # 渲染不透明度不足的区域
```

#### 来源 2: Depth Error Mask (深度穿透检测)
```
depth_error = |gt_depth - render_depth|
depth_error_mask = (render_depth > gt_depth)  # 渲染深度比真值远（穿透）
                 & (depth_error > 10 * median_error)  # 误差显著大于中位数
```

#### 来源 3: RGB Error Mask (`use_rgb_error_mask`)
```
rgb_error = sum(|gt_rgb - render_rgb|)  # 逐通道 L1
rgb_error_mask = (rgb_error > rgb_error_th) & valid_depth  # RGB 表达误差大
```

**配置路径**: `Ablation.use_error_mask`, `Training.use_rgb_error_mask`, `Training.rgb_error_th`

### 2.6 三种深度监督源

**文件**: `submodules/diff-surfel-rasterization/cuda_rasterizer/forward.cu` + `gaussian_splatting/gaussian_renderer/__init__.py`

CUDA forward 渲染过程输出多种深度图，统一存储在 `allmap` 中：

#### 2.6.1 Expected Depth (期望深度)

**来源**: `allmap[0]` = `Σ (w_i * d_i)` （alpha 加权深度和，未归一化）

**Python 归一化** (`render/__init__.py` line 188):
```python
render_depth_expected = allmap[0] / clamp(render_alpha, min=depth_eps)
```

**物理含义**: 沿像素射线，每个 splat 深度按其 alpha 贡献加权平均。是标准 alpha compositing 的自然深度定义。

#### 2.6.2 Median Depth (中值深度)

**来源**: `allmap[5]` — CUDA 在 alpha accumulation 过程中记录 alpha 首次超过 0.5 时的 splat 深度。

**物理含义**: 沿射线的 "表面" 深度——即 alpha 累积刚好过半时的深度，近似光线首次击中不透明表面的位置。

#### 2.6.3 Surfel-Aware (SA) Depth — 置信度加权期望深度

**来源**: CUDA forward kernel（`use_sa=true` 时）

**核心思想**: 在 alpha accumulation 之前，对每个 splat 计算其深度与当前表面估计（median depth）的偏差，用置信度将 outlier 深度拉回表面。

```
median_depth = alpha 首次过半时的当前 splat 深度
error = d_i - median_depth
conf = exp(-error² / (4σ²))      # σ 来自 splat 的 2D 投影尺度
adjusted_depth = median_depth + conf * error  # conf→1 保持原位, conf→0 拉回表面
```

然后使用 `adjusted_depth` 替代原始 `d_i` 进行 alpha-weighted accumulation。

**效果**: 遮挡边界处，背景 splat 的深度被拉向前景表面，减少深度混叠（depth aliasing）。

**Python 层选择** (`render/__init__.py` line 199):
```python
if use_sa_depth:
    surf_depth = render_depth_expected  # 直接用 SA 期望深度
else:
    surf_depth = (1-depth_ratio) * render_depth_expected + depth_ratio * render_depth_median
```

**配置路径**: `pipeline_params.use_sa`, `pipeline_params.use_sa_depth`, `pipeline_params.depth_ratio`

### 2.7 `use_dist` — 深度失真损失

**文件**: `utils/slam_utils.py:get_loss_mapping_rgbd` (line 119)

深度失真损失强制每个像素沿射线的 Gaussian 深度分布集中，减少 "floaters"（浮空物）。

**两种模式**:

| `use_sa` | `use_sa_dist` | 含义 |
|---|---|---|
| false | — | 标准 2DGS distortion: `Σ w_i·(d_i - D)²` (m-based) |
| true | false | SA variance 禁用（`use_sa` 和 `use_sa_dist` 解耦） |
| true | true | SA depth variance: `Σ w_i·(d_i - d_median)²` |

**Loss 计算**:
```python
dist_loss = lambda_dist * rend_dist[rgb_pixel_mask].mean()
loss += dist_loss
```

**消融结论**（来自 SA depth 技术报告）: `use_sa_dist=true` 改善 Depth L1 但劣化 ATE，已否决。推荐 `use_sa=true, use_sa_depth=true, use_sa_dist=false`。

**配置路径**: `opt_params.lambda_dist`, `opt_params.use_sa_dist`

### 2.8 Color Refinement — 颜色持续优化

**含义**: 后端 mapping 始终优化 Gaussian 的颜色参数（`_features_dc` 和 `_features_rest`），无需显式开关。

在 `gaussian_model.py:training_setup` 中，`f_dc` 和 `f_rest` 始终在 Adam optimizer 的 param groups 中，学习率分别为 `feature_lr` 和 `feature_lr / 20.0`。

Mapping loss 中的 RGB L1 + DSSIM 项持续驱动颜色优化。

---

## 3. 数据流

### 3.1 整体 Pipeline

```
                    ┌──────────────────────────────────────────────────────┐
                    │                 RGBD Dataset                         │
                    │         (TUM / Replica / ScanNet++)                  │
                    └──────────────────────┬───────────────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────────────┐
                    │              Camera Sequence (FrontEnd)               │
                    │                                                      │
                    │  每帧:                                                │
                    │  1. VOPrior.track()  →  est_c2w (初值)               │
                    │  2. Candidate Selection → 最佳初始位姿                │
                    │  3. Render Refinement → 精化位姿 (Adam)              │
                    │  4. Keyframe Decision → 是否插关键帧                  │
                    │  5. Sliding Window Management                        │
                    └──────┬──────────────────┬────────────────────────────┘
                           │                  │
              queue: keyframe + viewpoint     queue: sync_backend
                           │                  │ (Gaussian snapshot + visibility)
                           ▼                  ▼
                    ┌──────────────────────────────────────────────────────┐
                    │               BackEnd (独立进程)                      │
                    │                                                      │
                    │  初始化: Seed frame → extend_from_pcd_seq            │
                    │    (FFT mask 控制采样密度 + 初始尺度)                  │
                    │                                                      │
                    │  每关键帧:                                            │
                    │  1. add_next_kf:                                     │
                    │     - FFTFilter.generate_frequency_mask() → freq_mask │
                    │     - Render → error_mask (alpha+depth+rgb)          │
                    │     - extend_from_pcd_seq(freq_mask & error_mask)    │
                    │  2. map(current_window, prune=False):                 │
                    │     - RSKM 采样监督帧                                 │
                    │     - 对每个监督帧 render → get_loss_mapping          │
                    │       (RGB L1 + Depth L1 + dist loss)                 │
                    │     - backward + Adam step                           │
                    │     - densify_and_prune (周期触发)                    │
                    └──────────────────────────────────────────────────────┘
```

### 3.2 关键数据流详解

#### 3.2.1 FFT Mask → Gaussian 播种

```
FrontEnd: 无（论文版 FFT mask 在后端计算）
    ↓
BackEnd.add_next_kf():
    FFTFilter.generate_frequency_mask(RGB_BGR)
    → viewpoint.freq_mask (torch.bool, H×W)
    ↓
GaussianModel.extend_from_pcd_seq():
    ↓
GaussianModel.create_pcd_from_image():
    [use_freq_sampling_density=true]
    freq_sampling = (高频区 & 2-stride) | (低频区 & 4-stride)
    → valid_mask = valid_mask & freq_sampling
    ↓
GaussianModel.create_pcd_from_image_and_depth():
    is_high_freq = freq_mask_np[v, u]  # 按反投影坐标查表
    scale_multiplier[低频区] = low_freq_scale_multiplier (1.05)
    → scales = base_scales + log(scale_multiplier)
```

#### 3.2.2 Error Mask → Gaussian 播种

```
BackEnd.add_next_kf():
    render(当前地图, 新关键帧视角)
    → render_opacity, render_depth
    ↓
    alpha_mask = (render_opacity < 0.98)
    depth_error_mask = (render_depth > gt_depth) & (depth_error > 10*median_error)
    rgb_error_mask = (sum|gt_rgb - render_rgb| > rgb_error_th) & valid_depth
    ↓
    viewpoint.error_mask = alpha_mask | depth_error_mask | rgb_error_mask
    ↓
GaussianModel.create_pcd_from_image():
    valid_mask = valid_mask & error_mask_np
    → 只在渲染不足区域播新 Gaussian
```

#### 3.2.3 渲染器 → 深度 Loss

```
CUDA forward (forward.cu):
    per-pixel alpha accumulation
    allmap[0] = Σ(w_i * d_i)           # expected depth sum
    allmap[5] = median_depth           # alpha 过半深度
    allmap[6] = depth variance         # SA: Σ w_i·(d_i-d_median)²
    allmap[1] = Σ w_i                  # render alpha
    ↓
Python render (__init__.py):
    render_depth_expected = allmap[0] / allmap[1]
    surf_depth = use_sa_depth ? render_depth_expected :
                  (1-ratio)*expected + ratio*median
    rend_dist = allmap[6]
    ↓
get_loss_mapping_rgbd():
    depth_loss = L1(surf_depth, gt_depth)
    dist_loss = lambda_dist * rend_dist  (use_dist=true 时)
    loss = alpha*RGB_loss + (1-alpha)*depth_loss + dist_loss
```

#### 3.2.4 RSKM 采样 → Mapping Loss

```
BackEnd.map(current_window):
    if use_rskm and not prune:
        supervised_kf_ids = _select_rskm_keyframes(current_window, num_samples)
        // 每 4 个采样中 1 个是当前帧，其余均匀随机
    else:
        supervised_kf_ids = current_window (+ 随机非窗口帧)
    ↓
    for kf_idx in supervised_kf_ids:
        render(viewpoint[kf_idx]) → image, depth, rend_dist
        loss += get_loss_mapping(image, depth, rend_dist)
    ↓
    loss.backward() → Adam.step()  # 对所有 Gaussian 参数
```

---

## 4. 渲染输出全图

CUDA rasterizer 每次 render 调用输出 (`render/__init__.py`):

| 字段 | 维度 | 含义 | 消费者 |
|---|---|---|---|
| `render` | (3, H, W) | 渲染 RGB | tracking/mapping loss (RGB L1+DSSIM) |
| `depth` | (1, H, W) | 最终深度 (expected/SA/混合) | tracking/mapping depth L1 loss |
| `opacity` | (1, H, W) | 累积 alpha | opacity mask (threshold 0.95/0.98) |
| `rend_normal` | (3, H, W) | 累积法线 (世界系) | 本论文不使用 |
| `surf_normal` | (3, H, W) | 深度图推导法线 | 本论文不使用 |
| `rend_dist` | (1, H, W) | 深度失真/方差 | dist loss (`use_dist=true` 时) |
| `viewspace_points` | (N, 3) | 屏幕空间 2D 均值 | densification stats |
| `visibility_filter` | (N,) bool | 可视高斯 (radii>0) | visibility update / densify stats |
| `radii` | (N,) | 屏幕空间半径 | max_radii2D 更新 |
| `n_touched` | (N,) | 累计触碰次数 | occ_aware_visibility |

CUDA 内部分配 (`allmap`):

| allmap 索引 | 内容 |
|---|---|
| `[0]` | `Σ(w_i * d_i)` 期望深度和 |
| `[1]` | `Σ w_i` 累积 alpha |
| `[2:5]` | 累积法线 (视图系) |
| `[5]` | median depth (alpha 过半深度) |
| `[6]` | SA: `Σ w_i·(d_i-d_median)²` / 标准: m-based distortion |

---

## 5. Tracking Loss vs Mapping Loss

### Tracking Loss (`utils/slam_utils.py:get_loss_tracking_rgbd`, line 62)

```
loss = alpha * L1(rgb, gt_rgb, opacity_weighted)
     + (1-alpha) * L1(depth, gt_depth, opacity_gt_095 & depth_gt_001)
```

- 使用 `opacity > 0.95` 掩码（仅高置信度像素监督深度）
- 同时优化 RGB 和 depth
- 不包含 DSSIM 和 dist loss

### Mapping Loss (`utils/slam_utils.py:get_loss_mapping_rgbd`, line 102)

```
loss = alpha * L1(rgb, gt_rgb).mean()
     + (1-alpha) * L1(depth, gt_depth).mean()
     + lambda_dist * rend_dist.mean()    [if lambda_dist > 0]
```

- 不使用 opacity masking（所有有效像素平等监督）
- 包含 dist loss（可选）
- 可选的 exposure compensation（`image_ab = exp(a)*image + b`）

---

## 6. 坐标系约定

| 变量 | 语义 | 格式 |
|---|---|---|
| `viewpoint.T` | 全局 **W2C** | 4×4 tensor (CUDA) |
| `inv(viewpoint.T)` | 全局 **C2W** | 4×4 (需计算) |
| `cam_rot_delta` | SE(3) 旋转 delta | 3-vector (so3) |
| `cam_trans_delta` | SE(3) 平移 delta | 3-vector |
| `update_pose(viewpoint)` | 应用 delta 到 `viewpoint.T` | W2C = exp(delta)⁻¹ @ W2C |

---

## 7. 配置关键路径

论文模块涉及的配置项：

```yaml
# VOPrior: tracking 初值
VOPrior:
  type: simple_rgbd_odom
  tracking_refine_iters: 40
  tracking_fallback_iters: 100
# SimpleRGBDOdom 子配置
SimpleRGBDOdom:
  min_valid_keypoints: 80
  min_inliers: 20
  min_inlier_ratio: 0.15

# RSKM: 随机关键帧重放
Training:
  use_rskm: true
  rskm_current_frame_interval: 4
  rskm_seed: 42

# FFT Mask + 频率感知采样
Ablation:
  use_fft_mask: true
Training:
  use_freq_sampling_density: true
  low_freq_scale_multiplier: 1.05
  high_freq_sample_stride: 2
  low_freq_sample_stride: 4

# Error Mask + RGB Error Mask
Ablation:
  use_error_mask: true
Training:
  use_rgb_error_mask: true
  rgb_error_th: 0.5
  depth_error_median_factor: 10.0

# SA Depth + 三种深度 + dist
pipeline_params:
  use_sa: true
  use_sa_depth: true
  depth_ratio: 1.0       # 0=纯expected, 1=纯median
  depth_eps: 1.0e-6
opt_params:
  use_sa_dist: false       # SA dist 已否决
  lambda_dist: 0.0         # dist loss 权重
```

---

## 8. 简化的 Pipeline 架构图（论文版）

```
┌─────────────────────────────────────────────────────────────────┐
│                         RGBD Sequence                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FrontEnd (主进程)                                               │
│                                                                  │
│  For each frame:                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Step 1: VOPrior (Simple RGBD Odometry)                     │   │
│  │   RGBD → ORB + RANSAC PnP → frame-to-frame delta           │   │
│  │   → est_c2w (初值，不替代渲染精化)                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Step 2: Candidate Selection                                │   │
│  │   {previous, constant_velocity, external_vo} → 渲染预检    │   │
│  │   → 选最佳初始位姿                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Step 3: Render Refinement                                  │   │
│  │   Adam on cam_rot_delta / cam_trans_delta                  │   │
│  │   Loss: RGB L1 + Depth L1 (opacity-masked)                 │   │
│  │   收敛 or 达到 refine_iters                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Step 4: Keyframe Decision                                  │   │
│  │   重叠率 < kf_overlap & 平移 > kf_min_translation           │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Step 5: Sliding Window + Visibility Sync                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  queue → BackEnd: ["keyframe", idx, viewpoint, window]           │
│  queue ← BackEnd: ["sync_backend", gaussians, visibility, ...]   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  BackEnd (独立进程)                                              │
│                                                                  │
│  Init:                                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Seed frame → create_pcd_from_image                         │   │
│  │   FFT mask → freq_sampling_density (高频密/低频疏)         │   │
│  │   → 初始 Gaussian 集合                                     │   │
│  │ initialize_map: iters=init_itr_num, RGB+depth loss         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Per Keyframe:                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ add_next_kf:                                               │   │
│  │   1. FFTFilter.generate_frequency_mask(RGB) → freq_mask    │   │
│  │   2. render(gaussians, viewpoint)                          │   │
│  │      → alpha_mask | depth_error_mask | rgb_error_mask     │   │
│  │      → viewpoint.error_mask                                │   │
│  │   3. extend_from_pcd_seq(freq_mask & error_mask)           │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ map(current_window):                                       │   │
│  │   RSKM 采样:                                               │   │
│  │     supervised_kfs = _select_rskm_keyframes(window, N)     │   │
│  │     (每4次采样1次当前帧 + 均匀随机历史帧)                    │   │
│  │                                                            │   │
│  │   For each kf in supervised_kfs:                          │   │
│  │     render → image, depth, rend_dist, visibility, ...     │   │
│  │     loss = L1(rgb) + L1(depth) + lambda_dist*dist         │   │
│  │     loss.backward()                                        │   │
│  │                                                            │   │
│  │   Adam step → update ALL Gaussian params                   │   │
│  │     _xyz, _features_dc, _features_rest,                    │   │
│  │     _opacity, _scaling, _rotation                         │   │
│  │                                                            │   │
│  │   Periodic: densify_and_prune / opacity_reset              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

最终: 保存完整 Gaussian map → ATE 评估 → 渲染评估
```

---

## 9. 模块间依赖关系

```
VOPrior ──→ FrontEnd.tracking (初值)
                │
                ├──→ Candidate Selection (位姿选择)
                │
                └──→ Render Refinement (位姿精化)
                         │
                         ▼
FFTFilter ──→ BackEnd.add_next_kf (freq_mask 生成)
                │
                ├──→ GaussianModel.create_pcd_from_image
                │      freq_sampling_density (采样密度)
                │
                └──→ GaussianModel.create_pcd_from_image_and_depth
                       freq_mask → scale_multiplier (初始尺度)

Render ──→ BackEnd.add_next_kf (error_mask 生成)
   │           alpha_mask | depth_error_mask | rgb_error_mask
   │
   ├──→ BackEnd.map (mapping loss)
   │      render → RGB, depth, rend_dist
   │
   ├──→ FrontEnd.tracking (tracking loss)
   │      render → RGB, depth, opacity
   │
   └──→ SA Depth (CUDA forward)
          use_sa=true → confidence-weighted expected depth
          use_sa_depth=true → surf_depth = SA expected depth

RSKM ──→ BackEnd.map (监督帧采样)
           _select_rskm_keyframes → supervised_kf_ids

Color Refinement ──→ BackEnd.map (始终启用)
                       features_dc + features_rest in Adam params

Dist Loss ──→ get_loss_mapping_rgbd
              rend_dist (SA variance or standard distortion)
              lambda_dist > 0 → loss += lambda_dist * rend_dist
```

---

## 10. 与完整项目的关系

论文版本 = 完整项目 **减去**：

| 减去模块 | 影响 |
|---|---|
| Submap Strategy | 不再切子图，全局单一 Gaussian map |
| Gaussian Inheritance | 无跨子图状态继承 |
| RAP2DGS Lite | 无继承评分 |
| Loop Closure + PGO | 无全局一致性校正 |
| Reloc3R | 无关键帧对粗位姿估计 |
| FDN Normal (`use_fdn`) | 无法线监督 loss |

论文版本 **保留** 的核心链路：

```
Tracking: VOPrior 初值 → Candidate 选择 → Render 精化 → 关键帧决策
Mapping:  FFT mask + Error mask 播种 → RSKM 采样 → RGB+D+dist loss → Gaussian 优化
Representation: 2DGS surfel 集合 + SA depth + 三种深度 + dist 压实
```
