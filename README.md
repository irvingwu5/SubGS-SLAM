# FVO-GS-SLAM: RGBD 2D Gaussian Splatting SLAM with VO Prior

FVO-GS-SLAM is an RGBD SLAM system based on 2D Gaussian Splatting and differentiable surfel rendering. The paper system uses Simple RGBD Odometry as VO prior for tracking initialization, render-based pose refinement with three depth supervision sources (expected depth, median depth, surfel-aware depth), FFT frequency-aware Gaussian seeding, error-mask guided dynamic point insertion, and RSKM random keyframe replay for mapping.

The maintained direction is RGBD indoor SLAM on TUM RGBD, Replica, and ScanNet++ datasets.

<p align="center">
  <a href="">
    <img src="./media/pipeline.png" alt="pipeline" width="100%">
  </a>
</p>

---

## Statement

This repository is developed from the MonoGS style Gaussian SLAM framework. The **paper system** includes:

**Core paper modules:**

- RGBD dataset loading (TUM, Replica, ScanNet++)
- **VOPrior**: Simple RGBD Odometry (ORB + RANSAC PnP) as tracking prior, frame-to-frame delta alignment
- **Candidate Selection**: {previous, constant_velocity, external_vo} → render precheck → best initial pose
- Render-based pose refinement (RGB + depth tracking loss, Adam on SE(3) delta)
- **FFT high-frequency mask** (CLAHE → FFT → Gaussian HPF → IFFT → triangle threshold)
  - `use_freq_sampling_density`: frequency-aware sampling (dense in high-freq, sparse in low-freq)
  - Initial scale control via `low_freq_scale_multiplier`
- **Error mask** guided dynamic point insertion (alpha holes + depth penetration + RGB error)
  - `use_rgb_error_mask` synchronized with `use_error_mask`
- **Three depth supervision sources**:
  - Expected depth (alpha-composited `Σw_i·d_i/Σw_i`)
  - Median depth (alpha > 0.5 first-hit depth)
  - Surfel-Aware (SA) depth (confidence-weighted expected depth, reduces aliasing at occlusion boundaries)
- **Depth distortion loss** (`use_dist`): compacts Gaussian depth distribution along rays
- 2D Gaussian map representation with differentiable surfel rendering
- Gaussian densification, opacity reset, and pruning
- **RSKM** (Random Sampling Keyframe Mapping): random keyframe replay for balanced mapping
- **Color refinement**: always-on RGB parameter optimization
- Keyframe selection and sliding window management
- Visibility maintenance with `occ_aware_visibility`
- ATE trajectory evaluation and rendering quality evaluation
- Ablation switches for controlled experiments
- GPU memory monitoring

**Modules NOT in the paper** (exist in codebase for compatibility but disabled/not described):
- FFT Edge VO (replaced by VOPrior)
- Submap strategy + Gaussian Inheritance + RAP2DGS Lite
- FDN normal supervision (`use_fdn`)
- Loop closure + PGO + Reloc3R + depth geometric verification

---

## Main Technical Components (Paper)

### 1. RGBD 2D Gaussian SLAM Pipeline

The system uses RGBD observations to initialize, track, and optimize a 2D Gaussian map. The differentiable surfel renderer outputs RGB, depth (expected/median/SA), opacity, visibility, radii, normal, and `n_touched`. These feed tracking loss, mapping loss, visibility update, densification, and pruning.

### 2. VOPrior: Simple RGBD Odometry for Tracking Initialization

`utils/rgbd_vo_prior/__init__.py` wraps upstream Simple-RGBD-Odometry (C++ pybind accelerated) as a tracking prior provider.

- **Frame-to-frame delta alignment**: `est_c2w = init_c2w @ delta`, where `delta = inv(VO_prev) @ VO_curr`. This prevents VO drift accumulation — each frame starts from the last refined pose.
- **Quality gates**: min valid keypoints (80), min inliers (20), min inlier ratio (0.15), max motion translation (0.50m), max rotation (30°).
- **Candidate Selection** (Stage 2): builds candidates from {previous, constant_velocity, external_vo}, runs lightweight render precheck on each, selects best by combined loss.
- VOPrior provides the initial pose but does NOT replace differentiable render-based refinement.

Tracking flow per frame:
1. VOPrior.track() → initial est_c2w from ORB + RANSAC PnP
2. Candidate Selection → render precheck → best initial pose
3. Render-based pose refinement (Adam on `cam_rot_delta` / `cam_trans_delta`, RGB + depth L1 loss)
4. Keyframe decision (overlap ratio + translation threshold)

### 3. FFT Frequency-Aware Gaussian Seeding

`utils/fft_filter.py` generates a high-frequency mask from RGB: CLAHE → FFT → Gaussian HPF → IFFT → triangle threshold → bool mask.

**Two roles in Gaussian seeding:**

- **Frequency-aware sampling density** (`use_freq_sampling_density`): high-frequency regions sampled at stride 2 (dense), low-frequency at stride 4 (sparse). Texture-rich areas get more Gaussians.
- **Initial scale control**: low-frequency Gaussians receive `low_freq_scale_multiplier` (1.05×) larger initial scale to compensate for sparser sampling.

### 4. Error Mask Guided Dynamic Point Insertion

In `utils/slam_backend.py:add_next_kf`, the system renders the current map into the new keyframe and detects under-rendered regions:

- **Alpha mask**: `render_opacity < 0.98` (holes)
- **Depth error mask**: `render_depth > gt_depth` & `depth_error > 10×median_error` (depth penetration)
- **RGB error mask** (`use_rgb_error_mask`): `sum|gt_rgb - render_rgb| > rgb_error_th` (color mismatch)

These three masks are OR-combined into `error_mask`, guiding new Gaussian insertion only in under-rendered regions.

### 5. Three Depth Supervision Sources

The CUDA forward rasterizer outputs three depth representations via `allmap`:

| Depth Type | Source | Meaning |
|---|---|---|
| **Expected Depth** | `allmap[0] / allmap[1]` = `Σ(w_i·d_i) / Σw_i` | Alpha-composited weighted average depth |
| **Median Depth** | `allmap[5]` | Depth at which accumulated alpha first exceeds 0.5 (first-hit surface) |
| **SA Depth** | Confidence-weighted expected depth | Outlier splat depths pulled toward median surface before accumulation |

**Surfel-Aware (SA) Depth**: When `use_sa=true`, the CUDA forward kernel adjusts each splat's depth before alpha accumulation. For each splat, deviation from the current median surface estimate is computed:

```
conf = exp(-(d_i - d_median)² / 4σ²)
adjusted_depth = median_depth + conf × (d_i - d_median)
```

As `conf → 0` (far from surface), depth is pulled to `d_median`. As `conf → 1` (near surface), depth is unchanged. This reduces depth aliasing at occlusion boundaries.

**Depth selection** in Python layer:
```python
if use_sa_depth:
    surf_depth = render_depth_expected  # SA expected depth
else:
    surf_depth = (1-depth_ratio)*expected + depth_ratio*median  # mixed
```

Key design decisions:
- **Preserve pose gradient**: SA backward correctly propagates gradients through the adjusted depth path.
- **Dual-switch control**: `use_sa` (CUDA) and `use_sa_depth` (Python loss selection) independently configurable for ablation.
- **SA distortion**: When `use_sa=true`, `allmap[6]` outputs SA depth variance `Σ w_i·(d_i-d_median)²` instead of standard m-based distortion.
- **Zero overhead**: Confidence computation (4 FMA + 1 exp per splat) inside existing CUDA kernel; FPS difference < 5%.

Ablation results (TUM fr3_office):
- SA depth alone (A1): ATE 0.02204m, Depth L1 0.1641m — **best trajectory**
- SA depth + dist λ=0.01 (A4): ATE 0.02423m, Depth L1 0.1495m — depth improves but trajectory degrades
- **SA dist vetoed**: use_sa_dist=false recommended

### 6. Depth Distortion Loss (`use_dist`)

`utils/slam_utils.py:get_loss_mapping_rgbd` — compacts Gaussian depth distribution along each pixel ray:

- **`use_sa=false`**: standard 2DGS distortion `Σ w_i·(d_i - D)²` (m-based, around mean depth)
- **`use_sa=true`**: SA variance `Σ w_i·(d_i - d_median)²` (around median surface)

```python
if lambda_dist > 0:
    dist_loss = lambda_dist * rend_dist[rgb_pixel_mask].mean()
    loss += dist_loss
```

### 7. RSKM: Random Sampling Keyframe Mapping

`utils/slam_backend.py:_select_rskm_keyframes` — during mapping, supervision frames are randomly sampled from the active keyframe pool instead of just the sliding window. Every `rskm_current_frame_interval` (4) samples, one is forced to be the current frame; the rest are uniformly random from all active keyframes.

This prevents the most recent keyframes from dominating Gaussian optimization, improving rendering quality from older viewpoints. (Reference: GS3SLAM)

### 8. Back End Gaussian-Only Mapping

The back end is an independent process. Mapping includes:
- RGB L1 + DSSIM loss (always on → color refinement)
- Depth L1 loss (using selected depth source)
- Depth distortion loss (`lambda_dist > 0` when enabled)
- FFT frequency mask guided sampling density + initial scale
- Error mask guided dynamic point insertion
- Periodic densify, opacity reset, and prune
- `occ_aware_visibility` per keyframe
- RSKM random keyframe replay supervision
- Backend pose policy: `optimize_keyframe_pose=true`, `optimize_keyframe_exposure=false`

### 9. Color Refinement

Color refinement is always on — the back end continuously optimizes `_features_dc` and `_features_rest` (spherical harmonics coefficients) via RGB L1 + DSSIM loss in every mapping iteration. No explicit switch needed; it is inherent to the Gaussian optimization pipeline.

---

## Non-Paper Modules (Code Present but Not Described)

The following modules exist in the codebase for compatibility but are NOT part of the paper:

| Module | How to disable |
|---|---|
| FFT Edge VO | Set `VOPrior.type=simple_rgbd_odom` |
| Submap Strategy | `Ablation.use_submap=false` |
| Gaussian Inheritance + RAP2DGS Lite | `Submap.use_inheritance=false` |
| FDN Normal Supervision | `Ablation.use_fdn=false` |
| Loop Closure + PGO + Reloc3R | `LoopClosure.mode=off` |

---

## System Architecture (Paper)
5. **Top-K selection**: Retain top-scoring Gaussians. Auto-supplement with simple scoring if RAP2DGS selection falls below target.
6. **Safety**: Never prunes active map. Auto-fallback to simple heuristic on any failure.

```yaml
RAP2DGSLite:
  enable: true                # master switch
  use_in_inheritance: true
  knn: {k: 16, chunk_size: 4096}
  features: {use_support, use_opacity, use_observation, use_area, use_normal, use_density: true}
  score_weights: {support: 0.25, opacity: 0.20, observation: 0.20, area: 0.10, normal: 0.15, density: 0.10}
  selection: {keep_percent: 0.25, max_keep: 8000}
```

---

## Repository Structure

```text
FVO-GS-SLAM
├── slam.py                         # main entry, process orchestration, evaluation
├── configs/
│   └── rgbd/
│       ├── tum/                    # TUM RGBD: base_config.yaml + scene overrides
│       ├── replica/                # Replica: base_config.yaml + scene overrides
│       └── scannetpp/              # ScanNet++: base_config.yaml + scene overrides
├── gaussian_splatting/
│   ├── gaussian_renderer/          # differentiable 2DGS surfel rendering
│   └── scene/gaussian_model.py     # Gaussian params, densify, prune, optimizer state
├── utils/
│   ├── slam_frontend.py            # tracking, VOPrior, candidate selection, keyframes
│   ├── slam_backend.py             # Gaussian mapping, FFT/error mask, RSKM, densify/prune
│   ├── rgbd_vo_prior/              # VOPrior: Simple-RGBD-Odometry wrapper
│   ├── fft_filter.py               # FFT high-frequency mask generation
│   ├── slam_utils.py               # tracking/mapping loss functions
│   ├── pose_utils.py               # SE(3) pose update utilities
│   ├── camera_utils.py             # Camera class (viewpoint.T = W2C)
│   ├── dataset.py                  # dataset loading (TUM, Replica, ScanNet++)
│   ├── eval_utils.py               # ATE and rendering evaluation
│   ├── config_utils.py             # YAML config loading
│   └── ...                         # non-paper auxiliary modules
├── submodules/                     # diff-surfel-rasterization, simple-knn
└── docs/                           # architecture analysis, experiment reports
```

---

## System Architecture (Paper)

```text
RGBD sequence
    ↓
Dataset loader → Camera objects
    ↓
FrontEnd (main process)
    ├── VOPrior (Simple RGBD Odometry): ORB + RANSAC PnP → frame-to-frame delta → est_c2w
    ├── Candidate Selection: {previous, constant_velocity, external_vo} → render precheck → best
    ├── Render-based pose refinement
    │     Adam on cam_rot_delta / cam_trans_delta
    │     RGB L1 + depth L1 tracking loss (opacity-masked)
    │     SA depth (use_sa=true): confidence-weighted expected depth
    ├── Keyframe decision + sliding window + visibility sync
    └── queue → BackEnd
    ↓ queue messages
BackEnd (independent process)
    ├── Seed frame init → Gaussian map initialization (FFT freq-aware sampling)
    ├── Keyframe → FFT mask + error mask → extend_from_pcd_seq
    ├── Gaussian only mapping (RGB L1 + DSSIM + depth L1 + dist loss)
    │     └── RSKM: randomly sampled keyframe supervision
    ├── Densify / prune / opacity reset
    ├── occ_aware_visibility + pose sanity check
    └── Push Gaussian snapshot → FrontEnd
    ↓
Main process after tracking
    ├── Stop backend
    ├── Evaluate ATE + rendering quality
    └── Optional: save PLY
```

---

## Module Roles (Paper)

### `slam.py` — Main Entry

Main entry and system controller. Creates Gaussian model, dataset, front end, back end, and optional GUI. Handles evaluation mode overrides, GPU memory monitoring, ATE and rendering evaluation.

### `utils/slam_frontend.py` — Front End

The front end is the main process (online tracking and scheduling).

Key responsibilities:
- Construct per-frame `Camera` objects (viewpoint.T = global W2C)
- Run VOPrior (Simple RGBD Odometry) for initial pose estimation
- Candidate Selection from {previous, constant_velocity, external_vo}
- Refine pose via render-based differentiable optimization
- Insert keyframes based on overlap ratio and translation
- Manage sliding window and visibility synchronization
- Send `init`, `keyframe`, `stop` to back end
- Receive Gaussian snapshots and visibility from back end

### `utils/rgbd_vo_prior/` — VOPrior

Simple RGBD Odometry wrapper for tracking initialization.

- **Frame-to-frame delta alignment**: `est_c2w = init_c2w @ delta` (prevents VO drift)
- **Quality gates**: min valid keypoints, min inliers, min inlier ratio, max motion t/r
- Returns `(success, est_c2w, info_dict)`

### `utils/fft_filter.py` — FFT Mask

Builds a high-frequency mask from RGB: CLAHE → FFT → Gaussian HPF → IFFT → triangle threshold → bool mask. Controls Gaussian sampling density (dense in high-freq, sparse in low-freq) and initial scale (larger in low-freq).

### `utils/slam_backend.py` — Back End

Asynchronous local mapping (independent process).

Key responsibilities:
- Receive `init`, `keyframe` messages from front end
- Initialize Gaussian map from seed keyframe
- Compute FFT mask (via FFTFilter) and error mask (via render + gt comparison)
- Add new keyframes with FFT mask + error mask guided point insertion (`extend_from_pcd_seq`)
- Optimize Gaussian parameters with RSKM-sampled keyframe supervision
- RGB L1 + DSSIM + depth L1 + dist loss
- Collect visibility and densification statistics
- Densify, reset opacity, and prune Gaussian points
- Maintain `occ_aware_visibility` keyed by keyframe index
- Push Gaussian snapshots to front end
### `gaussian_splatting/scene/gaussian_model.py` — 2DGS Model

Stores and updates 2DGS surfel map parameters: `_xyz` (position), `_features_dc/_rest` (SH color), `_opacity`, `_scaling` (2D tangent-space), `_rotation` (quaternion→normal). Key methods: `extend_from_pcd_seq()` (RGBD back-projection + FFT/error mask filtering + KNN scale init), `densify_and_prune()` (gradient-driven densify + opacity/scale prune), `training_setup()` (Adam optimizer with exponential position LR decay).

### `gaussian_splatting/gaussian_renderer/__init__.py` — Differentiable Renderer

Differentiable 2DGS surfel renderer wrapping CUDA rasterizer. Outputs:

| Field | Meaning |
|---|---|
| `render` | RGB image (3×H×W) |
| `depth` | Selected depth: SA expected / median-expected mixed |
| `opacity` | Accumulated alpha |
| `rend_dist` | Depth variance: SA `Σw_i·(d_i-d_median)²` or standard distortion |
| `visibility_filter` | Visible Gaussians (radii > 0) |
| `n_touched` | Per-Gaussian touch count |

Depth selection: `use_sa_depth=true` → SA expected depth; `use_sa_depth=false` → `(1-ratio)×expected + ratio×median`.

---

## Main Queue Messages (Paper)

Front end → back end (multiprocessing Queue):

```text
["init", cur_frame_idx, viewpoint, depth_map]
["keyframe", cur_frame_idx, viewpoint, current_window, depth_map]
["stop"]
```

Back end → front end:

```text
["init", gaussians, occ_aware_visibility, keyframes]
["keyframe", gaussians, occ_aware_visibility, keyframes]
["sync_backend", gaussians, occ_aware_visibility, keyframes]
```

Do not change these message formats unless every sender and receiver is updated together.

---

## Pose Conventions

| Variable | Semantic | Source |
|---|---|---|
| `viewpoint.T` | global **W2C** (4×4) | FrontEnd tracking writes |
| `torch.linalg.inv(viewpoint.T)` | global **C2W** (4×4) | Computed as needed |
| `cam_rot_delta` / `cam_trans_delta` | SE(3) delta for render refinement | Per-frame Adam optimizer |

**Do not change these conventions.**

---

## Paper Configuration Overview

Important configuration groups:

| Group | Controls |
|---|---|
| `Results` | save path, trajectory saving, GUI, rendering eval |
| `Dataset` | dataset type, sensor type, camera params, point sampling |
| `Training` | init/mapping/tracking iters, keyframe, window, LR, densify/prune, RSKM |
| `VOPrior` / `SimpleRGBDOdom` | VO prior type, quality gates, refinement iters |
| `Backend` | keyframe pose policy, pose sanity check |
| `Submap` | motion thresholds (TUM: 2.0m/80°), seed init, Gaussian Inheritance |
| `RAP2DGSLite` | inheritance scoring: KNN k, features, weights, selection budget |
| `LoopClosure` | mode control, keyframe retrieval, depth verify, keyframe PGO safety, Reloc3R |
| `opt_params` | Gaussian optimizer, densification, lambda_dist, lambda_sensor_normal |
| `model_params` | SH degree, data device |
| `pipeline_params` | renderer settings, use_sa, use_sa_depth, depth_ratio, depth_eps |
| `Ablation` | submap, loop closure, FDN, FFT mask, error mask, color refinement |

Three base configs must be kept in sync for generic parameters:

```text
configs/rgbd/tum/base_config.yaml
configs/rgbd/replica/base_config.yaml
configs/rgbd/scannetpp/base_config.yaml
```

Scene-specific configs should only override existing parameters.

---

## Installation

```bash
git clone https://github.com/irvingwu5/FVO-GS-SLAM.git --recursive
cd FVO-GS-SLAM
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate your_env_name
```

Depending on your CUDA and PyTorch versions, you may need to adjust versions in `environment.yml` or install PyTorch manually.

---

## Download Datasets

```bash
bash scripts/download_tum.sh      # TUM RGBD
bash scripts/download_replica.sh  # Replica
bash scripts/download_euroc.sh    # EuRoC MAV (legacy)
```

---

## Run

### TUM RGBD

```bash
python slam.py --config configs/rgbd/tum/fr1_desk.yaml --eval
python slam.py --config configs/rgbd/tum/fr3_office.yaml --eval
```

### Replica

```bash
python slam.py --config configs/rgbd/replica/office0.yaml --eval
python slam.py --config configs/rgbd/replica/office0_sp.yaml --eval  # single process
```

### ScanNet++

```bash
python slam.py --config configs/rgbd/scannetpp/8b5caf3398.yaml --eval
```

---

## Evaluation

Use `--eval` to force evaluation mode, which overrides:

```text
save_results = True
use_gui = False
eval_rendering = True
use_wandb = False
```

Output directory contains:

```text
config.yml
frame_to_submap.pt
submaps/*.ckpt
submaps/*_img_*.pt   (keyframe images for CosPlace)
point_cloud/final/point_cloud.ply
rendering evaluation outputs
trajectory and ATE outputs
```

Evaluation logs include FPS, ATE, rendering metrics, GPU memory peak (minus baseline), and final map size.

---

## Paper Ablation Switches

```yaml
Ablation:
  use_fft_mask: True          # FFT frequency mask for sampling + scale
  use_error_mask: True        # render error mask for dynamic point insertion

# Non-paper ablation switches (set to false / off for paper experiments):
Ablation:
  use_submap: False           # submap cutting (paper: single global map)
  use_loop_closure: False     # loop detection + PGO (paper: disabled)
  use_fdn: False              # FDN normal supervision (paper: disabled)

VOPrior:
  type: simple_rgbd_odom      # VO prior type (paper: Simple RGBD Odometry)

Backend:
  optimize_keyframe_pose: true        # keyframe pose optimization in back end
  optimize_keyframe_exposure: false   # keyframe exposure optimization

# Paper key configs for three depth modes:
pipeline_params:
  use_sa: true              # CUDA SA depth adjustment
  use_sa_depth: true        # use SA expected depth in loss
  depth_ratio: 1.0          # 0=expected, 1=median (only when use_sa_depth=false)
opt_params:
  use_sa_dist: false        # SA dist vetoed (improves depth L1, degrades ATE)
  lambda_dist: 0.0          # distortion loss weight (0=off)
Training:
  use_rskm: true            # RSKM random keyframe replay
  rskm_current_frame_interval: 4
  use_freq_sampling_density: true   # frequency-aware sampling density
  use_rgb_error_mask: true          # RGB error mask (sync with use_error_mask)
```

---

## Reproducibility Notes

- The system is sensitive to CUDA, PyTorch, Open3D, and differentiable Gaussian rasterizer versions.
- Recommended workflow: run a short smoke test first, then full evaluation.
- For fair comparison, keep the same dataset sequence, image resolution, tracking/mapping iterations, and ablation switches.

---

## Acknowledgement

This project is developed based on the MonoGS style Gaussian SLAM framework. The paper system features VOPrior (Simple RGBD Odometry) tracking initialization, FFT frequency-aware Gaussian seeding, error-mask guided dynamic point insertion, three depth supervision sources (expected/median/surfel-aware), RSKM random keyframe replay, and depth distortion loss.

References:
- 2D Gaussian Splatting: [Surfel-based Gaussian Rendering](https://surfelsplatting.github.io/)
- Simple RGBD Odometry: [Simple-RGBD-Odometry](https://github.com/HKCLynn/Simple-RGBD-Odometry)
- RSKM: [GS3LAM](https://github.com/lif314/GS3LAM)
- SA Depth: [GauS-SLAM](https://github.com/irvingwu5/gaus-slam)
- FFT Filter: [FGS-SLAM](https://github.com/3DV-Coder/FGS-SLAM)

---

## License

This modified version follows the license terms inherited from the original project. Please refer to `LICENSE.md` for details.
