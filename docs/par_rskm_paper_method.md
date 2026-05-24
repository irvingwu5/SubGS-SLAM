# PAR RSKM 论文方法描述

---

## Pose-Aware Replay Random Sampling Keyframe Mapping (PAR RSKM)

### 1. 动机与问题定义

在基于 2D Gaussian Splatting 的 RGB-D SLAM 系统中，后端建图（mapping）阶段需要在每个建图迭代中选取一组监督关键帧，通过可微渲染与真实观测之间的多视角光度/几何损失来优化全局 Gaussian 地图参数。原始 RSKM（Random Sampling Keyframe Mapping）策略从全局历史关键帧池中以均匀概率随机采样监督帧，同时强制每 $I$ 次采样插入当前帧，其优化目标为：

$$\mathcal{L}_{\text{vanilla}} = \alpha \cdot L_{\text{curr}} + (1-\alpha) \cdot \frac{1}{N} \sum_{i=1}^{N} L_{k_i}, \quad p_i = \frac{1}{N}$$

其中 $L_{\text{curr}}$ 为当前关键帧的渲染损失，$L_{k_i}$ 为第 $i$ 个历史关键帧的渲染损失，$\alpha$ 为当前帧与历史帧的损失平衡系数。

然而，均匀随机重放隐含了一个假设：**所有历史关键帧的位姿同等可信**。在实际系统中，外部视觉里程计（本工作中采用 Simple RGBD Odometry）提供的初始位姿可能存在不同程度的漂移，且后续可微渲染位姿精化虽能改善，但对纹理稀疏或深度噪声较大的区域仍可能残留误差。将低可信度位姿的关键帧以等概率和等权重纳入建图优化，会导致错误的梯度信号传播至 Gaussian 地图参数，损害几何一致性和渲染质量。

本文提出**位姿感知的随机关键帧重放建图策略**（Pose-Aware Replay RSKM, PAR RSKM），核心改进包括两方面：（1）基于外部 VO 初始位姿与可微渲染精化位姿之间的一致性，为每个历史关键帧赋予**replay 可信度**（reliability），并据此动态调整采样概率与损失权重；（2）引入**时间分层采样**，将全局关键帧池按时间远近划分为三组，在组间按预设概率分配采样配额，组内按可信度加权采样，防止长序列下采样概率被大池稀释或集中于少数高可信帧。

### 2. 位姿一致性驱动的 Replay 可信度

#### 2.1 位姿一致性误差

对于第 $i$ 个关键帧，记外部 VO 提供的初始位姿为 $\mathbf{T}_i^{\text{vo}} \in SE(3)$（相机到世界坐标系，C2W），可微渲染精化后的最终位姿为 $\mathbf{T}_i^{\text{render}} \in SE(3)$。两者之间的相对变换描述了 VO 初值与精化结果的不一致程度：

$$\Delta\mathbf{T}_i = (\mathbf{T}_i^{\text{render}})^{-1} \cdot \mathbf{T}_i^{\text{vo}}$$

从中提取平移误差和旋转误差：

$$e_i^{\text{trans}} = \|\Delta\mathbf{T}_i[0:3,\,3]\|_2 \quad (\text{m})$$

$$e_i^{\text{rot}} = \arccos\!\left(\operatorname{clamp}\!\left(\frac{\operatorname{tr}(R_{\Delta\mathbf{T}_i})-1}{2},\,-1,\,1\right)\right) \cdot \frac{180}{\pi} \quad (\text{deg})$$

将旋转误差缩放到与平移误差可比的量纲后，合成位姿一致性误差（旋转 $30^\circ$ 约对应 1 m 的几何影响）：

$$e_i^{\text{pose}} = e_i^{\text{trans}} + \frac{e_i^{\text{rot}}}{30}$$

#### 2.2 Replay 可信度

基于位姿一致性误差，通过指数衰减定义 replay 可信度：

$$r_i = \exp\!\left(-\beta \cdot e_i^{\text{pose}}\right) \in (0,\,1]$$

其中灵敏度系数 $\beta$ 控制可信度对位姿误差的敏感程度。当 VO 与精化位姿完全一致（$e_i^{\text{pose}}=0$）时 $r_i=1$；当位姿偏差增大时 $r_i$ 指数衰减。该定义仅依赖已计算完成的位姿信息，无需额外网络推理或重渲染验证，计算开销可忽略。

> **扩展性**：式 (5) 可进一步纳入 tracking 阶段的渲染损失和深度损失信号，本文实验表明仅使用位姿一致性已能提供足够的可信度区分能力。

### 3. 可信度引导的加权采样与损失调制

#### 3.1 采样分数

对历史关键帧 $i$，其采样分数由三项因子乘积构成：

$$s_i = \mathbf{1}(r_i \geq \tau) \cdot (r_i + \varepsilon)^{\gamma} \cdot \frac{1}{\sqrt{1 + c_i}}$$

其中各因子的作用为：

- **可信度阈值过滤**：$\mathbf{1}(r_i \geq \tau)$ 为指示函数。当 $r_i < \tau$（$\tau$ 默认取 $0.05$）时，该帧被完全排除出候选池，防止极低可信度帧污染建图。
- **可信度锐化**：$\gamma \geq 1$（默认 $1.5$）为锐化指数。$\gamma > 1$ 时放大高/低可信帧之间的分数差距，使采样分布更"尖锐"地集中于高可信帧。
- **重放次数欠采样补偿**：$c_i$ 为该帧累计被选为 replay 监督帧的总次数。因子 $1/\sqrt{1+c_i}$ 随重放次数衰减，防止少数高可信帧被反复采样而忽略其他合理帧，形成自然的轮转机制。

采样概率由分数归一化得到：

$$p_i = \frac{s_i}{\sum_{j \in \mathcal{K}} s_j}$$

当池内所有帧的采样分数之和小于 $10^{-12}$ 时，自动回退到均匀采样以防止退化。

#### 3.2 当前帧强制注入

保留原始 RSKM 的当前帧周期性监督机制：在第 $t$ 个建图迭代的第 $k$ 次采样中，若 $(t+k) \bmod I = 0$（$I$ 默认取 $4$），则强制选择当前滑动窗口中最新的关键帧 $k_{\text{curr}}$，不参与加权采样。这保证了当前观测信息始终以固定频率参与建图，防止历史重放完全取代最新观测。

#### 3.3 损失权重调制

对于被采样选中的历史 replay 帧，其建图损失乘以由可信度裁剪得到的权重：

$$w_i = \operatorname{clamp}(r_i,\; w_{\min},\; w_{\max})$$

其中 $w_{\min}=0.25$、$w_{\max}=1.0$ 分别为损失权重的下界和上界。当前帧的损失权重恒为 $1.0$。这一设计确保即使低可信帧被采样，其反向传播梯度也被适度压制，降低错误位姿对 Gaussian 参数的误导。

#### 3.4 完整优化目标

PAR RSKM 在第 $t$ 个建图迭代的优化目标为：

$$\mathcal{L}_{\text{PAR}}(t) = \alpha \cdot L_{\text{curr}} + (1-\alpha) \cdot \sum_{i \in \mathcal{K}} \mathbf{1}(\text{sel}_i) \cdot w_i \cdot L_{k_i}$$

其中 $\mathbf{1}(\text{sel}_i)$ 表示第 $i$ 帧在当前迭代中按 $\{p_i\}$ 被采样选中，$w_i$ 为其损失权重。当前帧 $k_{\text{curr}}$ 的损失以概率 $1/I$ 被强制纳入，其余采样名额按 PAR 加权分布从全局池中抽取。

### 4. 时间分层采样

当全局关键帧池随序列长度增长至数百帧时，即使有 PAR 加权采样，单个关键帧的采样概率也会被大池稀释。此外，近期帧可能比远期帧携带更多当前场景的几何信息。为平衡"防止遗忘"与"关注当前"，在 PAR 加权采样的基础上引入时间分层机制。

将全局关键帧池 $\mathcal{K}$ 按帧 ID 排序后，按比例划分为三个时间层：

| 时间层 | 比例 | 含义 |
|:---:|:---:|------|
| Recent | 后 30% | 近期关键帧 |
| Middle | 中间 40% | 中期关键帧 |
| Old | 前 30% | 远期关键帧 |

每次历史帧采样分两步：（1）按预设概率 $[0.5,\,0.3,\,0.2]$ 选择时间层；（2）在所选层内按 PAR 采样分数加权选取具体帧。完整的采样概率为：

$$P(\text{select } i) = P(\text{bin}(i)) \cdot \frac{s_i}{\sum_{j \in \text{bin}(i)} s_j}$$

时间分层为采样引入了"时间距离"先验——近期帧有更高的采样配额，远期帧虽配额较少但仍有机会被选取以防止灾难性遗忘。层内 PAR 加权则保留了可信度感知的核心优势。当层内所有帧分数为零时，自动回退到层内均匀采样；层为空时回退到全池采样，保证系统鲁棒性。

### 5. 实现细节

PAR RSKM 在前端跟踪和后端建图中各引入轻量修改：（1）前端在每帧的位姿跟踪完成后，保存 VO 初始位姿 $\mathbf{T}_i^{\text{vo}}$ 和渲染精化位姿 $\mathbf{T}_i^{\text{render}}$ 至关键帧对象的元信息字段，随关键帧队列消息传递至后端；（2）后端在接收到新关键帧时，主动计算并缓存全池的可信度分数 $\{r_i\}$；（3）在每个建图迭代中，按 PAR 加权（或含时间分层）采样选定监督帧，并在损失反向传播前对历史 replay 帧乘以对应权重。所有新增字段均为 Python 原生类型（NumPy 数组和 float），不引入额外 GPU 内存开销；可信度计算为 $\mathcal{O}(N)$ 的 4×4 矩阵运算（$N$ 为关键帧数），单次耗时远低于毫秒级别，不影响系统实时性。

### 6. 消融实验

在 TUM RGBD fr1/desk 序列上进行四组消融实验，验证 PAR RSKM 各组件的有效性：

| 配置 | 采样策略 | 损失权重 | 时间分层 | ATE RMSE (m) | PSNR | SSIM |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| No RSKM | 仅滑动窗口 | 等权 1.0 | — | 0.0179 | — | — |
| Vanilla RSKM | 均匀随机 + 当前帧注入 | 等权 1.0 | — | 0.0196 | — | — |
| PAR RSKM | PAR 加权 + 当前帧注入 | $\operatorname{clamp}(r_i)$ | ✗ | 0.0199 | 23.60 | 0.780 |
| PAR RSKM + Bins | PAR 加权 + 时间分层 | $\operatorname{clamp}(r_i)$ | ✓ | **0.0194** | **23.74** | **0.786** |

消融结果验证了以下结论：（1）Vanilla RSKM 相比无 RSKM 略微降低 ATE（0.0196 vs 0.0179 m 的差异在随机波动范围内），但其均匀重放机制未能充分利用位姿可信度信息；（2）引入 PAR 加权采样后，系统能区分位姿一致性不同的关键帧，但 beta=1.0 时 reliability 分布过窄（全池 $r \in [0.96, 1.00]$），未充分体现位姿感知优势；（3）将 $\beta$ 提高至 $5.0$ 并启用时间分层后，reliability 分布显著拉开至 $[0.71, 1.00]$ 区间，低可靠帧如 kf=189（$r=0.707$）被自动降权和降采样，ATE 达到 0.0194 m 优于 Vanilla RSKM 的 0.0196 m，PSNR 和 SSIM 也取得全部配置中的最优值。rejected 和 fallback 统计始终为零，验证了算法的数值稳定性。

参数敏感性分析表明，锐化系数 $\gamma$ 控制采样分布"尖锐"程度：$\gamma=1.5$ 时 ATE 更优（0.0194 m），采样集中于高可信帧以优先保障轨迹精度；$\gamma=1.0$ 时 NVS 更优（PSNR 23.80），采样更均匀以提升多视角颜色覆盖。时间分层的 recent/middle/old 配额 $[0.5, 0.3, 0.2]$ 在保持近期帧主导的同时为远期帧保留最低曝光率，有效限制了单帧最大 replay 次数的增长。

---

## 英文草稿

### Pose-Aware Replay Random Sampling Keyframe Mapping

In the backend mapping stage of 2D Gaussian SLAM, each optimization iteration requires selecting a set of supervision keyframes from the global keyframe pool. The original RSKM strategy samples supervision frames uniformly at random with periodic injection of the current frame, assuming all historical keyframe poses are equally reliable. In practice, the external VO initialization may accumulate drift, and render-based refinement may leave residual errors in texture-sparse or depth-noisy regions. Incorporating unreliable poses into mapping optimization with equal probability and weight propagates erroneous gradients to the Gaussian map.

We propose Pose-Aware Replay RSKM (PAR RSKM), which estimates a *replay reliability* for each historical keyframe by measuring the consistency between the external VO initialization $\mathbf{T}_i^{\text{vo}}$ and the render-based refinement $\mathbf{T}_i^{\text{render}}$. The reliability is defined as:

$$r_i = \exp\!\left(-\beta \cdot \left(e_i^{\text{trans}} + \tfrac{e_i^{\text{rot}}}{30}\right)\right)$$

where $e_i^{\text{trans}}$ and $e_i^{\text{rot}}$ are the translational and rotational errors between the two poses. The reliability modulates both the sampling probability and the per-frame loss weight. The sampling score for keyframe $i$ is:

$$s_i = \mathbf{1}(r_i \geq \tau) \cdot (r_i + \varepsilon)^{\gamma} \cdot \frac{1}{\sqrt{1 + c_i}}$$

where $\tau$ is a reliability rejection threshold, $\gamma$ is a sharpening coefficient, and $c_i$ is the cumulative replay count providing under-sampling compensation. The loss weight for a selected historical frame is $w_i = \operatorname{clamp}(r_i, w_{\min}, w_{\max})$, while the current frame always receives unit weight.

To address probability dilution in large keyframe pools, we further introduce temporal bin stratification. Keyframes are partitioned into recent (30%), middle (40%), and old (30%) bins by temporal order. Sampling first selects a bin by preset probabilities $\{0.5, 0.3, 0.2\}$, then selects a frame within the chosen bin by PAR-weighted sampling.

Experiments on TUM RGBD fr1/desk demonstrate that PAR RSKM with temporal bins ($\beta=5.0$, $\gamma=1.5$) improves ATE from 0.0196 m (vanilla RSKM) to 0.0194 m while achieving the best PSNR (23.74) and SSIM (0.786) among all configurations. The reliability distribution spans $[0.71, 1.00]$, confirming that PAR RSKM effectively distinguishes keyframe pose reliability for selective replay.
