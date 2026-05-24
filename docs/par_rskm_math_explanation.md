# PAR RSKM 数学描述与通俗化解释

## 1. 问题定义

### 1.1 背景

在 RGBD 2D Gaussian SLAM 中，Backend 建图阶段需要从历史关键帧池中选取监督帧来优化全局 Gaussian 地图。原始 RSKM 的优化目标为：

$$J_{\text{vanilla}}(t) = \alpha \cdot L_{\text{current}} + (1-\alpha) \cdot \frac{1}{N}\sum_{i=1}^{N} L_{k_i}$$

其中每个历史关键帧被选中的概率相等：$p_i = \frac{1}{N}$。

**问题**：并非所有历史关键帧位姿都同等可信。VO 初始位姿可能存在漂移，render-based tracking 精化后仍可能有残留误差。将低可信位姿的关键帧以等权重纳入建图优化，会向 Gaussian 地图传播错误的梯度信号。

### 1.2 核心思想

PAR RSKM 将均匀随机重放改造为**位姿可信度感知**的加权重放。每帧被分配一个"可信分"（reliability），由外部 VO 初值与渲染精化位姿之间的一致性决定。高分帧优先重放，低分帧降权或跳过。

---

## 2. 完整数学描述

### 2.1 符号定义

| 符号 | 含义 |
|---|---|
| $\mathcal{K} = \{1, 2, \dots, N\}$ | 全局历史关键帧池 |
| $T_i^{\text{vo}}$ | 第 $i$ 帧的 VO 初始位姿（C2W） |
| $T_i^{\text{render}}$ | 第 $i$ 帧的渲染跟踪精化位姿（C2W） |
| $r_i \in [0, 1]$ | 第 $i$ 帧的 replay 可信度（reliability） |
| $p_i$ | 第 $i$ 帧被选为 replay 监督帧的概率 |
| $w_i$ | 第 $i$ 帧被选中后的 loss 权重 |
| $c_i$ | 第 $i$ 帧的历史 replay 次数 |
| $k_{\text{curr}}$ | 当前关键帧（滑动窗口中最新的帧） |

### 2.2 位姿一致性误差

定义 VO 初值与渲染精化位姿之间的相对变换：

$$\Delta T_i = (T_i^{\text{render}})^{-1} \cdot T_i^{\text{vo}}$$

平移误差：

$$e_i^{\text{trans}} = \|\Delta T_i[0:3, 3]\|_2 \quad (\text{米})$$

旋转误差：

$$e_i^{\text{rot}} = \arccos\left(\text{clamp}\left(\frac{\text{tr}(R_{\Delta T_i}) - 1}{2}, -1, 1\right)\right) \cdot \frac{180}{\pi} \quad (\text{度})$$

合成 pose error（旋转缩放到与平移可比量纲）：

$$e_i^{\text{pose}} = e_i^{\text{trans}} + \frac{e_i^{\text{rot}}}{30}$$

> 注：旋转 30° 大约等价于平移 1m 的几何影响，因此用 30 作为归一化系数。

### 2.3 Replay 可信度

第 $i$ 帧的可信度由 pose error 经指数衰减得到：

$$r_i = \exp\left(-\beta_{\text{pose}} \cdot e_i^{\text{pose}}\right)$$

其中 $\beta_{\text{pose}}$ 是缩放系数（默认 1.0）。

**性质**：
- 当 $e_i^{\text{pose}} = 0$（VO 与 render 完全一致）：$r_i = 1.0$
- 当 $e_i^{\text{pose}} \to \infty$：$r_i \to 0$
- $\beta_{\text{pose}}$ 越大，对位姿误差越敏感

第一版仅使用 pose consistency。后续可扩展加入 tracking loss 和 depth loss：

$$r_i = \exp\left(-\beta_{\text{pose}} \cdot e_i^{\text{pose}} - \beta_{\text{track}} \cdot L_i^{\text{track}} - \beta_{\text{depth}} \cdot L_i^{\text{depth}}\right) \cdot \rho_i^{\text{valid}}$$

当前实现中 $\beta_{\text{track}} = \beta_{\text{depth}} = 0$，$\rho_i^{\text{valid}} = 1$。

### 2.4 采样分数与采样概率

采样分数引入三个机制：可信度阈值过滤、可信度锐化、重放次数欠采样补偿。

$$s_i = \mathbf{1}(r_i \geq \tau_r) \cdot (r_i + \varepsilon)^{\gamma} \cdot \frac{1}{\sqrt{1 + c_i}}$$

其中：
- $\tau_r$：最低可信度阈值（默认 0.05）。$r_i < \tau_r$ 的帧采样分数为 0，被完全排除
- $\gamma$：锐化系数（默认 1.5）。$\gamma > 1$ 拉大高/低可信帧之间的分数差距，使采样更加"尖锐"
- $\varepsilon$：数值稳定项（默认 $10^{-6}$），防止 $r_i=0$ 时 $(0)^\gamma$ 退化为 0
- $c_i$：该帧历史被 replay 的总次数。$\frac{1}{\sqrt{1+c_i}}$ 是欠采样补偿因子——被 replay 越多的帧，分数越低，防止少数高可信帧被反复采样

采样概率由分数归一化得到：

$$p_i = \frac{s_i}{\sum_{j \in \mathcal{K}} s_j}$$

**Fallback**：当 $\sum_j s_j < 10^{-12}$（所有帧分数均为 0）时，自动回退到均匀采样：$p_i = \frac{1}{N}$。

### 2.5 当前帧强制注入

保留原始 RSKM 的当前帧周期性监督机制：

$$\text{若 } (t + k) \bmod I = 0 \text{，则必选 } k_{\text{curr}}$$

其中 $t$ 为当前 mapping iteration，$k$ 为采样序号，$I$ 为 `rskm_current_frame_interval`（默认 4）。

### 2.6 Loss 权重

被选中的历史 replay 帧，其 mapping loss 乘以可信度裁剪权重：

$$w_i = \text{clamp}(r_i, w_{\min}, w_{\max})$$

其中 $w_{\min} = 0.25$，$w_{\max} = 1.0$。

当前帧 loss 权重始终为 1.0，不受 $r_i$ 影响：

$$w_{\text{curr}} = 1.0$$

### 2.7 PAR RSKM 优化目标

$$J_{\text{PAR}}(t) = \alpha \cdot L_{\text{current}} + (1-\alpha) \cdot \sum_{i \in \mathcal{K}} \mathbf{1}(\text{selected}_i) \cdot w_i \cdot L_{k_i}$$

其中 $\mathbf{1}(\text{selected}_i)$ 表示第 $i$ 帧在当前迭代中被采样选中，采样概率由 $\{p_i\}$ 决定。

---

## 3. 时间分层采样扩展

### 3.1 动机

当全局关键帧池 $\mathcal{K}$ 增大到数百帧后，即使有加权采样，采样概率也会被大池子稀释。同时，少数极高可信帧可能持续主导采样。时间分层为采样引入**时间距离先验**，使近期帧优先被重放。

### 3.2 时间分箱

将 $\mathcal{K}$ 中关键帧按帧 ID 排序（ID 越大越新），按比例划分为三组：

| Bin | 比例 | 含义 |
|---|---|---|
| recent | 最后 30% | 近期关键帧 |
| middle | 中间 40% | 中期关键帧 |
| old | 最早 30% | 远期关键帧 |

### 3.3 分层采样

采样分两步：

**Step 1**：按预设概率选择 bin：

$$P(\text{bin} = \text{recent}) = 0.5, \quad P(\text{bin} = \text{middle}) = 0.3, \quad P(\text{bin} = \text{old}) = 0.2$$

**Step 2**：在所选 bin 内按 PAR score 加权采样（与 §2.4 相同）：

$$p_i^{\text{bin}} = \frac{s_i}{\sum_{j \in \text{bin}} s_j}$$

### 3.4 采样概率完整表达式（含时间分层）

$$P(\text{select } i) = \underbrace{P(\text{bin}(i))}_{\text{bin 先验}} \cdot \underbrace{\frac{s_i}{\sum_{j \in \text{bin}(i)} s_j}}_{\text{bin 内 PAR 加权}}$$

其中 $s_i$ 定义同 §2.4。

### 3.5 容错

- Bin 内所有 score 为 0：先在 bin 内 uniform fallback
- Bin 为空：回退到全池采样
- 全局 score 全 0：回退到全局 uniform

---

## 4. 通俗化理解

### 4.1 用一个比喻理解 PAR RSKM

想象你在复习备考，手上有 100 份过去的错题（历史关键帧）。原始 RSKM 的做法是：每轮随机抽 10 份错题重做，每份权重一样。

但有些错题是你状态不好时做的——当时脑子不清楚，题目本身可能记错了。这些"低质量错题"如果和高质量错题一样频繁复习，反而会误导你。

PAR RSKM 的做法是：

1. **给每份错题打分**（reliability）：看当时 VO 初值和最终精化结果之间的差距。差距越小，说明这份题目是"清醒时做的"，分数越高。
2. **高分优先抽**（加权采样）：分数高的错题被抽到的概率大。分数低于阈值的题目直接排除。
3. **适度降权**（loss weight）：即使被抽到，低分题权重也被压低，不会对学习产生太大影响。
4. **防刷题**（under-sampling）：被反复抽到的题目，分数逐渐降低，避免"刷同一道题"。

### 4.2 时间分层采样的比喻

题目池变大后，出现新问题：200 道题里即使按分数抽，好题也可能被淹没。

时间分层加了一个规则：**先按"时间远近"分三堆**——最近做的题（recent）、一段时间前的题（middle）、很久以前的题（old）。然后**优先抽最近的题**（50% 概率），再抽中间的（30%），最后抽旧的（20%）。堆内仍按分数抽。

这保证了新题不会被老题挤掉，老题也不会完全被遗忘。

### 4.3 一句话总结

> **PAR RSKM 让 SLAM 系统在建图时"知道哪些过去的关键帧值得信赖"，优先重放高可信帧，抑制低可信帧的干扰；时间分层则保证新旧关键帧都有合理的曝光机会。**

---

## 5. 消融实验配置对照

| 模式 | 采样方式 | Loss 权重 | 时间分层 |
|---|---|---|---|
| No RSKM | 仅滑动窗口 | 等权 1.0 | — |
| Vanilla RSKM | 均匀随机 + 当前帧周期注入 | 等权 1.0 | — |
| PAR RSKM | PAR 加权 + 当前帧周期注入 | $w_i = \text{clip}(r_i)$ | 默认关闭 |
| PAR RSKM + Bins | bin 先验 + bin 内 PAR 加权 | $w_i = \text{clip}(r_i)$ | 开启 |

---

## 6. 参数速查

| 参数 | 默认值 | 调整直觉 |
|---|---|---|
| $\beta_{\text{pose}}$ | 1.0 | 越大越"严格"，可靠性区分度越大 |
| $\gamma$ | 1.5 | 越大采样越"尖锐"（集中高分帧） |
| $\tau_r$ | 0.05 | 低于此分的帧被排除 |
| $w_{\min}$ | 0.25 | 最不可信帧的最低 loss 权重 |
| $w_{\max}$ | 1.0 | 最可信帧的最高 loss 权重 |
| $I$ | 4 | 每 N 次采样中至少 1 次当前帧 |
| recent/middle/old | 0.5/0.3/0.2 | 调整时间偏好的分布 |
