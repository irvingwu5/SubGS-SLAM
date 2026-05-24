# TUM fr1_desk 渐进消融实验分析报告

> 实验日期：2026-05-13
> 配置目录：`configs/rgbd/tum/ablation_fr1/`
> 日志目录：`/opt/results/Ablation/aba_fr1/`
> 渐进顺序：A(All OFF) → B(+VOPrior) → C(+RSKM) → D(+FFT+Freq) → E(+Error+RGB) → F(+SA depth) → G(+SA dist)

## 一、核心指标总览

| Round | 新增模块 | ATE (m) | 高斯数 | 时间 (s) | FPS | 显存 (MB) |
|---|---|---|---|---|---|---|
| A | All OFF | 0.02148 | 53,821 | 397.2 | 1.49 | 4,040 |
| B | +VOPrior | 0.01788 | 46,870 | 312.9 | 1.89 | 4,215 |
| C | +RSKM | 0.01958 | 38,939 | 304.0 | 1.95 | 4,233 |
| D | +FFT+Freq | 0.01786 | **19,330** | 328.5 | 1.80 | 4,219 |
| E | +Error+RGB | **0.01770** | 20,279 | 327.8 | 1.81 | 4,225 |
| F | +SA depth | 0.02116 | 19,515 | 336.7 | 1.76 | 4,241 |
| G | +SA dist | 0.02404 | 21,498 | 333.4 | 1.78 | 4,201 |

> 注：PSNR/SSIM/LPIPS/Depth L1 未开启 `eval_rendering`，未采集。

## 二、边际贡献分析

| 步骤 | 模块 | ATE 变化 | 幅度 | 判定 |
|---|---|---|---|---|
| B-A | VOPrior | 0.02148→0.01788 | **-16.8%** | 强正面 |
| C-B | RSKM | 0.01788→0.01958 | +9.5% | 负面 |
| D-C | FFT+Freq | 0.01958→0.01786 | **-8.8%** | 正面（逆转 RSKM 损伤） |
| E-D | Error+RGB | 0.01786→0.01770 | -0.9% | 中性 |
| F-E | SA depth | 0.01770→0.02116 | **+19.5%** | 严重负面 |
| G-F | SA dist | 0.02116→0.02404 | +13.6% | 负面 |

## 三、ATE 轨迹

```
        A      B(VO)  C(RSKM) D(FFT)  E(Err)  F(SAD)  G(Full)
start   .007   .006   .005    .012    .009    .006    .012
25%     .027   .017   .023    .020    .016    .026    .027
50%     .025   .018   .021    .019    .018    .024    .025
75%     .022   .018   .020    .018    .018    .022    .025
final   .021   .018   .020    .018    .018    .021    .024
```

- **B (VOPrior)**：全程最稳定，ATE 收敛到 0.018 后不再漂移
- **C (RSKM)**：初期极好（0.005），但 25% 处跳升到 0.023，之后缓慢收敛
- **D (FFT+Freq)**：起点差（0.012），但全程改善趋势，最终追平 B
- **E (Error+RGB)**：最平滑的下降曲线，最终 ATE 最优
- **F (SA depth)**：起点好但全程高 ATE，被 SA expected depth 拖累
- **G (SA dist)**：起点差、全程差，SA dist 叠加后更恶化

## 四、模块结论

### 强正面

**VOPrior**：-16.8% ATE，同时减少 13% 运行时间（312s vs 397s）。Simple-RGBD-Odometry 在 fr1_desk 的小运动场景中提供稳定初值。

### 正面

**FFT+Freq**：-8.8% ATE（相比前轮），且高斯数从 39K 暴跌到 **19K（-50%）**。频率感知采样密度在 TUM 上的精度-效率 trade-off 非常优秀。

### 中性

**Error+RGB mask**：ATE 几乎不变，高斯数微增，边际贡献不明显。

### 负面

**RSKM**：+9.5% ATE。与 Replica room0 一致，RSKM 在全局关键帧池中随机采样，早期关键帧位姿不精确时重放会传播误差。fr1_desk ~600 帧，关键帧池约 120 个，早期帧误差影响显著。

### 严重负面

- **SA depth**：**+19.5% ATE**。TUM Kinect 含噪声，SA expected depth 的 outlier 拉回在边界处产生错误深度梯度。
- **SA dist**：+13.6% ATE。SA variance 进 loss 后放大深度不确定性，加速漂移。

## 五、与 Replica room0 对比

| 模块 | Replica room0 | TUM fr1_desk |
|---|---|---|
| VOPrior | -65% ATE | -17% ATE |
| RSKM | +54% ATE | +10% ATE |
| FFT+Freq | +8% ATE | -9% ATE |
| SA depth | +69% ATE | +20% ATE |
| SA dist | - | +14% ATE |

**一致性**：VOPrior 正面、RSKM 负面、SA depth 负面——两个数据集高度一致。

**差异**：FFT+Freq 在 TUM 上正面（-9%）但在 Replica 上中性偏负（+8%）。TUM 图像纹理丰富，FFT 高通滤波能有效区分几何边界和纹理平面；Replica 合成纹理简单，FFT 高频区分度低。

## 六、建议

1. **VOPrior 必须开启**——两个数据集均强正面
2. **RSKM 需加时间窗口**——当前全局采样在无子图策略下伤害明显，建议 `rskm_window` 限制最近 N 个关键帧
3. **SA depth 在 Kinect 噪声数据和合成数据上均有害**——建议默认关闭，或仅在深度传感器质量高的场景尝试
4. **FFT+Freq 组合效果良好**——TUM 上高斯数减半且 ATE 改善，建议默认开启
5. G_full 比 A_all_off 还差（0.024 vs 0.021），说明 SA depth+SA dist 的负面效应超过了 VOPrior+FFT 的正面贡献
