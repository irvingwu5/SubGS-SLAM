"""
验证 rend_normal 法线方向修复的正确性。
模拟 CUDA rasterizer 的 viewmatrix 惯例 (W2C^T) + Python 端解变换。
对比修复前后的法线方向。
用法: python tests/verify_normal_fix.py
"""
import torch
import numpy as np


def verify_normal_fix():
    print("=" * 70)
    print("rend_normal 法线方向修复验证")
    print("=" * 70)

    # 模拟 surfel 世界法线 (指向相机)
    n_world = torch.tensor([
        [-1.0, 0.0, 0.0],
        [-0.95, 0.1, 0.0],
        [-0.9, -0.05, 0.0],
    ])
    w = torch.tensor([0.5, 0.3, 0.2])
    correct_world = sum(w[i] * n_world[i] for i in range(3))

    test_angles = [0, 10, 30, 45, 60, 90, 135, 180]
    all_pass = True

    for angle_deg in test_angles:
        theta = np.radians(angle_deg)

        # 绕 Y 轴旋转 (常见 SLAM 运动)
        R_w2c = torch.tensor([
            [np.cos(theta), 0.0, -np.sin(theta)],
            [0.0, 1.0, 0.0],
            [np.sin(theta), 0.0, np.cos(theta)],
        ])
        t_w2c = torch.tensor([1.0, 0.2, 0.5])
        W2C = torch.eye(4)
        W2C[:3, :3] = R_w2c
        W2C[:3, 3] = t_w2c

        # 当前代码惯例: T = W2C, world_view_transform = T^T
        T = W2C
        world_view_transform = T.transpose(0, 1)
        R_c2w = world_view_transform[:3, :3]  # viewmatrix 的 3x3

        # --- 模拟 CUDA 累积 ---
        n_cuda_list = []
        for i in range(3):
            n_cuda_i = R_c2w @ n_world[i]
            # 模拟 backface culling: 确保法线指向相机方向
            p_world = torch.zeros(3)
            p_view = R_c2w @ p_world
            cos = -torch.dot(p_view, n_cuda_i)
            if cos < 0:  # cos <= 0 时翻转
                n_cuda_i = -n_cuda_i
            n_cuda_list.append(n_cuda_i)
        N_cuda = sum(w[i] * nc for i, nc in enumerate(n_cuda_list))

        # --- 旧代码: @ world_view_transform[:3,:3].T (= R_w2c) ---
        old_result = N_cuda @ world_view_transform[:3, :3].T
        old_err = float(
            1.0
            - torch.dot(
                old_result / old_result.norm(),
                correct_world / correct_world.norm(),
            )
        )

        # --- 新代码: @ world_view_transform[:3,:3] (= R_c2w) ---
        new_result = N_cuda @ world_view_transform[:3, :3]
        new_err = float(
            1.0
            - torch.dot(
                new_result / new_result.norm(),
                correct_world / correct_world.norm(),
            )
        )

        status = "PASS" if new_err < 1e-5 else "FAIL"
        if new_err >= 1e-5:
            all_pass = False

        print(
            f"  旋转 {angle_deg:>4}°: "
            f"旧方向误差={old_err:.4f}  "
            f"新方向误差={new_err:.6f}  "
            f"[{status}]"
        )

    # 绕 Z 轴测试
    print("  --- 绕 Z 轴 ---")
    for angle_deg in [30, 90]:
        theta = np.radians(angle_deg)
        R_w2c = torch.tensor([
            [np.cos(theta), np.sin(theta), 0.0],
            [-np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        t_w2c = torch.tensor([0.3, -0.1, 1.2])
        W2C = torch.eye(4)
        W2C[:3, :3] = R_w2c
        W2C[:3, 3] = t_w2c

        T = W2C
        world_view_transform = T.transpose(0, 1)
        R_c2w = world_view_transform[:3, :3]

        n_cuda_list = []
        for i in range(3):
            n_cuda_i = R_c2w @ n_world[i]
            p_world = torch.zeros(3)
            p_view = R_c2w @ p_world
            cos = -torch.dot(p_view, n_cuda_i)
            if cos < 0:
                n_cuda_i = -n_cuda_i
            n_cuda_list.append(n_cuda_i)
        N_cuda = sum(w[i] * nc for i, nc in enumerate(n_cuda_list))

        old_err = float(
            1.0
            - torch.dot(
                (N_cuda @ world_view_transform[:3, :3].T)
                / (N_cuda @ world_view_transform[:3, :3].T).norm(),
                correct_world / correct_world.norm(),
            )
        )
        new_err = float(
            1.0
            - torch.dot(
                (N_cuda @ world_view_transform[:3, :3])
                / (N_cuda @ world_view_transform[:3, :3]).norm(),
                correct_world / correct_world.norm(),
            )
        )
        status = "PASS" if new_err < 1e-5 else "FAIL"
        if new_err >= 1e-5:
            all_pass = False
        print(
            f"  旋转 {angle_deg:>4}°: "
            f"旧方向误差={old_err:.4f}  "
            f"新方向误差={new_err:.6f}  "
            f"[{status}]"
        )

    print()
    if all_pass:
        print("✓ 所有测试通过：修复后法线方向始终正确")
    else:
        print("✗ 有测试失败")
    return all_pass


if __name__ == "__main__":
    verify_normal_fix()
