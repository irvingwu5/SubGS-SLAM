"""
Minimal test to determine Open3D PoseGraphEdge transformation direction.

Problem: The code passes T_o3d = inv(T_source_from_target) as the edge
transformation. This test verifies whether that direction is correct for
Open3D's GlobalOptimizationLevenbergMarquardt.

Runs without any project dependencies — only needs numpy + open3d.
"""

import numpy as np


def _rot_x(deg):
    """Rotation around X axis by deg degrees."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def _rot_error_deg(R_a, R_b):
    """Geodesic rotation error between two 3x3 rotation matrices, in degrees."""
    R = R_a @ R_b.T
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def _se3_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def run_test(label, T_edge_candidate):
    """
    Run a single O3D optimization with two nodes and one edge.

    node0.pose = I (fixed)
    node1.pose = T_gt (optimizable)
    edge: node0 → node1 with transformation = T_edge_candidate

    If T_edge_candidate is correct, optimized node1.pose should ≈ T_gt.
    """
    import open3d as o3d

    # Ground truth: node1 C2W = translation [1, 2, 0.5] + rotation 10 deg around X
    R_gt = _rot_x(10.0)
    t_gt = np.array([1.0, 2.0, 0.5])
    T_gt = _se3_to_T(R_gt, t_gt)

    # Node0 starts at identity (fixed)
    # Node1 starts at a deliberately perturbed pose (will be optimized toward T_gt)
    R_perturbed = _rot_x(25.0)
    t_perturbed = np.array([0.3, 3.0, 0.1])
    T_perturbed = _se3_to_T(R_perturbed, t_perturbed)

    g = o3d.pipelines.registration.PoseGraph()
    g.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    g.nodes.append(o3d.pipelines.registration.PoseGraphNode(T_perturbed.copy()))

    # Single edge from node0 to node1
    info = np.eye(6) * 100.0
    g.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(
            0, 1, T_edge_candidate, info, uncertain=False,
        )
    )

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.02 * 1.5,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        g,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    opt_pose = np.array(g.nodes[1].pose, dtype=np.float64)
    t_err = float(np.linalg.norm(opt_pose[:3, 3] - t_gt))
    r_err = _rot_error_deg(opt_pose[:3, :3], R_gt)

    print(f"  {label}: t_err={t_err:.6f}m  r_err={r_err:.6f}deg  "
          f"({'PASS' if t_err < 0.01 and r_err < 0.1 else 'FAIL'})")
    return t_err, r_err


def main():
    print("=" * 68)
    print("Open3D PoseGraphEdge direction test")
    print("=" * 68)
    print()
    print("Ground truth: node0 = I, node1 = R_x(10°) + [1, 2, 0.5]")
    print("node1 initially perturbed, should converge back to ground truth.")
    print()

    # Ground truth poses
    R_gt = _rot_x(10.0)
    t_gt = np.array([1.0, 2.0, 0.5])
    T_gt = _se3_to_T(R_gt, t_gt)

    # Candidate A: T_edge = inv(c2w_0) @ c2w_1 = I @ T_gt = T_gt
    # This is what T_source_from_target computes when src=0, tgt=1.
    T_candidate_A = np.linalg.inv(np.eye(4)) @ T_gt  # = T_gt

    # Candidate B: T_edge = inv(c2w_1) @ c2w_0 = inv(T_gt) @ I = inv(T_gt)
    # This is what the code's inversion produces: inv(T_source_from_target).
    T_candidate_B = np.linalg.inv(T_gt) @ np.eye(4)  # = inv(T_gt)

    print("Testing edge directions (lower error = correct convention):")
    print()
    t_a, r_a = run_test("A) T_edge = inv(c2w_0) @ c2w_1     (no inversion) ", T_candidate_A)
    t_b, r_b = run_test("B) T_edge = inv(T_source_from_target) (current)  ", T_candidate_B)
    print()

    # Conclusion
    print("-" * 68)
    if t_a < 0.01 and r_a < 0.1:
        print("CONCLUSION: Candidate A is CORRECT.")
        print("  O3D constraint: pose_target = pose_source @ T_edge")
        print("  T_edge should be T_source_from_target (= inv(c2w_src) @ c2w_tgt)")
        print("  The current code's inv() at keyframe_pgo.py:777 is WRONG.")
        print("  FIX: Remove the inversion, use T_edge = T_source_from_target directly.")
        print()
        print("  For pose_graph_edge(si, ti, transformation):")
        print("    transformation = inv(c2w_source) @ c2w_target  (no extra inv)")
    elif t_b < 0.01 and r_b < 0.1:
        print("CONCLUSION: Candidate B is CORRECT.")
        print("  O3D constraint: pose_source = pose_target @ T_edge  (or equiv form)")
        print("  T_edge should be inv(T_source_from_target) = T_target_from_source")
        print("  The current code's inv() at keyframe_pgo.py:777 is CORRECT.")
        print("  No fix needed for the inversion direction.")
    else:
        print("WARNING: Neither candidate converged to ground truth!")
        print("  O3D convention may differ from both tested hypotheses.")
        print(f"  Candidate A errors: t={t_a:.3f}m r={r_a:.3f}deg")
        print(f"  Candidate B errors: t={t_b:.3f}m r={r_b:.3f}deg")
    print("-" * 68)


if __name__ == "__main__":
    main()
