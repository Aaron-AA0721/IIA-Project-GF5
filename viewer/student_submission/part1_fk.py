from __future__ import annotations

import numpy as np


def forward_kinematics(
    joints: list[object],
    local_rotations: list[np.ndarray],
    root_offset: np.ndarray,
    topological_order: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Student part-1 implementation.

    Expected inputs:
    - joints: each joint has `.parent` and `.translation`
    - local_rotations: one 3x3 local rotation matrix per joint
    - root_offset: global translation applied to the root
    - topological_order: optional parent-before-child traversal order

    Expected outputs:
    - world_rotations: shape (J, 3, 3)
    - world_positions: shape (J, 3)
    """
    joint_count = len(joints)
    world_rotations = np.tile(
        np.eye(3, dtype=np.float32)[None, :, :],
        (joint_count, 1, 1),
    )
    world_positions = np.zeros((joint_count, 3), dtype=np.float32)
    return world_rotations, world_positions
