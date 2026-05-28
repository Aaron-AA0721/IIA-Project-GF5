from __future__ import annotations

import numpy as np


def forward_kinematics(
    joints: list[object],
    local_rotations: list[np.ndarray],
    root_offset: np.ndarray,
    topological_order: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    world_rotations = np.zeros((len(joints), 3, 3))
    world_positions = np.zeros((len(joints), 3))
    joint_completed = [False] * len(joints)
    def get_world_transform(joint):
        
        if(joint == -1):
            return np.eye(3), root_offset
        if(joint_completed[joint]):
            return world_rotations[joint], world_positions[joint]
        parent = joints[joint].parent
        local_rotation = local_rotations[joint]
        local_position = joints[joint].translation
        
        # if(parent != -1):
        parent_rotation, parent_position = get_world_transform(parent)
        world_rotation = parent_rotation @ local_rotation
        world_position = parent_position + parent_rotation @ local_position
        joint_completed[joint] = True
        return world_rotation, world_position
        # else:
        #     world_position = local_position + root_offset
        #     joint_completed[joint] = True
        #     return local_rotation, world_position
        
    
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
    
    for joint in range(len(joints)):
        world_rotations[joint], world_positions[joint] = get_world_transform(joint)

        # if(joints[joint].parent != -1):
        #     parent = joints[joint].parent
        #     world_rotations[joint] = world_rotations[parent] @ local_rotation
        #     world_positions[joint]  = world_positions[parent] + world_rotations[parent] @ local_position
            
        # else:
        #     world_positions[joint] = local_position + root_offset
        #     world_rotations[joint] = local_rotation

    return world_rotations, world_positions

    raise NotImplementedError(
        "Implement forward_kinematics() in student_submission/part1_fk.py"
    )
    world_positions = np.zeros((joint_count, 3), dtype=np.float32)
    return world_rotations, world_positions
