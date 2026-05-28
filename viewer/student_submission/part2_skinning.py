from __future__ import annotations

import numpy as np


def make_one_hot_skinning_weights(weights: np.ndarray) -> np.ndarray:
    """Student part-2 task: convert a dense weight matrix into one-hot weights."""
    one_hot_weights = np.zeros_like(weights)
    for vertex in range(weights.shape[0]):
        max_weight_index = np.argmax(weights[vertex])
        one_hot_weights[vertex][max_weight_index] = 1
    return one_hot_weights


def skin_smpl_mesh(
    model_data: object,
    world_rotations: np.ndarray,
    world_positions: np.ndarray,
    *,
    use_blended_weights: bool,
) -> np.ndarray:
    """Student part-2 task: pose the SMPL mesh with one-hot or blended weights."""
    if use_blended_weights:
        skinning_weights = model_data.skinning_weights
    else:
        skinning_weights = make_one_hot_skinning_weights(model_data.skinning_weights)
    rest_vertices = np.asarray(model_data.rest_vertices, dtype=np.float32)
    rest_joints = np.asarray(model_data.rest_joints, dtype=np.float32)
    current_vertices = np.zeros_like(rest_vertices)

    
    # for vertex in range(rest_vertices.shape[0]):
    #     for joint in range(world_rotations.shape[0]):
    #         TransformationMatrix4x4 = np.eye(4)
    #         TransformationMatrix4x4[:3, :3] = world_rotations[joint]
    #         TransformationMatrix4x4[:3, 3] = world_positions[joint]
    #         ExtendedRestVertex = np.append(rest_vertices[vertex] - rest_joints[joint], 1.0)
    #         ExpectedPosition = TransformationMatrix4x4 @ ExtendedRestVertex
    #         current_vertices[vertex] += skinning_weights[vertex][joint] * ExpectedPosition[:3]

    for joint in range(world_rotations.shape[0]):
        posed_vertices = (rest_vertices - rest_joints[joint]) @ world_rotations[joint].T
        posed_vertices += world_positions[joint]
        current_vertices += skinning_weights[:, joint : joint + 1] * posed_vertices
    return current_vertices
