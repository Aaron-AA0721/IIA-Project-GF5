from __future__ import annotations

import numpy as np


def make_one_hot_skinning_weights(weights: np.ndarray) -> np.ndarray:
    """Student part-2 task: convert a dense weight matrix into one-hot weights."""
    raise NotImplementedError(
        "Implement make_one_hot_skinning_weights() in student_submission/part2_skinning.py"
    )


def skin_smpl_mesh(
    model_data: object,
    world_rotations: np.ndarray,
    world_positions: np.ndarray,
    *,
    use_blended_weights: bool,
) -> np.ndarray:
    """Student part-2 task: pose the SMPL mesh with one-hot or blended weights."""
    raise NotImplementedError(
        "Implement skin_smpl_mesh() in student_submission/part2_skinning.py"
    )
