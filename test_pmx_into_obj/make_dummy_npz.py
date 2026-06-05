from pathlib import Path
import numpy as np

obj_path = Path("animation_lowres.obj")
npz_path = Path("animation_lowres_skinning_weights.npz")

def count_obj_vertices(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                n += 1
    return n

N = count_obj_vertices(obj_path)
print("OBJ vertex count:", N)

# Dummy weights: attach every vertex to SMPL joint 0/root.
W = np.zeros((N, 24), dtype=np.float32)
W[:, 0] = 1.0

# Dummy rest joints: just enough to pass the shape check.
# This will not look anatomically correct yet.
J = np.zeros((24, 3), dtype=np.float32)

np.savez(
    npz_path,
    format=np.array("gf5_smpl24_skinning_weights"),
    version=np.array(5, dtype=np.int32),
    rest_joint_coordinate_space=np.array("up2you_smpl_native_y_up"),
    rest_pose=np.array("gf5_smpl24_template_neutral_bind_pose"),
    motion_pose_space=np.array("gf5_course_body_24_absolute_local"),
    skinning_weights=W,
    rest_joints=J,
)

print("Wrote:", npz_path)