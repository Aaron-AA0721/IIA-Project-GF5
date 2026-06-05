from pathlib import Path
import numpy as np

template_npz = Path("libraries/avatars/Aaron/outputs/animation_lowres_skinning_weights.npz")
avatar_dir = Path("libraries/avatars/hilichurl_test/outputs")

obj_path = avatar_dir / "animation_lowres.obj"
out_npz = avatar_dir / "animation_lowres_skinning_weights.npz"


def load_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(verts, dtype=np.float32)


V = load_obj_vertices(obj_path)
N = V.shape[0]
print("PMX-exported OBJ vertices:", N)

with np.load(template_npz, allow_pickle=False) as t:
    rest_joints = np.asarray(t["rest_joints"], dtype=np.float32)
    joint_names = np.asarray(t["joint_names"])
    parents = np.asarray(t["parents"])

# Very rough one-hot nearest-joint weights.
# This is just to make the avatar load and deform roughly.
dist = np.linalg.norm(V[:, None, :] - rest_joints[None, :, :], axis=2)
nearest = np.argmin(dist, axis=1)

W = np.zeros((N, 24), dtype=np.float32)
W[np.arange(N), nearest] = 1.0

np.savez(
    out_npz,
    format=np.array("gf5_smpl24_skinning_weights"),
    version=np.array(5, dtype=np.int64),
    mesh_name=np.array("animation_lowres.obj"),
    profile_name=np.array("smpl_24"),
    joint_names=joint_names,
    parents=parents,
    rest_joints=rest_joints,
    rest_joint_coordinate_space=np.array("up2you_smpl_native_y_up"),
    rest_pose=np.array("gf5_smpl24_template_neutral_bind_pose"),
    motion_pose_space=np.array("gf5_course_body_24_absolute_local"),
    bind_pose_correction=np.array("exported_mesh_and_rest_joints_are_already_canonical"),
    canonical_bind_source=np.array("smplx_template_v_template_smpl24_joints"),
    skinning_weights=W,
    matched=np.zeros((N,), dtype=bool),
)

print("Wrote:", out_npz)
print("skinning_weights:", W.shape)
print("rest_joints:", rest_joints.shape)