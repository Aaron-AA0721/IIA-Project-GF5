from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

src_dir = Path("libraries/avatars/Aaron/outputs")
dst_dir = Path("libraries/avatars/Collei/outputs")

src_obj = src_dir / "animation_lowres.obj"
src_npz = src_dir / "animation_lowres_skinning_weights.npz"

dst_obj = dst_dir / "animation_lowres.obj"
dst_npz = dst_dir / "animation_lowres_skinning_weights.npz"


def load_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(verts, dtype=np.float32)


V_src = load_obj_vertices(src_obj)
V_dst = load_obj_vertices(dst_obj)

with np.load(src_npz, allow_pickle=False) as src:
    W_src = np.asarray(src["skinning_weights"], dtype=np.float32)
    rest_joints = np.asarray(src["rest_joints"], dtype=np.float32)

    # copy useful metadata from Aaron
    meta = {k: src[k] for k in src.files if k not in {"skinning_weights"}}

print("source verts:", V_src.shape)
print("target verts:", V_dst.shape)
print("source weights:", W_src.shape)
print("rest_joints:", rest_joints.shape)

# crude bbox alignment for matching only
src_center = 0.5 * (V_src.min(axis=0) + V_src.max(axis=0))
dst_center = 0.5 * (V_dst.min(axis=0) + V_dst.max(axis=0))

src_size = np.max(V_src.max(axis=0) - V_src.min(axis=0))
dst_size = np.max(V_dst.max(axis=0) - V_dst.min(axis=0))
scale = src_size / max(dst_size, 1e-8)

V_dst_for_matching = (V_dst - dst_center) * scale + src_center

tree = cKDTree(V_src)

# Use k-nearest blending instead of single nearest vertex.
# This makes smoother weights.
dist, idx = tree.query(V_dst_for_matching, k=8)

inv = 1.0 / np.maximum(dist, 1e-6)
inv = inv / inv.sum(axis=1, keepdims=True)

W_dst = (W_src[idx] * inv[..., None]).sum(axis=1)
W_dst = np.maximum(W_dst, 0.0)
W_dst = W_dst / np.clip(W_dst.sum(axis=1, keepdims=True), 1e-8, None)
W_dst = W_dst.astype(np.float32)

np.savez(
    dst_npz,
    **meta,
    skinning_weights=W_dst,
)

print("wrote:", dst_npz)
print("new weights:", W_dst.shape)