"""Generate simple SMPL-compatible meshes and save them as .pkl model files.

The output .pkl files use the same key layout as the official SMPL basicmodel
files so they can be loaded by ``smplx.SMPL`` (and therefore by the project's
``load_smpl_model_data``). Generated shapes are static: every vertex is
rigidly bound to the root joint, so posing the model only translates/rotates
the mesh as a whole.

Coordinate convention follows SMPL's native frame (+Y up). The viewer layer
re-orients meshes to +Z up via ``SMPL_TO_VIEWER_ROTATION``.

CLI examples::

    python -m model.object_generator cube --length 1 --width 1 --height 1 -o assets/smpl/models/cube.pkl
    python -m model.object_generator sphere --radius 0.5 -o assets/smpl/models/sphere.pkl
    python -m model.object_generator capsule --radius 0.2 --length 0.8 -o assets/smpl/models/capsule.pkl
    python -m model.object_generator obj path/to/mesh.obj -o assets/smpl/models/from_obj.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix


SMPL_NUM_JOINTS = 24
SMPL_NUM_POSE_BLENDS = (SMPL_NUM_JOINTS - 1) * 9  # 207
SMPL_NUM_SHAPE_BLENDS = 10
# ``smplx.SMPL`` looks up hard-coded landmark vertex indices (nose, eyes,
# fingers, toes, ...) whose max value is 6787. Pad the saved mesh to the
# canonical SMPL vertex count so those indices stay in bounds; padding
# vertices live at the origin and are not referenced by any face.
SMPL_VERTEX_COUNT = 6890

SMPL_KINTREE_TABLE = np.asarray(
    [
        [4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    ],
    dtype=np.uint32,
)

DEFAULT_MODEL_DIR = Path("assets/smpl/models")


class ObjectGenerator:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_DIR) -> None:
        self.model_path = Path(model_path)

    def generate_cube(
        self,
        length: float = 1.0,
        width: float = 1.0,
        height: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        hx, hy, hz = length / 2.0, width / 2.0, height / 2.0
        vertices = np.asarray(
            [
                [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
            ],
            dtype=np.float32,
        )
        faces = np.asarray(
            [
                [0, 2, 1], [0, 3, 2],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [2, 3, 7], [2, 7, 6],
                [1, 2, 6], [1, 6, 5],
                [3, 0, 4], [3, 4, 7],
            ],
            dtype=np.uint32,
        )
        return vertices, faces

    def generate_sphere(
        self,
        radius: float = 0.5,
        lat_segments: int = 16,
        lon_segments: int = 24,
    ) -> tuple[np.ndarray, np.ndarray]:
        lat_segments = max(2, int(lat_segments))
        lon_segments = max(3, int(lon_segments))
        vertices: list[list[float]] = []
        for i in range(lat_segments + 1):
            theta = np.pi * i / lat_segments
            sin_t, cos_t = float(np.sin(theta)), float(np.cos(theta))
            for j in range(lon_segments):
                phi = 2.0 * np.pi * j / lon_segments
                vertices.append([
                    radius * sin_t * float(np.cos(phi)),
                    radius * cos_t,
                    radius * sin_t * float(np.sin(phi)),
                ])
        faces: list[list[int]] = []
        for i in range(lat_segments):
            for j in range(lon_segments):
                a = i * lon_segments + j
                b = i * lon_segments + (j + 1) % lon_segments
                c = (i + 1) * lon_segments + (j + 1) % lon_segments
                d = (i + 1) * lon_segments + j
                faces.append([a, b, c])
                faces.append([a, c, d])
        return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)

    def generate_capsule(
        self,
        radius: float = 0.3,
        length: float = 1.0,
        lat_segments: int = 8,
        lon_segments: int = 24,
    ) -> tuple[np.ndarray, np.ndarray]:
        lat_segments = max(2, int(lat_segments))
        lon_segments = max(3, int(lon_segments))
        half_length = max(0.0, float(length)) / 2.0
        vertices: list[list[float]] = []
        for i in range(lat_segments + 1):
            theta = (np.pi / 2.0) * i / lat_segments
            sin_t, cos_t = float(np.sin(theta)), float(np.cos(theta))
            for j in range(lon_segments):
                phi = 2.0 * np.pi * j / lon_segments
                vertices.append([
                    radius * sin_t * float(np.cos(phi)),
                    radius * cos_t + half_length,
                    radius * sin_t * float(np.sin(phi)),
                ])
        for i in range(1, lat_segments + 1):
            theta = (np.pi / 2.0) + (np.pi / 2.0) * i / lat_segments
            sin_t, cos_t = float(np.sin(theta)), float(np.cos(theta))
            for j in range(lon_segments):
                phi = 2.0 * np.pi * j / lon_segments
                vertices.append([
                    radius * sin_t * float(np.cos(phi)),
                    radius * cos_t - half_length,
                    radius * sin_t * float(np.sin(phi)),
                ])
        total_rings = 2 * lat_segments + 1
        faces: list[list[int]] = []
        for i in range(total_rings - 1):
            for j in range(lon_segments):
                a = i * lon_segments + j
                b = i * lon_segments + (j + 1) % lon_segments
                c = (i + 1) * lon_segments + (j + 1) % lon_segments
                d = (i + 1) * lon_segments + j
                faces.append([a, b, c])
                faces.append([a, c, d])
        return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)

    def from_obj(self, obj_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        path = Path(obj_path)
        if not path.exists():
            raise FileNotFoundError(f"OBJ file not found: {path}")
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split()
                tag = tokens[0]
                if tag == "v" and len(tokens) >= 4:
                    vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
                elif tag == "f" and len(tokens) >= 4:
                    indices = [int(token.split("/")[0]) - 1 for token in tokens[1:]]
                    for i in range(1, len(indices) - 1):
                        faces.append([indices[0], indices[i], indices[i + 1]])
        if not vertices or not faces:
            raise ValueError(f"OBJ file {path} did not contain any vertices/faces")
        return (
            np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.uint32),
        )

    def save_as_pkl(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_path: str | Path,
        *,
        joint_position: np.ndarray | None = None,
    ) -> Path:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.uint32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must have shape (N, 3); got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (F, 3); got {faces.shape}")

        real_vertex_count = vertices.shape[0]
        if joint_position is None:
            joint_position = vertices.mean(axis=0)
        joint_position = np.asarray(joint_position, dtype=np.float64).reshape(3)

        padded_count = max(real_vertex_count, SMPL_VERTEX_COUNT)
        v_template = np.zeros((padded_count, 3), dtype=np.float64)
        v_template[:real_vertex_count] = vertices

        weights = np.zeros((padded_count, SMPL_NUM_JOINTS), dtype=np.float64)
        weights[:, 0] = 1.0

        regressor = np.zeros((SMPL_NUM_JOINTS, padded_count), dtype=np.float64)
        regressor[0, :real_vertex_count] = 1.0 / float(real_vertex_count)
        j_regressor = csc_matrix(regressor)

        shapedirs = np.zeros((padded_count, 3, SMPL_NUM_SHAPE_BLENDS), dtype=np.float64)
        posedirs = np.zeros((padded_count, 3, SMPL_NUM_POSE_BLENDS), dtype=np.float64)
        joints = np.tile(joint_position, (SMPL_NUM_JOINTS, 1))

        payload = {
            "v_template": v_template,
            "f": faces,
            "weights": weights,
            "weights_prior": weights.copy(),
            "kintree_table": SMPL_KINTREE_TABLE.copy(),
            "J_regressor": j_regressor,
            "J_regressor_prior": j_regressor.copy(),
            "shapedirs": shapedirs,
            "posedirs": posedirs,
            "J": joints,
            "bs_style": "lbs",
            "bs_type": "lrotmin",
            "vert_sym_idxs": np.arange(padded_count, dtype=np.int64),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output_path


def _resolve_output_path(generator: ObjectGenerator, output: str | None, default_stem: str) -> Path:
    if output is None:
        return generator.model_path / f"{default_stem}.pkl"
    output_path = Path(output)
    if output_path.suffix == "":
        output_path = output_path / f"{default_stem}.pkl"
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Default output directory when --output is not given.",
    )
    subparsers = parser.add_subparsers(dest="shape", required=True)

    cube_parser = subparsers.add_parser("cube", help="Generate an axis-aligned box.")
    cube_parser.add_argument("--length", type=float, default=1.0)
    cube_parser.add_argument("--width", type=float, default=1.0)
    cube_parser.add_argument("--height", type=float, default=1.0)
    cube_parser.add_argument("-o", "--output", default=None)

    sphere_parser = subparsers.add_parser("sphere", help="Generate a UV sphere.")
    sphere_parser.add_argument("--radius", type=float, default=0.5)
    sphere_parser.add_argument("--lat-segments", type=int, default=16)
    sphere_parser.add_argument("--lon-segments", type=int, default=24)
    sphere_parser.add_argument("-o", "--output", default=None)

    capsule_parser = subparsers.add_parser("capsule", help="Generate a Y-axis capsule.")
    capsule_parser.add_argument("--radius", type=float, default=0.3)
    capsule_parser.add_argument("--length", type=float, default=1.0)
    capsule_parser.add_argument("--lat-segments", type=int, default=8)
    capsule_parser.add_argument("--lon-segments", type=int, default=24)
    capsule_parser.add_argument("-o", "--output", default=None)

    obj_parser = subparsers.add_parser("obj", help="Convert a Wavefront .obj file to a .pkl.")
    obj_parser.add_argument("path")
    obj_parser.add_argument("-o", "--output", default=None)

    args = parser.parse_args(argv)
    generator = ObjectGenerator(args.model_dir)

    if args.shape == "cube":
        vertices, faces = generator.generate_cube(args.length, args.width, args.height)
        output_path = _resolve_output_path(generator, args.output, "cube")
    elif args.shape == "sphere":
        vertices, faces = generator.generate_sphere(args.radius, args.lat_segments, args.lon_segments)
        output_path = _resolve_output_path(generator, args.output, "sphere")
    elif args.shape == "capsule":
        vertices, faces = generator.generate_capsule(
            args.radius, args.length, args.lat_segments, args.lon_segments
        )
        output_path = _resolve_output_path(generator, args.output, "capsule")
    elif args.shape == "obj":
        vertices, faces = generator.from_obj(args.path)
        output_path = _resolve_output_path(generator, args.output, Path(args.path).stem)
    else:
        parser.error(f"Unknown shape: {args.shape}")
        return

    saved_path = generator.save_as_pkl(vertices, faces, output_path)
    print(f"Wrote {len(vertices)} vertices, {len(faces)} faces to {saved_path}")


if __name__ == "__main__":
    main()
