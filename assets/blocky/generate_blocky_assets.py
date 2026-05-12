from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


Vec3 = npt.NDArray[np.float64]


def as_vec3(values: tuple[float, float, float] | list[float]) -> Vec3:
    return np.asarray(values, dtype=np.float64)


def normalize(vector: Vec3) -> Vec3:
    length = np.linalg.norm(vector)
    if length == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / length


def orthonormal_basis(axis: Vec3) -> tuple[Vec3, Vec3]:
    axis = normalize(axis)
    helper = as_vec3([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = as_vec3([0.0, 1.0, 0.0])
    side = normalize(np.cross(axis, helper))
    up = normalize(np.cross(side, axis))
    return side, up


def make_box(center: Vec3, size: Vec3) -> tuple[list[list[float]], list[list[int]]]:
    sx, sy, sz = size / 2.0
    offsets = [
        as_vec3([-sx, -sy, -sz]),
        as_vec3([sx, -sy, -sz]),
        as_vec3([sx, sy, -sz]),
        as_vec3([-sx, sy, -sz]),
        as_vec3([-sx, -sy, sz]),
        as_vec3([sx, -sy, sz]),
        as_vec3([sx, sy, sz]),
        as_vec3([-sx, sy, sz]),
    ]
    vertices = [[round(float(v), 6) for v in center + offset] for offset in offsets]
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    return vertices, faces


def make_bone_block(
    end: Vec3,
    width: float,
    depth: float,
) -> tuple[list[list[float]], list[list[int]]]:
    axis = end
    side, up = orthonormal_basis(axis)
    ring_start = [
        side * -width + up * -depth,
        side * width + up * -depth,
        side * width + up * depth,
        side * -width + up * depth,
    ]
    ring_end = [offset + axis for offset in ring_start]

    vertices = [
        [round(float(v), 6) for v in vertex]
        for vertex in [*ring_start, *ring_end]
    ]
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 5, 1],
        [0, 4, 5],
        [1, 6, 2],
        [1, 5, 6],
        [2, 7, 3],
        [2, 6, 7],
        [3, 4, 0],
        [3, 7, 4],
    ]
    return vertices, faces


def compute_rest_positions(joints: list[dict[str, Any]]) -> list[Vec3]:
    positions: list[Vec3] = [np.zeros(3, dtype=np.float64) for _ in joints]
    for idx, joint in enumerate(joints):
        parent = int(joint["parent"])
        local = as_vec3(joint["translation"])
        if parent < 0:
            positions[idx] = local
        else:
            positions[idx] = positions[parent] + local
    return positions


def add_bone_part(
    parts: list[dict[str, Any]],
    part_name: str,
    joint_name: str,
    end: list[float],
    width: float,
    depth: float,
    color: list[int],
) -> None:
    vertices, faces = make_bone_block(as_vec3(end), width, depth)
    parts.append(
        {
            "name": part_name,
            "joint": joint_name,
            "vertices": vertices,
            "faces": faces,
            "color": color,
            "flat_shading": True,
            "side": "double",
        }
    )


def add_joint_box(
    parts: list[dict[str, Any]],
    part_name: str,
    joint_name: str,
    center: list[float],
    size: list[float],
    color: list[int],
) -> None:
    vertices, faces = make_box(as_vec3(center), as_vec3(size))
    parts.append(
        {
            "name": part_name,
            "joint": joint_name,
            "vertices": vertices,
            "faces": faces,
            "color": color,
            "flat_shading": True,
            "side": "double",
        }
    )


def base_joint_spec(
    root_height: float,
    shoulder_span: float,
    upper_arm: float,
    forearm: float,
    hip_offset: float,
    thigh: float,
    shin: float,
    foot: float,
) -> list[dict[str, Any]]:
    return [
        {"name": "root", "parent": -1, "translation": [0.0, 0.0, root_height]},
        {"name": "spine", "parent": 0, "translation": [0.0, 0.0, 0.18]},
        {"name": "chest", "parent": 1, "translation": [0.0, 0.0, 0.18]},
        {"name": "neck", "parent": 2, "translation": [0.0, 0.0, 0.12]},
        {"name": "head", "parent": 3, "translation": [0.0, 0.0, 0.10]},
        {"name": "left_shoulder", "parent": 2, "translation": [shoulder_span, 0.0, 0.05]},
        {"name": "left_elbow", "parent": 5, "translation": [upper_arm, 0.0, 0.0]},
        {"name": "left_wrist", "parent": 6, "translation": [forearm, 0.0, 0.0]},
        {"name": "right_shoulder", "parent": 2, "translation": [-shoulder_span, 0.0, 0.05]},
        {"name": "right_elbow", "parent": 8, "translation": [-upper_arm, 0.0, 0.0]},
        {"name": "right_wrist", "parent": 9, "translation": [-forearm, 0.0, 0.0]},
        {"name": "left_hip", "parent": 0, "translation": [hip_offset, 0.0, -0.06]},
        {"name": "left_knee", "parent": 11, "translation": [0.0, 0.0, -thigh]},
        {"name": "left_ankle", "parent": 12, "translation": [0.0, 0.0, -shin]},
        {"name": "left_toe", "parent": 13, "translation": [0.0, foot, 0.0]},
        {"name": "right_hip", "parent": 0, "translation": [-hip_offset, 0.0, -0.06]},
        {"name": "right_knee", "parent": 15, "translation": [0.0, 0.0, -thigh]},
        {"name": "right_ankle", "parent": 16, "translation": [0.0, 0.0, -shin]},
        {"name": "right_toe", "parent": 17, "translation": [0.0, foot, 0.0]},
    ]


def build_rigid_asset(
    name: str,
    description: str,
    color_body: list[int],
    color_joints: list[int],
    proportions: dict[str, float],
) -> dict[str, Any]:
    root_height = 0.06 + proportions["thigh"] + proportions["shin"] + proportions["limb_depth"] * 0.8
    joints = base_joint_spec(
        root_height=root_height,
        shoulder_span=proportions["shoulder_span"],
        upper_arm=proportions["upper_arm"],
        forearm=proportions["forearm"],
        hip_offset=proportions["hip_offset"],
        thigh=proportions["thigh"],
        shin=proportions["shin"],
        foot=proportions["foot"],
    )
    rest_positions = compute_rest_positions(joints)
    parts: list[dict[str, Any]] = []

    torso_width = proportions["torso_width"]
    torso_depth = proportions["torso_depth"]
    limb_width = proportions["limb_width"]
    limb_depth = proportions["limb_depth"]
    joint_size = proportions["joint_size"]

    add_bone_part(parts, "pelvis", "root", [0.0, 0.0, 0.12], torso_width, torso_depth, color_body)
    add_bone_part(parts, "spine", "spine", [0.0, 0.0, 0.18], torso_width * 0.9, torso_depth, color_body)
    add_bone_part(parts, "chest", "chest", [0.0, 0.0, 0.12], torso_width * 1.1, torso_depth * 1.1, color_body)
    add_bone_part(parts, "neck", "neck", [0.0, 0.0, 0.08], torso_width * 0.35, torso_depth * 0.35, color_body)

    add_bone_part(parts, "left_upper_arm", "left_shoulder", joints[6]["translation"], limb_width, limb_depth, color_body)
    add_bone_part(parts, "left_forearm", "left_elbow", joints[7]["translation"], limb_width * 0.9, limb_depth * 0.9, color_body)
    add_bone_part(parts, "right_upper_arm", "right_shoulder", joints[9]["translation"], limb_width, limb_depth, color_body)
    add_bone_part(parts, "right_forearm", "right_elbow", joints[10]["translation"], limb_width * 0.9, limb_depth * 0.9, color_body)

    add_bone_part(parts, "left_thigh", "left_hip", joints[12]["translation"], limb_width * 1.15, limb_depth * 1.15, color_body)
    add_bone_part(parts, "left_shin", "left_knee", joints[13]["translation"], limb_width, limb_depth, color_body)
    add_bone_part(parts, "left_foot", "left_ankle", joints[14]["translation"], limb_width * 0.95, limb_depth * 0.8, color_body)
    add_bone_part(parts, "right_thigh", "right_hip", joints[16]["translation"], limb_width * 1.15, limb_depth * 1.15, color_body)
    add_bone_part(parts, "right_shin", "right_knee", joints[17]["translation"], limb_width, limb_depth, color_body)
    add_bone_part(parts, "right_foot", "right_ankle", joints[18]["translation"], limb_width * 0.95, limb_depth * 0.8, color_body)

    add_joint_box(parts, "root_joint", "root", [0.0, 0.0, 0.03], [joint_size * 1.6, joint_size * 1.2, joint_size * 1.4], color_joints)
    add_joint_box(parts, "chest_joint", "chest", [0.0, 0.0, 0.02], [joint_size * 1.2, joint_size * 1.2, joint_size * 1.2], color_joints)
    add_joint_box(parts, "head_box", "head", [0.0, 0.0, 0.08], [0.16, 0.14, 0.16], color_joints)
    add_joint_box(parts, "left_shoulder_joint", "left_shoulder", [0.0, 0.0, 0.0], [joint_size, joint_size, joint_size], color_joints)
    add_joint_box(parts, "left_elbow_joint", "left_elbow", [0.0, 0.0, 0.0], [joint_size * 0.95, joint_size * 0.95, joint_size * 0.95], color_joints)
    add_joint_box(parts, "left_wrist_joint", "left_wrist", [0.0, 0.0, 0.0], [joint_size * 0.8, joint_size * 0.8, joint_size * 0.8], color_joints)
    add_joint_box(parts, "right_shoulder_joint", "right_shoulder", [0.0, 0.0, 0.0], [joint_size, joint_size, joint_size], color_joints)
    add_joint_box(parts, "right_elbow_joint", "right_elbow", [0.0, 0.0, 0.0], [joint_size * 0.95, joint_size * 0.95, joint_size * 0.95], color_joints)
    add_joint_box(parts, "right_wrist_joint", "right_wrist", [0.0, 0.0, 0.0], [joint_size * 0.8, joint_size * 0.8, joint_size * 0.8], color_joints)
    add_joint_box(parts, "left_hip_joint", "left_hip", [0.0, 0.0, 0.0], [joint_size, joint_size, joint_size], color_joints)
    add_joint_box(parts, "left_knee_joint", "left_knee", [0.0, 0.0, 0.0], [joint_size, joint_size, joint_size], color_joints)
    add_joint_box(parts, "left_ankle_joint", "left_ankle", [0.0, 0.0, 0.0], [joint_size * 0.9, joint_size * 0.9, joint_size * 0.9], color_joints)
    add_joint_box(parts, "right_hip_joint", "right_hip", [0.0, 0.0, 0.0], [joint_size, joint_size, joint_size], color_joints)
    add_joint_box(parts, "right_knee_joint", "right_knee", [0.0, 0.0, 0.0], [joint_size, joint_size, joint_size], color_joints)
    add_joint_box(parts, "right_ankle_joint", "right_ankle", [0.0, 0.0, 0.0], [joint_size * 0.9, joint_size * 0.9, joint_size * 0.9], color_joints)

    return {
        "asset_format": "gf5_rigid_character",
        "asset_version": 1,
        "name": name,
        "description": description,
        "units": "meters",
        "display": {
            "up_axis": [0.0, 0.0, 1.0],
            "forward_axis": [0.0, 1.0, 0.0],
        },
        "skeleton": {
            "joints": [
                {
                    "name": joint["name"],
                    "parent": joint["parent"],
                    "translation": joint["translation"],
                    "rest_position": [round(float(v), 6) for v in rest_positions[idx]],
                }
                for idx, joint in enumerate(joints)
            ],
            "bone_edges": [
                [int(joint["parent"]), idx]
                for idx, joint in enumerate(joints)
                if int(joint["parent"]) >= 0
            ],
        },
        "rigid_parts": parts,
    }


def write_asset(path: Path, asset: dict[str, Any]) -> None:
    path.write_text(json.dumps(asset, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    marigold = build_rigid_asset(
        name="Marigold",
        description="Part 1 rigid character for forward-kinematics exercises.",
        color_body=[230, 170, 90],
        color_joints=[80, 80, 80],
        proportions={
            "shoulder_span": 0.17,
            "upper_arm": 0.20,
            "forearm": 0.18,
            "hip_offset": 0.09,
            "thigh": 0.27,
            "shin": 0.25,
            "foot": 0.11,
            "torso_width": 0.09,
            "torso_depth": 0.06,
            "limb_width": 0.035,
            "limb_depth": 0.035,
            "joint_size": 0.06,
        },
    )
    azure = build_rigid_asset(
        name="Azure",
        description="Shorter, chunkier variant for testing asset loading in the viewer.",
        color_body=[110, 180, 210],
        color_joints=[45, 65, 85],
        proportions={
            "shoulder_span": 0.16,
            "upper_arm": 0.17,
            "forearm": 0.16,
            "hip_offset": 0.10,
            "thigh": 0.23,
            "shin": 0.21,
            "foot": 0.10,
            "torso_width": 0.10,
            "torso_depth": 0.07,
            "limb_width": 0.042,
            "limb_depth": 0.042,
            "joint_size": 0.065,
        },
    )

    marigold_path = out_dir / "marigold.asset.json"
    azure_path = out_dir / "azure.asset.json"
    write_asset(marigold_path, marigold)
    write_asset(azure_path, azure)

    print(f"Wrote {marigold_path.name} and {azure_path.name}")
    print(
        "Marigold asset:",
        f"{len(marigold['skeleton']['joints'])} joints,",
        f"{len(marigold['rigid_parts'])} rigid parts",
    )


if __name__ == "__main__":
    main()
