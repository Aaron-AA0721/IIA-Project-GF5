from __future__ import annotations

import importlib
from types import ModuleType


VALID_IMPLEMENTATION_SOURCES = ("student",)

_active_fk_source = "student"
_active_skinning_source = "student"


def _load_module(module_path: str) -> ModuleType:
    return importlib.import_module(module_path)


def set_active_implementation_sources(
    *,
    fk_source: str,
    skinning_source: str,
) -> None:
    if fk_source not in VALID_IMPLEMENTATION_SOURCES:
        raise ValueError(f"Unknown FK implementation source: {fk_source}")
    if skinning_source not in VALID_IMPLEMENTATION_SOURCES:
        raise ValueError(f"Unknown skinning implementation source: {skinning_source}")

    global _active_fk_source, _active_skinning_source
    _active_fk_source = fk_source
    _active_skinning_source = skinning_source


def get_part1_fk_module() -> ModuleType:
    return _load_module("student_submission.part1_fk")


def get_part2_skinning_module() -> ModuleType:
    return _load_module("student_submission.part2_skinning")
