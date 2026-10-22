# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Atomic JSON, content hashes and stable repository paths shared by the pipeline."""

import hashlib
import os
from pathlib import Path

from isaaclab_arena.utils.file_io import digest_file, write_json

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def resolve_path(value):
    """Resolve repository paths or an explicitly configured asset-root prefix."""
    value = str(value)
    token = "${LISIM_ASSETS_PATH}"
    if value == token or value.startswith(token + "/"):
        assets = os.environ.get("LISIM_ASSETS_PATH")
        if not assets:
            raise ValueError("Set LISIM_ASSETS_PATH to the mounted Aloha asset directory")
        value = assets + value[len(token) :]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def source_hash(source):
    return hashlib.sha256(source.encode()).hexdigest()


def framework_hashes():
    """One complete provenance list for persistent and isolated workers."""
    curobo = PROJECT_ROOT / "submodules/IsaacLab/source/isaaclab_mimic/isaaclab_mimic/motion_planners/curobo"
    paths = [
        *PACKAGE_ROOT.rglob("*.py"),
        PROJECT_ROOT / "isaaclab_arena/evaluation/codegen_worker.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/action_planner.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/endpoint_ik.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/cartesian_collision.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/cartesian_insertion.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/cartesian_motion.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/joint_actions.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/profiles.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/grasp_catalog.py",
        PROJECT_ROOT / "isaaclab_arena/server/robo_challenge/recorder.py",
        PROJECT_ROOT / "isaaclab_arena/server/robo_challenge/observations.py",
        PROJECT_ROOT / "isaaclab_arena/utils/path_tracing_replay.py",
        *curobo.glob("*.py"),
        *(PROJECT_ROOT / "isaaclab_arena/grasping/library").glob("*.py"),
        PROJECT_ROOT / "isaaclab_arena/grasping/geometry.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/skills.py",
        PROJECT_ROOT / "isaaclab_arena/utils/layout_preview.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/robot_config.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/solver.py",
        PROJECT_ROOT / "isaaclab_arena/grasping/robot_layout.py",
        PROJECT_ROOT / "isaaclab_arena/environments/tabletop_environment.py",
        PROJECT_ROOT / "isaaclab_arena/utils/file_io.py",
    ]
    return {str(path.relative_to(PROJECT_ROOT)): digest_file(path) for path in sorted(set(paths))}
