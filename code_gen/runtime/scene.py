# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Import compatibility for previously serialized Path Tracing scene bundles."""

from isaaclab_arena.environments.tabletop_environment import make_aloha_env as make_env
from isaaclab_arena.environments.tabletop_environment import prepare_scene, spawn_mgg_table

__all__ = ["make_env", "prepare_scene", "spawn_mgg_table"]
