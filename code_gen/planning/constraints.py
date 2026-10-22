# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task-declared contact regions and bounded release jaw openings."""

import numpy as np

# After release, the object is an ordinary obstacle. Reserve tracking/settling clearance.
INSERTION_RELEASE_CLEARANCE_M = 0.0005


def contact_supports(scene, snapshot, object_id, *, env_id=0):
    """Explicit contact pairs valid only near their declared target region."""
    paths, names = [], []
    if object_id is None:
        return names, paths
    obj = snapshot["objects"][object_id]
    # Legacy task files declare an initial support without a target region.
    # Keep that pair local to its initial footprint; the shared guard still
    # checks robot/support collisions and bounds object/support penetration.
    legacy = scene.get("support_contacts", {}).get(object_id)
    if legacy and not any(r["object_id"] == object_id for r in scene.get("contact_surfaces", [])):
        initial = obj["initial_position"]
        if (
            np.linalg.norm(np.asarray(obj["position"][:2]) - initial[:2]) <= 0.01
            and -0.0005 <= obj["position"][2] - initial[2] <= legacy["distance_m"]
        ):
            names.append(legacy["support_id"])
    for row in scene.get("contact_surfaces", []):
        if row["object_id"] != object_id:
            continue
        target = snapshot["targets"][row["target_id"]]["position"]
        # A local segment may leave this region; contact tolerance is never
        # applied to undeclared pairs, even if they cause a planning failure.
        if np.linalg.norm(np.asarray(obj["position"][:2]) - target[:2]) > row.get("xy_radius_m", 0.01):
            continue
        # Region membership is not penetration acceptance. A compliant grasp
        # may tilt the object's world AABB below the nominal support plane.
        # Keep the declared pair active near that plane; the shared guard still
        # enforces its unchanged 0.3 mm sphere/support contact tolerance.
        if not -0.003 <= obj["lowest_z"] - target[2] <= 0.21:
            continue
        if "support_id" in row:
            names.append(row["support_id"])
        else:
            paths.extend(
                path.replace("/World/envs/env_0/", f"/World/envs/env_{env_id}/", 1) for path in row["obstacle_paths"]
            )
    return names, paths


def insertion_release_widths(grasp_width_m, maximum_width_m=0.075):
    """Bounded jaw openings; collision and physical release still need checking.

    Fine increments allow densely assembled parts to be released without
    unnecessarily moving a fingertip into a neighboring object. Cover the full
    aperture too: narrow objects can need more than 16 mm of extra opening for
    the conservative finger spheres to clear them. Return total jaw aperture,
    never a command that closes the measured grasp further.
    """
    width, maximum = float(grasp_width_m), float(maximum_width_m)
    if not np.isfinite(width) or not np.isfinite(maximum) or not 0 <= width <= 0.08 or not 0 < maximum <= 0.08:
        raise ValueError("Invalid insertion release jaw aperture")
    increments = (0.00025, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016, 0.024, 0.032, 0.048, 0.064)
    return sorted({min(maximum, width + gap) for gap in (*increments, maximum) if min(maximum, width + gap) > width})
