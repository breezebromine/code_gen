"""Layout-scoped retry evidence for runtime-selected grasps.

A selected pose is not a stable grasp. Preferences require a completed physical
pick/transport/insert/release sequence, including any declared retreat; failures only exclude the implicated
grasp, never the last grasp merely because the overall task failed.
"""

import ast
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

PICK_ACTIONS = {"pick", "grasp_actor"}
INFRASTRUCTURE_TERMS = (
    "Attachment collision requires",
    "Contact sphere budget insufficient",
    "CUDA out of memory",
    "CUDA error:",
    "illegal memory access",
    "Missing or malformed simulation report",
)
INFRASTRUCTURE_TYPES = {
    "ImportError",
    "ModuleNotFoundError",
    "FileNotFoundError",
    "KeyError",
    "AttributeError",
    "MemoryError",
    "OutOfMemoryError",
    "TimeoutError",
}


def is_planning_failure(error):
    """Recognize the motion adapter's explicit solver failures, not arbitrary exceptions."""
    return (
        isinstance(error, dict)
        and error.get("type") == "RuntimeError"
        and re.fullmatch(
            r"cuRobo planning failed in (approach|descend|lift|move|home): "
            r"MotionGenStatus\.(IK_FAIL|TRAJOPT_FAIL|FINETUNE_TRAJOPT_FAIL|GRAPH_FAIL|DT_EXCEPTION); no IK fallback",
            error.get("message", ""),
        )
        is not None
    )


def prepare_grasp_retry_state(worker, task, snapshot=None):
    """Expire layout-local evidence when scene, library, task, or robot changes."""
    task_data = dict(task.to_dict())
    scene_file = task_data.pop("scene_file", None)
    scene = json.loads(Path(scene_file).read_text()) if scene_file else None
    libraries = {}
    if scene:
        for name, obj in scene.get("objects", {}).items():
            path = obj.get("grasp_library")
            if isinstance(path, str):
                resolved = Path(path)
                if not resolved.is_absolute():
                    resolved = Path(scene_file).parent / resolved
                if resolved.is_file():
                    libraries[name] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    state = snapshot or {}
    fingerprint = {
        "task": task_data,
        "scene": scene,
        "libraries": libraries,
        "seed": getattr(worker, "seed", None),
        "robot": state.get("robot", getattr(worker, "robot", None)),
        "arm": state.get("active_arm", getattr(worker, "arm", None)),
    }
    # Built-in scenes have no scene JSON; retain layout and complete grasp frames
    # but ignore contact forces/velocities and other mutable observations.
    if not scene:
        fingerprint["objects"] = {
            name: {key: obj.get(key) for key in ("position", "quaternion", "dimensions_m", "grasp_frames")}
            for name, obj in state.get("objects", {}).items()
        }
        fingerprint["targets"] = state.get("targets", {})
    scope = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    previous = getattr(worker, "grasp_retry_scope", None)
    if previous is not None and previous != scope:
        worker.excluded_grasp_candidates = {}
        worker.preferred_grasp_candidates = {}
    worker.grasp_retry_scope = scope
    return scope


def _candidate(row):
    value = row.get("selected_candidate_id")
    return value if value is not None else row.get("selected_yaw_deg")


def _succeeded(event):
    result = event.get("result", {})
    return result["success"] is True if "success" in result else result.get("status") == "success"


def _retreat_height(selection):
    """Use the chosen path's recorded declaration, not a mutable scene file."""
    declared = selection.get("declared_retreat_height_m")
    if isinstance(declared, (int, float)) and not isinstance(declared, bool):
        if math.isfinite(declared) and 0.01 <= declared <= 0.15:
            return float(declared)
    heights = {
        segment.get("retreat_height_m")
        for candidate in selection.get("candidates", [])
        if candidate.get("candidate_id") == _candidate(selection) and candidate.get("preflight_success") is True
        for segment in candidate.get("segments", [])
        if segment.get("phase") == "release_retreat"
    }
    if len(heights) != 1:
        return None
    value = heights.pop()
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
        else None
    )


def _matches_retreat(event, height):
    args = event.get("arguments", {})
    displacement = args.get("displacement")
    return (
        event.get("action") == "move_by_displacement"
        and args.get("frame") == "world"
        and isinstance(displacement, (list, tuple))
        and len(displacement) == 3
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in displacement)
        and all(abs(v - expected) <= 1e-9 for v, expected in zip(displacement, (0, 0, height)))
    )


def _grasp_progress(simulation):
    """Match selections to physical events; a declared release retreat is part of completion."""
    selections = simulation.get("planning", {}).get("grasp_selections", [])
    selection_index = 0
    active, completed = None, []
    for event in simulation.get("events", []):
        action, args = event.get("action"), event.get("arguments", {})
        if action in PICK_ACTIONS:
            active = None
            # Explicit grasp requests do not consume another object's selection.
            object_id = args.get("object_id")
            if args.get("grasp_id") is not None:
                continue
            row = selections[selection_index] if selection_index < len(selections) else None
            if row and row.get("object_id") == object_id:
                selection_index += 1
            else:
                continue
            if _candidate(row) is not None and _succeeded(event):
                active = {
                    "object_id": object_id,
                    "candidate_id": _candidate(row),
                    "selection": row,
                    "transported": False,
                    "inserted": False,
                }
        elif active and active.get("released"):
            # Only the immediately following declared motion completes this
            # sequence. Later navigation, waits or another object cannot stand in.
            if _matches_retreat(event, active["retreat_height_m"]):
                if _succeeded(event):
                    completed.append({"object_id": active["object_id"], "candidate_id": active["candidate_id"]})
                    active = None
                # A returned failure remains available to the strict attribution
                # below, just like an exception with no event appended yet.
            else:
                active = None
        elif active and not _succeeded(event):
            active = None
        elif active and action == "move_object_to_pose" and args.get("object_id") == active["object_id"]:
            active["transported"] = True
        elif active and action == "insert_object" and args.get("object_id") == active["object_id"]:
            active["inserted"] = active["transported"]
            active["insert_event"] = event
        elif active and action == "open_gripper":
            height = _retreat_height(active["selection"])
            if active["inserted"] and height is not None:
                active.update(released=True, release_event=event, retreat_height_m=height)
            else:
                if active["inserted"]:
                    completed.append({"object_id": active["object_id"], "candidate_id": active["candidate_id"]})
                active = None
    return completed, active if active and active.get("released") else None


def completed_grasps(simulation):
    """Return sequences that completed their declared physical placement actions."""
    return _grasp_progress(simulation)[0]


def _released_object_collision(simulation, context, selected, reason):
    """Recognize only the just-released object's exact declared retreat collision."""
    _, released = _grasp_progress(simulation)
    if not released or released["selection"] is not selected:
        return False
    object_id = released["object_id"]
    opening, insertion = released["release_event"], released["insert_event"]
    error = simulation.get("error")
    if error and (not isinstance(error, dict) or error.get("type") != "RuntimeError"):
        return False
    if (
        not _matches_retreat(context, released["retreat_height_m"])
        or context.get("held_object_id") is not None
        or context.get("phase") not in (None, "move")
        or opening.get("held_object_id") != object_id
        or "holding" not in opening.get("result", {})
        or opening["result"]["holding"] is not None
        or insertion.get("held_object_id") != object_id
        or insertion.get("result", {}).get("holding") != object_id
        or insertion.get("result", {}).get("object_id") != object_id
        or "holding" not in simulation.get("final_snapshot", {})
        or simulation["final_snapshot"]["holding"] is not None
    ):
        return False
    # The native guard currently serializes this dictionary inside RuntimeError.
    # literal_eval accepts data only; never execute an exception's text.
    if len(reason) > 65536 or reason.partition(":")[0].strip() != "CARTESIAN_COLLISION":
        return False
    try:
        collision = ast.literal_eval(reason.partition(":")[2].strip())
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return False

    def isolated_world_collision(row):
        if not isinstance(row, dict) or row.get("passed") is not False:
            return False
        fields = ("world_collision_m", "robot_clearance_collision_m", "support_collision_m", "self_collision_cost")
        values = [row.get(key) for key in fields]
        return (
            all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
            and values[0] > 1e-6
            and all(v == 0 for v in values[1:])
        )

    if not isolated_world_collision(collision):
        return False
    obstacles = collision.get("obstacles")
    return bool(
        isinstance(obstacles, list)
        and obstacles
        and all(
            isolated_world_collision(row)
            and isinstance(row.get("obstacle"), str)
            and PurePosixPath(row["obstacle"]).parts[:5] == ("/", "World", "envs", "env_0", object_id)
            for row in obstacles
        )
    )


def failed_grasp(simulation):
    """Return only a selected grasp supported by actual failure provenance."""
    if simulation.get("physical_success"):
        return None, "physical_success"
    error = simulation.get("error") or {}
    error_text = error.get("message", "") if isinstance(error, dict) else str(error)
    context = simulation.get("failed_action") or simulation.get("action_context") or {}
    action, args = context.get("action"), context.get("arguments", {})
    if (
        (isinstance(error, dict) and error.get("type") in INFRASTRUCTURE_TYPES)
        or any(term in error_text for term in INFRASTRUCTURE_TERMS)
        or simulation.get("timed_out")
    ):
        return None, "infrastructure"
    if isinstance(error, dict) and error.get("type") == "ValueError":
        return None, "action_precondition" if action and action not in PICK_ACTIONS else "infrastructure"
    reason = context.get("result", {}).get("reason", "") or error_text
    reason_code = reason.split(":", 1)[0].strip()
    rows = simulation.get("planning", {}).get("grasp_selections", [])
    selected = rows[-1] if rows else {}
    if action == "move_by_displacement" and reason_code == "CARTESIAN_COLLISION":
        if _released_object_collision(simulation, context, selected, reason):
            return selected, "placement_release_retreat_collision"
    if action == "open_gripper" and reason_code == "INSERTION_RELEASE_NO_CLEARANCE":
        # The measured inserted grasp cannot open around its destination
        # neighbors. Attribute only through the object held at action start:
        # post-release state or the last selection alone is not sufficient.
        held_object = context.get("held_object_id")
        if held_object is not None and selected.get("object_id") == held_object and _candidate(selected) is not None:
            return selected, "placement_release_clearance"
        return None, "no_matching_selection"
    if _candidate(selected) is None or selected.get("object_id") != args.get("object_id"):
        return None, "no_matching_selection"
    if (
        action == "move_object_to_pose"
        and is_planning_failure(error)
        and context.get("held_object_id") == args.get("object_id")
        and context.get("held_object_id") is not None
    ):
        if error["message"].endswith("MotionGenStatus.IK_FAIL; no IK fallback"):
            return selected, "placement_ik"
        return selected, "placement_planning"
    if (
        action == "insert_object"
        and reason_code == "CARTESIAN_COLLISION"
        and context.get("held_object_id") is not None
        and context["held_object_id"] == args.get("object_id")
    ):
        # Even after nominal preflight, the measured object-in-gripper transform
        # can make this grasp incompatible with the destination neighbors. Only
        # the explicitly held/requested/selected object earns this exclusion;
        # layout-scoped retry state prevents carrying it into unrelated scenes.
        return selected, "placement_path_collision"
    live_failures = {
        "LIVE_INSERTION_PREFLIGHT_IK_FAIL": (
            {"CARTESIAN_PREVIEW_IK_FAIL", "CARTESIAN_PREVIEW_JOINT_JUMP"},
            "placement_measured_grasp_ik",
        ),
        "LIVE_INSERTION_PREFLIGHT_JOINT_MARGIN": (
            {"CARTESIAN_PREVIEW_JOINT_MARGIN"},
            "placement_measured_grasp_joint_margin",
        ),
    }
    if action == "insert_object" and reason_code in live_failures:
        result = context.get("result", {})
        segment = result.get("failed_preflight_segment", {})
        expected_phases = {"align": "align", "descend": "descend", "retreat": "release_retreat"}
        statuses, failure_kind = live_failures[reason_code]
        object_id = args.get("object_id")
        if (
            object_id is not None
            and context.get("held_object_id") == object_id
            and result.get("object_id") == object_id
            and result.get("holding") == object_id
            and result.get("source") == "measured_grasp_before_insertion"
            and result.get("preflight") is True
            and result.get("execution_started") is False
            and result.get("success") is False
            and result.get("status") == "failed"
            and isinstance(segment, dict)
            and result.get("phase") in expected_phases
            and segment.get("phase") == expected_phases[result["phase"]]
            and segment.get("status") in statuses
            and segment.get("success") is False
        ):
            # An actual held transform can be locally unreachable or leave too
            # little joint headroom even though the nominal library pose passed.
            # Require the structured live guard result, never a code-shaped
            # substring, stale selection, generic IK error or an LLM precondition.
            return selected, failure_kind
    if action in PICK_ACTIONS:
        return selected, "pick_execution"
    if action in {"move_object_to_pose", "insert_object"} and any(
        term in reason for term in ("GRASP_LOST", "GRASP_NOT_ESTABLISHED", "GRASP_NOT_LIFTED")
    ):
        return selected, "grasp_lost"
    return None, "not_a_grasp_failure"


def update_grasp_retry_state(worker, simulation):
    """Preserve measured successes and exclude only an implicated failed grasp."""
    preferences = getattr(worker, "preferred_grasp_candidates", {})
    exclusions = getattr(worker, "excluded_grasp_candidates", {})
    completed = completed_grasps(simulation)
    for row in completed:
        object_id, candidate = row["object_id"], row["candidate_id"]
        if candidate not in exclusions.get(object_id, []):
            values = preferences.setdefault(object_id, [])
            if candidate not in values:
                values.append(candidate)
    selected, reason = failed_grasp(simulation)
    excluded = None
    if selected is not None:
        object_id, candidate = selected["object_id"], _candidate(selected)
        values = exclusions.setdefault(object_id, [])
        if candidate not in values:
            values.append(candidate)
            excluded = {"object_id": object_id, "candidate_id": candidate}
        preferences[object_id] = [value for value in preferences.get(object_id, []) if value != candidate]
        if not preferences[object_id]:
            preferences.pop(object_id, None)
    worker.excluded_grasp_candidates = exclusions
    worker.preferred_grasp_candidates = preferences
    return {
        "scope": getattr(worker, "grasp_retry_scope", None),
        "completed_grasps": completed,
        "excluded_grasp": excluded,
        "failure_kind": reason,
        "preferred_grasp_candidates": {name: list(values) for name, values in preferences.items()},
        "excluded_grasp_candidates": {name: list(values) for name, values in exclusions.items()},
    }
