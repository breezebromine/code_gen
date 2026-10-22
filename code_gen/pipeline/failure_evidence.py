"""Recognize measured rejections without hiding malformed or unknown execution failures."""

import ast
import math


def _finite_number(value):
    return type(value) in (int, float) and abs(value) <= 1e308 and math.isfinite(value)


def _finite_evidence(value):
    if isinstance(value, dict):
        return all(_finite_evidence(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_evidence(v) for v in value)
    return type(value) not in (int, float) or _finite_number(value)


def joint_limit_rejection(message):
    """Accept only the controller's bounded, numeric pre-physics rejection diagnosis."""
    prefix = "CARTESIAN_COMMAND_JOINT_LIMIT: "
    if not message.startswith(prefix) or len(message) > 65536:
        return False
    try:
        diagnosis = ast.literal_eval(message[len(prefix) :])
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return False
    if not isinstance(diagnosis, dict) or diagnosis.get("event") != "joint_limit_rejected":
        return False
    if not _finite_number(diagnosis.get("time")) or diagnosis["time"] < 0:
        return False
    target = diagnosis.get("target_tcp")
    if not isinstance(target, list) or len(target) != 7 or not all(_finite_number(v) for v in target):
        return False
    violations = diagnosis.get("violations")
    if not isinstance(violations, list) or not violations:
        return False
    for row in violations:
        if not isinstance(row, dict) or not isinstance(row.get("joint_name"), str) or not row["joint_name"]:
            return False
        if type(row.get("joint_index")) is not int or row["joint_index"] < 0:
            return False
        if not all(
            _finite_number(row.get(key))
            for key in ("command_rad", "measured_rad", "desired_rad", "previous_command_rad", "lower_rad", "upper_rad")
        ):
            return False
        if row["lower_rad"] >= row["upper_rad"] or row["lower_rad"] <= row["command_rad"] <= row["upper_rad"]:
            return False
    return True


def cartesian_rejection(context, error):
    """Recognize explicit physical guards only in actions that execute local motion."""
    phases = {
        "pick": {"descend", "lift"},
        "place": {"raise", "lower", "retreat"},
        "move_by_displacement": {"move"},
        "move_to_pose": {"move"},
        "move_object_to_pose": {"move"},
    }
    if error.get("type") != "RuntimeError" or context.get("phase") not in phases.get(context.get("action"), ()):
        return False
    message = error.get("message", "")
    if not isinstance(message, str) or len(message) > 65536:
        return False
    if joint_limit_rejection(message):
        return True
    if message in {
        "CARTESIAN_LINE_TIMEOUT",
        "CARTESIAN_LINE_REVERSING",
        "CARTESIAN_LINE_LATERAL_DRIFT",
    }:
        return True
    code, separator, detail = message.partition(": ")
    if not separator:
        return False
    if code == "CARTESIAN_OTHER_OBJECT_MOVED":
        return bool(detail.strip())
    if code == "CARTESIAN_LINE_TRACKING":
        try:
            distance = float(detail.removesuffix(" m"))
        except ValueError:
            return False
        return math.isfinite(distance) and distance > 0.006
    if code != "CARTESIAN_COLLISION":
        return False
    try:
        diagnosis = ast.literal_eval(detail)
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return False
    return _collision_rejection(diagnosis)


def _collision_rejection(diagnosis):
    if not isinstance(diagnosis, dict) or diagnosis.get("passed") is not False or not _finite_evidence(diagnosis):
        return False
    costs = [
        diagnosis.get(k)
        for k in ("world_collision_m", "support_collision_m", "self_collision_cost", "robot_clearance_collision_m")
    ]
    return all(_finite_number(v) and v >= 0 for v in costs) and any(v > 0 for v in costs)


def release_rejection(context, error):
    """All tested opening widths collided while releasing the explicitly held object."""
    prefix = "INSERTION_RELEASE_NO_CLEARANCE: "
    message = error.get("message", "")
    if (
        error.get("type") != "RuntimeError"
        or context.get("action") != "open_gripper"
        or context.get("phase") != "release"
        or not isinstance(context.get("held_object_id"), str)
        or not context["held_object_id"]
        or not isinstance(message, str)
        or not message.startswith(prefix)
        or len(message) > 65536
    ):
        return False
    try:
        trials = ast.literal_eval(message[len(prefix) :])
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return False
    return (
        isinstance(trials, list)
        and 0 < len(trials) <= 32
        and all(
            _collision_rejection(row)
            and _finite_number(row.get("width_m"))
            and 0 < row["width_m"] <= 0.1
            and _finite_number(row.get("clearance_m"))
            and 0 < row["clearance_m"] <= row["width_m"]
            for row in trials
        )
    )


def _goal_check(check):
    if not isinstance(check, dict) or type(check.get("success")) is not bool or not _finite_evidence(check):
        return False
    rows = check.get("checks")
    return (
        isinstance(rows, list)
        and bool(rows)
        and all(isinstance(row, dict) and type(row.get("passed")) is bool for row in rows)
        and check["success"] == all(row["passed"] for row in rows)
    )


def completed_task_rejection(simulation):
    """A completed program may still fail the independent hold or milestone acceptance."""
    if (
        simulation.get("execution_ok") is not True
        or simulation.get("program_completed") is not True
        or simulation.get("physical_success") is not False
        or simulation.get("joint_limits_ok") is not True
        or simulation.get("error")
        or simulation.get("finalization_error")
        or simulation.get("timed_out")
        or simulation.get("failed_action")
    ):
        return False
    samples, failed = simulation.get("hold_samples"), simulation.get("hold_failed_samples")
    seconds = simulation.get("hold_seconds")
    if (
        type(samples) is not int
        or samples < 1
        or type(failed) is not int
        or not 0 <= failed <= samples
        or not _finite_number(seconds)
        or seconds < 0
    ):
        return False
    final = simulation.get("goal_verification")
    if not _goal_check(final):
        return False
    required, completed = simulation.get("milestones_required"), simulation.get("milestones_completed")
    if type(required) is not int or required < 0 or not isinstance(completed, list) or len(completed) > required:
        return False
    for row in completed:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not row["name"]
            or type(row.get("step")) is not int
            or row["step"] < 0
            or not _finite_number(row.get("stable_seconds"))
            or row["stable_seconds"] < 0
            or not _goal_check(row.get("check"))
            or row["check"]["success"] is not True
        ):
            return False
    if "first_failed_hold" not in simulation:
        return False
    first = simulation["first_failed_hold"]
    if not failed:
        return first is None and final["success"] is True and len(completed) < required
    if (
        not isinstance(first, dict)
        or type(first.get("sample_index")) is not int
        or not 0 <= first["sample_index"] < samples
        or not _goal_check(first.get("check"))
        or first["check"]["success"] is not False
    ):
        return False
    # Failures cannot precede the first rejection or include a passing last sample.
    available = samples - first["sample_index"] - int(final["success"])
    if failed > available:
        return False
    if first["sample_index"] == samples - 1:
        return first["check"] == final
    return final["success"] is True or failed >= 2
