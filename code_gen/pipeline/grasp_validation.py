# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate grasps for single-object lift or pick/place tasks across seeds."""

import copy
import hashlib
import json
from pathlib import Path

from code_gen._io import resolve_path, write_json
from code_gen.task.config import BatchTaskInfo
from isaaclab_arena.grasping.library.schema import calibration, checked_library


def test_program(object_id, grasp_id, goal):
    """Execute the original final goal, not a weaker synthetic success check."""
    action = f"    result = yield ctx.pick(object_id={object_id!r}, grasp_id={grasp_id!r})\n"
    action += '    if not result["success"]:\n        return\n'
    if hasattr(goal, "target_id"):
        action += f"    yield ctx.place(target_id={goal.target_id!r})\n"
    return "def run(ctx):\n" + action


def validation_goal(info, object_id):
    """Reject tasks the fixed pick/place probe cannot execute, before simulation."""
    from code_gen.task.spec import LiftGoal, PlacementGoal

    if info.task.milestones or len(info.task.goals) != 1 or info.task.goals[0].object_id != object_id:
        raise ValueError(
            "Grasp validation requires one final goal for --object and no milestones; use the task pipeline"
        )
    goal = info.task.goals[0]
    if not isinstance(goal, (LiftGoal, PlacementGoal)) or object_id in info.scene.get("grasp_insertion_goals", {}):
        raise ValueError("Grasp validation supports lift or pick/place only; insertion requires the task pipeline")
    return goal


def validated_profile(info, object_id, frames, evidence, *, library):
    original = info.scene["objects"][object_id].get("grasps_by_robot", {}).get("aloha", {})
    cal = calibration(copy.deepcopy(frames))
    # Preserve notes/provenance, but prevent old orientation sweeps overriding tested frames.
    if original.get("note"):
        cal["note"] += " Source calibration: " + original["note"]
    return {
        "description": f"Physically validated Aloha grasps for {object_id}; scope is the recorded task and seeds.",
        "overrides": {"grasp_library": str(Path(library).resolve()), "grasps_by_robot": {"aloha": cal}},
        "validation": evidence,
    }


def run_validation(info, payload, object_id, output, *, seeds, limit=12, stop_after=1, recording="state_only"):
    goal = validation_goal(info, object_id)
    if not seeds or not all(type(seed) is int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct integer validation seeds")
    if not 1 <= limit <= 512 or not 1 <= stop_after <= limit:
        raise ValueError("Expected 1 <= stop-after <= limit <= 512")
    frames = checked_library(payload, resolve_path(info.scene["objects"][object_id]["usd_path"]))
    from code_gen.runtime.session import ProductionSession

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    document = copy.deepcopy(info.document)
    # Keep all candidate metadata fixed across episodes so the same environment
    # and warmed-up native planners can be reused.
    overrides = document.setdefault("scene_overrides", {}).setdefault("objects", {}).setdefault(object_id, {})
    old_cal = copy.deepcopy(info.scene["objects"][object_id].get("grasps_by_robot", {}).get("aloha", {}))
    old_cal.pop("yaw_samples", None)
    old_cal.update(calibration(frames))
    overrides.setdefault("grasps_by_robot", {})["aloha"] = old_cal
    execution_info = BatchTaskInfo(document)
    evidence = {
        "scope": "complete_declared_task_on_tested_seeds_only",
        "object_id": object_id,
        "seeds": seeds,
        "task": info.task.to_dict(),
        "input_manifest": info.manifest(),
        "library_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "collision_policy": "unchanged_from_taskinfo",
        "attempts": [],
        "accepted_ids": [],
    }
    accepted = []

    def save():
        write_json(output / "validation.json", evidence)
        write_json(
            output / "validated_grasps.json",
            {
                **{k: v for k, v in payload.items() if k not in ("grasps", "physics_validated", "scope")},
                "scope": evidence["scope"],
                "physics_validated": bool(accepted),
                "grasps": accepted,
                "validation_file": str((output / "validation.json").resolve()),
            },
        )
        if accepted:
            write_json(
                output / "object_profile.json",
                validated_profile(
                    info,
                    object_id,
                    accepted,
                    {
                        "seeds": seeds,
                        "task_name": info.task.name,
                        "scope": evidence["scope"],
                        "report": str((output / "validation.json").resolve()),
                    },
                    library=output / "validated_grasps.json",
                ),
            )

    save()
    with ProductionSession(execution_info, recording=recording) as session:
        for frame in frames[:limit]:
            row = {"grasp_id": frame["id"], "seeds": [], "passed": False}
            evidence["attempts"].append(row)
            for seed in seeds:
                directory = output / frame["id"] / f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=False)
                execution_info.prepare(directory / "input", seed)
                source = test_program(object_id, frame["id"], goal)
                program = directory / "candidate.py"
                program.write_text(source)
                print(f"GRASP_VALIDATE id={frame['id']} seed={seed}", flush=True)
                report = session.execute(
                    directory / "input/task.json", seed, "execute", directory / "simulation", program, None, {}
                )
                passed = bool(
                    report.get("execution_ok")
                    and report.get("physical_success")
                    and report.get("joint_limits_ok")
                    and report.get("program_completed")
                )
                result = {
                    "seed": seed,
                    "passed": passed,
                    "arm": report.get("arm"),
                    "error": report.get("error"),
                    "failed_action": report.get("failed_action"),
                    "grasp_evidence": report.get("grasp_evidence"),
                    "report": str((directory / "simulation/simulation.json").resolve()),
                }
                row["seeds"].append(result)
                print(f"GRASP_RESULT id={frame['id']} seed={seed} passed={passed}", flush=True)
                save()
                if not passed:
                    break
            row["passed"] = len(row["seeds"]) == len(seeds) and all(r["passed"] for r in row["seeds"])
            if row["passed"]:
                accepted.append({**frame, "physics_validated": True})
                evidence["accepted_ids"].append(frame["id"])
            save()
            if len(accepted) >= stop_after:
                break
        evidence["status"] = "validated" if accepted else "no_validated_grasps"
        save()
        print(f"GRASP_COMPLETE accepted={len(accepted)} tested={len(evidence['attempts'])}", flush=True)
    return 0 if accepted else 2
