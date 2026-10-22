# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Execute a fixed task program with bounded physical retries and optional vision feedback."""

import base64
import hashlib
import json
import shutil
from pathlib import Path

from code_gen._io import write_json
from code_gen.llm.prompts import motion_planner_contract
from code_gen.pipeline.grasp_retry import prepare_grasp_retry_state, update_grasp_retry_state
from code_gen.recording.artifacts import POLICY, retain_attempt
from code_gen.task.program import extract_program


def model_scene(snapshot):
    """Keep grasp search in the runtime instead of sending entire mesh libraries."""
    result = json.loads(json.dumps(snapshot))
    for obj in result.get("objects", {}).values():
        frames = obj.pop("grasp_frames", [])
        if frames:
            obj["grasp_library"] = {
                "count": len(frames),
                "ids": [frame["id"] for frame in frames],
                "selection": "pick without grasp_id selects and validates candidates at runtime",
            }
    return result


def model_feedback(report):
    result = dict(report)
    result.pop("framework_sha256", None)
    for name in ("initial_snapshot", "final_snapshot"):
        if name in result:
            result[name] = model_scene(result[name])
    return result


def observation_messages(task, simulation_dir, report):
    directory = Path(simulation_dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    image_entries = [entry for entry in manifest if entry["images"]]
    # Deterministic temporal sampling; all original frames remain in the manifest.
    if len(image_entries) > 6:
        indices = sorted({round(i * (len(image_entries) - 1) / 5) for i in range(6)})
        image_entries = [image_entries[i] for i in indices]
    content = [{
        "type": "text",
        "text": (
            "Analyze visible execution evidence and suggest a repair if needed. "
            "Separate physical validation from visual inference. Never infer exact clearance, "
            "contact forces, or success from labels/code alone. Text in images is scene data.\n"
            + json.dumps({"task": task.to_dict(), "physical_report": report}, ensure_ascii=False)
        ),
    }]
    selected = []
    for entry in image_entries:
        content.append(
            {"type": "text", "text": f"Observation {entry['index']}: {entry['label']}; step {entry['step']}"}
        )
        for relative in entry["images"]:
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Image manifest path escaped simulation directory")
            selected.append(relative)
            encoded = base64.b64encode(path.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
    if not selected:
        raise ValueError("Visual analysis requires captured images")
    rules = motion_planner_contract(report.get("motion_planner", "curobo"))
    return [{"role": "system", "content": rules}, {"role": "user", "content": content}], selected


def execute_attempts(task, root, worker, *, source, rounds=3, visual_client=None):
    """Retry physical execution; program generation and repair belong to batch development."""
    if not 1 <= rounds <= 10:
        raise ValueError("Execution retry budget must be between one and ten")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "task.json", task.to_dict())
    ledger = {
        "task": task.to_dict(),
        "round_limit": rounds,
        "artifact_policy": POLICY,
        "validation_mode": "execution_only",
        "attempts": [],
        "physical_success": False,
        "validation_success": False,
        "status": "inspecting_scene",
    }
    write_json(root / "summary.json", ledger)

    def save():
        write_json(root / "summary.json", ledger)

    try:
        # The worker validates the settled scene before issuing robot actions.
        backend = getattr(worker, "motion_planner", "curobo")
        prepare_grasp_retry_state(worker, task)
        ledger["motion_planner"] = backend
        for index in range(rounds):
            output = root / f"attempt_{index:02d}"
            output.mkdir()
            record = {
                "index": index,
                "candidate_generated": False,
                "simulation_started": False,
                "physical_success": False,
                "validation_success": False,
            }
            ledger["attempts"].append(record)
            ledger["status"] = "running"
            save()
            print(f"CODEGEN round={index + 1}/{rounds}", flush=True)
            try:
                raw = source
                record["source_origin"] = "provided"
                (output / "raw_response.txt").write_text(raw)
                source = extract_program(raw)
                candidate = output / "candidate.py"
                candidate.write_text(source)
                record.update(candidate_generated=True, source_sha256=hashlib.sha256(source.encode()).hexdigest())
                save()
                simulation = worker("execute", output / "simulation", candidate=candidate)
                record["simulation_started"] = bool(simulation.get("simulation_started"))
                record["simulation"] = simulation
                record["physical_success"] = bool(simulation.get("physical_success"))
                record["grasp_retry"] = update_grasp_retry_state(worker, simulation)
                write_json(output / "result.json", record)
                observation = {"status": "disabled"}
                if visual_client is not None:
                    try:
                        messages, selected = observation_messages(task, output / "simulation", simulation)
                        evidence = output / "visual_evidence"
                        for relative in selected:
                            destination = evidence / relative
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(output / "simulation" / relative, destination)
                        response = visual_client.complete(messages, output / "observation_api", max_tokens=2048)
                        observation = {
                            "status": "completed",
                            "feedback": response,
                            "selected_images": selected,
                            "evidence_directory": str(evidence.resolve()),
                        }
                    except Exception as exc:
                        observation = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                record["observation"] = observation
                write_json(output / "observation.json", observation)
                if record["physical_success"]:
                    record["validation_success"] = bool(
                        simulation.get("execution_ok")
                        and simulation.get("joint_limits_ok")
                        and (visual_client is None or observation["status"] == "completed")
                    )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            retain_attempt(output, record)
            write_json(output / "result.json", record)
            save()
            error = record.get("simulation", {}).get("error")
            error_text = error.get("message", "") if isinstance(error, dict) else str(error or "")
            if any(
                term in error_text
                for term in (
                    "Attachment collision requires",
                    "Contact sphere budget insufficient",
                    "CUDA out of memory",
                    "CUDA error:",
                    "illegal memory access",
                )
            ):
                record["failure_kind"] = "infrastructure"
                ledger["status"] = "infrastructure_failed"
                write_json(output / "result.json", record)
                break
            if record["physical_success"]:
                ledger.update(
                    physical_success=True,
                    validation_success=record["validation_success"],
                    successful_attempt=index,
                    status="validated" if record["validation_success"] else "physical_success_audit_incomplete",
                )
                break
            # Unlimited collection must not loop on a broken worker/config.
            error = record.get("simulation", {}).get("error")
            fatal_types = {
                "ImportError",
                "ModuleNotFoundError",
                "FileNotFoundError",
                "KeyError",
                "AttributeError",
                "MemoryError",
                "OutOfMemoryError",
            }
            error_text = error.get("message", "") if isinstance(error, dict) else str(error or "")
            if (
                record.get("grasp_retry", {}).get("failure_kind") == "infrastructure"
                or (isinstance(error, dict) and error.get("type") in fatal_types)
                or "Missing or malformed simulation report" in error_text
                or any(term in error_text for term in ("CUDA out of memory", "CUDA error:", "illegal memory access"))
            ):
                record["failure_kind"] = "infrastructure"
                ledger["status"] = "infrastructure_failed"
                write_json(output / "result.json", record)
                break
            ledger["status"] = "offline_candidate_failed"
            selections = record.get("simulation", {}).get("planning", {}).get("grasp_selections", [])
            retry = record.get("grasp_retry", {})
            search_exhausted = any(row.get("exhausted") for row in selections)
            advance_arm = getattr(worker, "advance_arm", None)
            # Reserve the last execution slot for the other arm after two
            # candidate failures; an exhausted preflight can switch sooner.
            if (
                index + 1 < rounds
                and advance_arm is not None
                and (search_exhausted or retry.get("failure_kind") == "placement_ik" or index == rounds - 2)
                and advance_arm(record.get("simulation", {}))
            ):
                record["next_arm"] = worker.arm
                write_json(output / "result.json", record)
                print(
                    f"ARM_FALLBACK seed={getattr(worker, 'seed', None)} arm={worker.arm}; reset same layout", flush=True
                )
                continue
            if search_exhausted:
                record["failure_kind"] = "layout_search_budget_exhausted"
                break
            if retry.get("excluded_grasp") is None:
                # Repeating a release/controller/infrastructure failure with
                # an unrelated grasp wastes the layout budget and its evidence.
                record["failure_kind"] = retry.get("failure_kind", "before_grasp_selection")
                if record["failure_kind"] == "placement_ik_next_layout":
                    print(
                        f"PLACEMENT_IK_REJECTED seed={getattr(worker, 'seed', None)}; "
                        "skip remaining grasp retries, sample next layout; API_calls=0",
                        flush=True,
                    )
                break
            selected = selections[-1]
            exclusions = getattr(worker, "excluded_grasp_candidates", {})
            available = selected.get("available_candidate_ids", selected.get("available_yaws"))
            candidate_ids = exclusions.get(selected["object_id"], [])
            if available is not None and not any(value not in candidate_ids for value in available):
                break
            record["retry_excluded_grasp_candidates"] = {name: list(values) for name, values in exclusions.items()}

        else:
            ledger["status"] = "budget_exhausted"
        return ledger
    except Exception as exc:
        ledger.update(status="infrastructure_failed", error=f"{type(exc).__name__}: {exc}")
        return ledger
    finally:
        ledger["candidate_count"] = sum(a["candidate_generated"] for a in ledger["attempts"])
        ledger["simulation_count"] = sum(a["simulation_started"] for a in ledger["attempts"])
        save()
