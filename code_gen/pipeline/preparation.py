# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare missing task-object grasp libraries before model calls or simulation."""

import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from code_gen._io import PACKAGE_ROOT, PROJECT_ROOT, digest_file, resolve_path, write_json

GENERATION = {"count": 128, "yaw_samples": 16, "tilts": [0, -15, 15, -30, 30, -45, 45]}


def missing_grasp_objects(info):
    """Only objects manipulated by declared goals, not supports or distractors."""
    sources = info.annotation_sources()
    return [
        name
        for name in info.grasp_object_ids
        if name not in sources
        and not info.scene["objects"][name].get("grasp_library")
        and (
            info.scene["objects"][name].get("usd_path")
            or not info.scene["objects"][name]
            .get("grasps_by_robot", {})
            .get(info.robot["name"], {})
            .get("grasp_frames")
        )
    ]


def _generate(asset, output, log, annotation=None, source_asset=None):
    command = [
        sys.executable,
        "-m",
        "code_gen.generate_grasps",
        "import" if annotation else "generate",
        "--mesh",
        str(asset),
        "--output",
        str(output),
    ]
    if annotation:
        command.extend(["--annotation", str(annotation)])
        if source_asset is not None:
            command.extend(["--source-asset", str(source_asset)])
    else:
        command.extend([
            "--count",
            str(GENERATION["count"]),
            "--yaw-samples",
            str(GENERATION["yaw_samples"]),
            "--tilts",
            *map(str, GENERATION["tilts"]),
        ])
    # USD bootstrap may re-exec this child. Never re-exec the pipeline or hold
    # SimulationApp/GPU resources while generating a geometric library.
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Grasp preparation failed for {asset}; inspect {log}")


def prepare_grasp_libraries(info, root):
    """Return resolved execution inputs, preserving the user's catalog and input manifest."""
    from isaaclab_arena.grasping.grasp_catalog import grasp_annotation_source_asset
    from isaaclab_arena.grasping.library.schema import calibration, checked_library

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prepared = copy.deepcopy(info)
    sources = info.annotation_sources()
    missing = missing_grasp_objects(info)
    report = {"objects": {}, "generator": "code_gen.generate_grasps", "physics_validated": False}
    generation = {
        **GENERATION,
        "implementation_sha256": {
            name: digest_file(PACKAGE_ROOT / name)
            for name in (
                "generate_grasps.py",
                "pipeline/preparation.py",
            )
        },
    }
    for name in (
        "grasp_catalog.py",
        "profiles.py",
        "library/sampling.py",
        "library/schema.py",
        "library/annotations.py",
        "library/panda_annotations.py",
    ):
        generation["implementation_sha256"][f"arena/{name}"] = digest_file(
            PROJECT_ROOT / "isaaclab_arena/grasping" / name
        )
    generation["implementation_sha256"]["arena/robot_layout.py"] = digest_file(
        PROJECT_ROOT / "isaaclab_arena/grasping/robot_layout.py"
    )
    targets = info.grasp_object_ids
    for object_id in targets:
        obj = prepared.scene["objects"][object_id]
        if obj.get("grasp_library"):
            # An explicit, provenance-checked library is authoritative. Do not
            # replace it with default geometric sampling or sibling discovery.
            library_file = resolve_path(obj["grasp_library"])
            obj["usd_path"] = str(resolve_path(obj["usd_path"]))
            payload = json.loads(library_file.read_text())
            frames = checked_library(payload, resolve_path(obj["usd_path"]))
            cal = obj.setdefault("grasps_by_robot", {}).setdefault(info.robot["name"], {})
            cal.update(calibration(copy.deepcopy(frames)))
            cal.pop("yaw_samples", None)
            override = (
                prepared.document.setdefault("scene_overrides", {}).setdefault("objects", {}).setdefault(object_id, {})
            )
            override.update(usd_path=obj["usd_path"], grasp_library=str(library_file))
            override.setdefault("grasps_by_robot", {})[info.robot["name"]] = copy.deepcopy(cal)
            library_hash = digest_file(library_file)
            previous_report = root / "grasp_preparation.json"
            if previous_report.exists():
                previous = json.loads(previous_report.read_text()).get("objects", {}).get(object_id, {})
                if previous.get("library_sha256") not in (None, library_hash):
                    raise ValueError("Provided grasp library changed; use a new output directory")
            artifact = root / "grasps" / "provided" / (library_hash + ".json")
            if artifact.exists() and digest_file(artifact) != library_hash:
                raise ValueError("Modified provided grasp cache: " + str(artifact))
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(library_file.read_bytes())
            report["objects"][object_id] = {
                "status": "provided_verified",
                "source": "explicit_library",
                "count": len(frames),
                "library": str(artifact),
                "library_sha256": library_hash,
                "source_library": str(library_file),
            }
            continue
        if object_id not in missing and object_id not in sources:
            report["objects"][object_id] = {
                "status": "provided",
                "count": len(obj["grasps_by_robot"][info.robot["name"]]["grasp_frames"]),
            }
            continue
        if not obj.get("usd_path"):
            raise ValueError(f"{object_id}: automatic grasp generation requires its single-rigid-object usd_path")
        asset = resolve_path(obj["usd_path"])
        if not asset.is_file():
            raise FileNotFoundError(asset)
        safe_name = (
            object_id
            if re.fullmatch(r"[A-Za-z0-9_-]+", object_id)
            else hashlib.sha256(object_id.encode()).hexdigest()[:16]
        )
        base = root / "grasps"
        base.mkdir(parents=True, exist_ok=True)
        directory = base / safe_name
        # Normalize the path for the strict provenance check without altering source inputs.
        obj["usd_path"] = str(asset)
        annotation = sources.get(object_id)
        source_asset = grasp_annotation_source_asset(annotation, asset) if annotation else None
        library_generation = {
            **generation,
            "annotation": (
                {
                    "path": str(annotation),
                    "sha256": digest_file(annotation),
                    "source_asset": str(source_asset),
                    "source_sha256": digest_file(source_asset),
                }
                if annotation
                else None
            ),
        }
        reused = directory.exists()
        if reused:
            receipt_file = directory / "receipt.json"
            if not receipt_file.is_file():
                raise ValueError(f"Incomplete grasp cache: {directory}; use a new output directory")
            receipt = json.loads(receipt_file.read_text())
            library_file = directory / "candidates.json"
            if receipt.get("generation") != library_generation or receipt.get("library_sha256") != digest_file(
                library_file
            ):
                raise ValueError(f"Grasp cache or generator changed: {directory}; use a new output directory")
            payload = json.loads(library_file.read_text())
            frames = checked_library(payload, resolve_path(obj["usd_path"]))
        else:
            if (root / "program.json").exists() or (root / "generation_context.json").exists():
                raise ValueError(
                    f"Missing grasp cache for an existing task run: {object_id}; "
                    "use a new output directory to generate and revalidate its program"
                )
            log = base / f"{safe_name}.generation.log"
            print(f"GRASPS {'import' if annotation else 'generate'} object={object_id} asset={asset}", flush=True)
            with tempfile.TemporaryDirectory(prefix=".generating_", dir=base) as temporary:
                pending = Path(temporary) / "library"
                _generate(asset, pending, log, annotation=annotation, source_asset=source_asset)
                payload = json.loads((pending / "candidates.json").read_text())
                frames = checked_library(payload, resolve_path(obj["usd_path"]))
                write_json(
                    pending / "receipt.json",
                    {"generation": library_generation, "library_sha256": digest_file(pending / "candidates.json")},
                )
                pending.rename(directory)
        # Sampling proposes candidates only; the normal pipeline still verifies
        # collisions, actual contact/lift, placement and multi-seed task success.
        cal = obj.setdefault("grasps_by_robot", {}).setdefault(info.robot["name"], {})
        cal.update(calibration(copy.deepcopy(frames)))
        cal.pop("yaw_samples", None)
        overrides = (
            prepared.document.setdefault("scene_overrides", {}).setdefault("objects", {}).setdefault(object_id, {})
        )
        overrides["usd_path"] = str(asset)
        overrides.setdefault("grasps_by_robot", {})[info.robot["name"]] = copy.deepcopy(cal)
        write_json(
            directory / "object_profile.json",
            {
                "description": "Prepared grasp candidates; task-specific physical validation is still required.",
                "overrides": {"grasps_by_robot": {info.robot["name"]: cal}},
            },
        )
        report["objects"][object_id] = {
            "status": "cached" if reused else ("imported" if annotation else "generated"),
            "source": "sibling_annotation" if annotation else "geometric_sampler",
            "annotation": str(annotation) if annotation else None,
            "source_asset": str(source_asset) if source_asset else None,
            "count": len(frames),
            "library": str(directory / "candidates.json"),
            "library_sha256": digest_file(directory / "candidates.json"),
        }
        print(f"GRASPS {report['objects'][object_id]['status']} object={object_id} count={len(frames)}", flush=True)
    write_json(root / "grasp_preparation.json", report)
    return prepared
