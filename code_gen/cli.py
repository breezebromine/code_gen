# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Develop a task program, or collect fresh layouts directly with an existing program."""

import argparse
import fcntl
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

from code_gen._io import write_json
from code_gen.llm.client import ChatClient
from code_gen.pipeline.batch import RenderYieldRequested, run_batch
from code_gen.pipeline.preparation import missing_grasp_objects, prepare_grasp_libraries
from code_gen.task.catalog import TaskCatalog
from code_gen.task.config import BatchTaskInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskinfo", type=Path, help="Simple JSON/YAML or legacy expanded JSON")
    parser.add_argument(
        "--catalog", type=Path, help="External scene/object catalog; built-in robot/run/goal defaults remain available"
    )
    parser.add_argument("--output", type=Path, help="Defaults to logs/codegen/<task_name>")
    parser.add_argument("--list-presets", action="store_true", help="List registered scenes, assets and defaults")
    parser.add_argument("--write-resolved", type=Path, help="Export the expanded v1 taskinfo for inspection")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--check-only", action="store_true", help="Validate taskinfo/resources without API or simulation"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate missing grasp libraries and save resolved inputs without API or simulation",
    )
    parser.add_argument(
        "--direct-collect",
        action="store_true",
        help="Fresh layouts; count every audited success without a development gate",
    )
    parser.add_argument("--program", type=Path, help="Existing task program for --direct-collect; no model calls")
    parser.add_argument("--phase", choices=("all", "develop", "collect"), default="all")
    parser.add_argument("--execution-backend", choices=("mgg", "isolated"), default="mgg")
    parser.add_argument("--recording", choices=("state_only", "online_rgb"), default="state_only")
    parser.add_argument("--visual-feedback", action="store_true", help="Ask the model to inspect validation images")
    parser.add_argument("--device", default="cpu", help="Physics device; cuRobo continues to use CUDA.")
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        type=int,
        default=1,
        help="Shared physics environments for direct state-only collection",
    )
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.num_envs > 1 and (not args.direct_collect or args.recording != "state_only" or args.resume):
        parser.error("Multi-env currently requires a new --direct-collect --recording state_only run")
    if args.list_presets:
        print(json.dumps(TaskCatalog(args.catalog).list(), ensure_ascii=False, indent=2))
        return 0
    if args.check_only and args.prepare_only:
        parser.error("--check-only and --prepare-only are mutually exclusive")
    if args.visual_feedback and args.recording != "online_rgb":
        parser.error("--visual-feedback requires --recording online_rgb")
    if args.direct_collect and (
        args.program is None or args.phase != "all" or args.execution_backend != "mgg" or args.visual_feedback
    ):
        parser.error("--direct-collect requires --program, --phase all, mgg backend, and no visual feedback")
    if args.program is not None and not args.direct_collect:
        parser.error("--program requires --direct-collect")
    if args.taskinfo is None:
        parser.error("--taskinfo is required unless --list-presets is used")
    if args.direct_collect:
        from code_gen.task.program import extract_program, validate_program

        source = args.program.read_text()
        validate_program(source)
        if extract_program(source) != source:
            parser.error("--program must be plain source with one trailing newline")
    info = BatchTaskInfo.load(args.taskinfo, catalog=TaskCatalog(args.catalog))
    if args.write_resolved and args.check_only:
        args.write_resolved.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.write_resolved, info.document)
    if args.check_only:
        print(
            json.dumps(
                {
                    "ready": True,
                    "grasp_generation_pending": missing_grasp_objects(info),
                    "grasp_import_pending": {name: str(path) for name, path in info.annotation_sources().items()},
                    "task": info.task.name,
                    "robot": info.robot,
                    "mode": "direct_random_v1" if args.direct_collect else "development_then_collection",
                    "validation_seeds": [] if args.direct_collect else info.development["validation_seeds"],
                    "collection": info.collection,
                    "hdrs": len(info.hdrs),
                    "tablecloths": len(info.cloths),
                    "catalog_files": list(info.catalog_files),
                    "diagnostic_contact_departure": info.scene.get("diagnostic_contact_departure"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    manifest = info.manifest()
    if args.execution_backend == "mgg":
        manifest["execution"] = {
            "backend": "persistent_mgg",
            "recording": args.recording,
            "device": args.device,
            "batch_planning": True,
            "visual_feedback": args.visual_feedback,
        }
    if args.num_envs > 1:
        manifest["execution"]["num_envs"] = args.num_envs
    if args.output is None:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", info.task.name):
            parser.error("Use --output for a legacy task name containing path characters")
        args.output = Path("logs/codegen") / info.task.name
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "direct_collection.lock.json").exists() and not args.direct_collect:
            raise ValueError("This output uses direct collection; resume with --direct-collect")
        config = root / "taskinfo.lock.json"
        if config.exists():
            if json.loads(config.read_text()) != manifest:
                raise ValueError("Taskinfo or input assets changed; use a new output directory")
        else:
            write_json(config, manifest)
        info = prepare_grasp_libraries(info, root)
        info.save_configuration(root)
        if args.write_resolved:
            args.write_resolved.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.write_resolved, info.document)
        if args.prepare_only:
            print(f"PREPARED {root / 'taskinfo.resolved.json'}", flush=True)
            return 0
        os.environ["LISIM_ASSETS_PATH"] = str(info.assets)

        if args.num_envs > 1:
            from code_gen.runtime.vectorized import collect_vectorized

            return collect_vectorized(info, root, source, num_envs=args.num_envs, device=args.device)

        client_factory = ChatClient
        visual_client_factory = ChatClient.for_vision

        if args.execution_backend == "isolated":
            if args.recording != "online_rgb":
                parser.error("The legacy isolated backend requires --recording online_rgb")
            return run_batch(
                info,
                root,
                client_factory,
                phase=args.phase,
                visual_feedback=args.visual_feedback,
                visual_client_factory=visual_client_factory,
            )
        from code_gen.pipeline.batch import ProductionSeedRunner
        from code_gen.runtime.session import ProductionSession

        with ExitStack() as stack:
            try:
                if args.direct_collect:
                    from code_gen.pipeline.direct_collection import DirectSeedRunner, run_direct_collection

                    direct_runner = None

                    def execute(directory, seed, program):
                        nonlocal direct_runner
                        if direct_runner is None:
                            session = stack.enter_context(
                                ProductionSession(info, recording=args.recording, device=args.device)
                            )
                            direct_runner = DirectSeedRunner(info, session, root)
                        return direct_runner(directory, seed, program)

                    return run_direct_collection(info, root, source, execute)
                session = stack.enter_context(ProductionSession(info, recording=args.recording, device=args.device))
                return run_batch(
                    info,
                    root,
                    client_factory,
                    runner=ProductionSeedRunner(info, session),
                    phase=args.phase,
                    visual_feedback=args.visual_feedback,
                    visual_client_factory=visual_client_factory,
                )
            except RenderYieldRequested:
                # Kit close may force exit 0; acknowledge explicitly for the scheduler.
                (root / ".render_yielded").write_text("ready\n")
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
