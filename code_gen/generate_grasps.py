# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI for mesh grasp generation and task-specific physical validation."""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from code_gen.pipeline.grasp_validation import run_validation
from code_gen.task.catalog import TaskCatalog
from code_gen.task.config import BatchTaskInfo
from isaaclab_arena.grasping.library.sampling import load_surface, sample_antipodal
from isaaclab_arena.grasping.library.schema import export_library


def ensure_usd():
    """Expose bundled USD for the CPU-only asset reader outside SimulationApp."""
    try:
        import pxr.Usd  # noqa: F401
    except ImportError:
        paths = sorted(glob.glob("/isaac-sim/extscache/omni.usd.libs-*"))
        if not paths or os.environ.get("CODEGEN_USD_BOOTSTRAPPED"):
            raise RuntimeError("USD reader unavailable; run inside the Isaac Sim Docker container")
        env = os.environ.copy()
        env["CODEGEN_USD_BOOTSTRAPPED"] = "1"
        env["PYTHONPATH"] = paths[-1] + os.pathsep + env.get("PYTHONPATH", "")
        env["LD_LIBRARY_PATH"] = paths[-1] + "/bin:" + env.get("LD_LIBRARY_PATH", "")
        os.execve(sys.executable, [sys.executable, "-m", "code_gen.generate_grasps", *sys.argv[1:]], env)


def load_info(args):
    info = BatchTaskInfo.load(args.taskinfo, catalog=TaskCatalog(args.catalog))
    if args.object not in info.scene["objects"]:
        raise ValueError("Requested object is absent from the task scene")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate geometric proposals, without simulation/API calls")
    generate.add_argument("--mesh", type=Path, help="Metric mesh or single-rigid-object USD")
    generate.add_argument("--count", type=int, default=64)
    generate.add_argument("--yaw-samples", type=int, default=16)
    generate.add_argument("--tilts", type=float, nargs="+", default=[0, -15, 15])
    generate.add_argument("--contact-to-tcp-m", type=float, default=0.0051650894)
    importer = sub.add_parser(
        "import", help="Import Isaac Grasp poses: Piper link6 or calibrated Panda hand to Aloha virtual TCP"
    )
    importer.add_argument("--mesh", type=Path, required=True, help="Exact runtime single-body USD")
    importer.add_argument(
        "--source-asset",
        type=Path,
        help="Annotated raw USD; otherwise resolve the exact named sibling before frame verification",
    )
    importer.add_argument("--annotation", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser(
        "validate",
        help="Validate single-object lift or pick/place tasks; milestones/insertion require the task pipeline",
    )
    validate.add_argument("--library", type=Path, required=True, help="candidates.json")
    validate.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    validate.add_argument("--limit", type=int, default=12, help="Maximum candidate frames tested")
    validate.add_argument("--stop-after", type=int, default=1, help="Stop after this many frames pass all seeds")
    validate.add_argument("--recording", choices=("state_only", "online_rgb"), default="state_only")
    for command in (generate, validate):
        command.add_argument("--taskinfo", type=Path)
        command.add_argument("--catalog", type=Path)
        command.add_argument("--object", help="Scene object ID, required with --taskinfo")
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "import":
        ensure_usd()
        from isaaclab_arena.grasping.library.annotations import import_annotation

        payload = import_annotation(args.mesh, args.annotation, args.output, source_asset=args.source_asset)
        print(f"IMPORTED {len(payload['grasps'])} candidates: {args.output / 'candidates.json'}")
        return 0
    if args.taskinfo and not args.object:
        parser.error("--taskinfo requires --object")
    if args.command == "validate" and not args.taskinfo:
        parser.error("validate requires --taskinfo and --object")
    info = load_info(args) if args.taskinfo else None
    if args.command == "generate":
        if args.mesh and info:
            parser.error("Choose --mesh or --taskinfo; scene generation uses the exact scene asset")
        asset = args.mesh or (Path(info.scene["objects"][args.object]["usd_path"]) if info else None)
        if asset is None:
            parser.error("generate requires --mesh or --taskinfo")
        if asset.suffix.lower() in (".usd", ".usda", ".usdc"):
            ensure_usd()
        mesh, provenance = load_surface(asset)
        provenance["sampling"] = {
            "method": "outer-surface antipodal ray pairs, tabletop local +Z",
            "contact_to_tcp_m": args.contact_to_tcp_m,
            "yaw_samples": args.yaw_samples,
            "tilts": args.tilts,
            "reference": "https://github.com/NVlabs/GraspDataGen",
            "implementation": "independent lightweight sampler; no pretrained model",
        }
        frames = sample_antipodal(
            mesh,
            count=args.count,
            yaw_samples=args.yaw_samples,
            tilts=args.tilts,
            contact_to_tcp_m=args.contact_to_tcp_m,
        )
        export_library(mesh, provenance, frames, args.output)
        print(f"GENERATED {len(frames)} geometric candidates: {args.output / 'candidates.json'}")
        return 0 if frames else 2
    return run_validation(
        info,
        json.loads(args.library.read_text()),
        args.object,
        args.output,
        seeds=args.seeds,
        limit=args.limit,
        stop_after=args.stop_after,
        recording=args.recording,
    )


if __name__ == "__main__":
    raise SystemExit(main())
