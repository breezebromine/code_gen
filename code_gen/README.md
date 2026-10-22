# Code generation pipeline

中文使用教程：[新任务生成 50 条成功数据](TUTORIAL.zh-CN.md)。

This package turns a declared tabletop task into one reusable Python program. The code agent generates a candidate, and the simulator executes it through the shared MGG ActionPlanner: cuRobo plans transport while checked local Cartesian controllers execute short constrained segments. An optional vision agent reviews recorded images. Physical retries reset the layout and reuse the same source; only the outer development loop asks the code agent for a new version using execution and visual feedback. A version is frozen only after the same source passes every development seed. Collection reuses that frozen source without model calls and publishes only successful MGG-format demonstrations.

## Module layout and dependency boundaries

The public commands remain `python -m code_gen` and `python -m code_gen.generate_grasps`.
Explicit endpoint/model/key configuration remains in `api_config.py`.

~~~text
code_gen/
  cli.py, generate_grasps.py   command parsing and dependency assembly
  api_config.py               explicit model endpoint configuration
  _io.py                      repository paths and complete source provenance
  task/                       taskinfo/catalog resolution, goals and program interpreter
  llm/                        model client and prompts
  pipeline/                   generation, validation, fixed-program retries and collection
  planning/
    motion.py                 task stages -> shared ActionPlanner
    grasp_selector.py         task constraints -> shared bounded grasp search
    trajectory/collision.py   task release-clearance checks and diagnostics
    constraints.py            task-declared contact regions and release limits
  runtime/
    backend.py                task execution, stepping and success evaluation
    session.py, episode.py    persistent session and episode lifecycle
    policy.py                 measured-state adapters for generated tasks
    skills.py                 task-specific grasp/transport verification
    cartesian.py              task adapters for shared line/insertion controllers
    scene.py                  compatibility for existing serialized replay bundles
    worker.py, arx5_policy.py  legacy isolated-worker/import compatibility
  recording/
    recorder.py               task context -> common recording API
    artifacts.py              task validation receipts and success-only publication
  examples/                   minimal taskinfo and external scene/object catalog examples

isaaclab_arena/
  grasping/
    action_planner.py         MGG batch search, cuRobo, line and insertion controller factories
    robot_layout.py           robot joints, jaw signs, action layout and planning TCP
    robot_config.py           planner configuration using registered robot models
    solver.py, geometry.py    frame conversion, precision scopes and measured grasp checks
    skills.py                 robot-independent geometric manipulation stages
    library/                  annotation conversion, sampling, schema and orientation ranking
  environments/
    tabletop_environment.py   declarative tabletop construction and same-topology reset
  server/robo_challenge/
    observations.py           common world/arm-base object observations
    recorder.py               common clock, serialization, scene states and replay finalization
  utils/
    file_io.py                atomic JSON and file hashes
    path_tracing_replay.py    state capture and offline replay bundles
~~~

Common Arena modules never import `code_gen`. The model client, task parser and CLI
remain usable before SimulationApp starts. Runtime-only imports stay behind
simulation entry points. Grasp annotations are read before geometric sampling is
considered; provided libraries retain their provenance and TCP calibration checks.

Both MGG and CodeGen call `RoboChallengeRecorder.record_step()` after `env.step()`
and `save_episode()` to write the recording and Path Tracing bundle. The MGG
executor keeps its vectorized scheduling. Task adapters supply active object and
container identities; common observation code handles transforms and units.

The source fingerprint covers CodeGen and all shared grasp, scene, recording and
replay implementations. Existing JSON taskinfo, grasp-library schemas and MGG
artifact layouts remain unchanged. Built-in `aloha_auto`, `aloha_left`,
`aloha_right` and `default` run presets are code defaults; external catalogs can
override them. `code_gen.runtime.scene` remains loadable for existing scene
bundles. Historical task programs using the public `ctx` APIs need no changes.
Developer scripts importing removed private modules should use the public Arena
modules shown above. Changed source/configuration fingerprints require a new run
directory; old data is not rewritten or resampled.

Run from a repository checkout mounted into the project Docker container. This is not a standalone wheel/image distribution: the wheel includes code_gen and its catalog data, but startup/rendering scripts remain in the repository and the Dockerfile does not copy this package into the image.

Run validation inside Docker. The CLI and task configuration can be imported
without starting SimulationApp; use `--check-only` for configuration validation.
Robot/motion integration requires Isaac Sim. A configuration or import check is not
evidence that a task succeeded physically.

## Minimal taskinfo

Scene setup and asset-specific grasp calibration belong in reusable catalog profiles. Taskinfo describes the intended outcome:

```json
{
  "task_name": "stack5_rectangle_to_table",
  "scene": "stack5_rectangle",
  "instruction": "将顶部蓝色长方形片拿下来，放在桌面空闲区域，不扰动其他物体。",
  "success": {
    "type": "table_placement",
    "object": "rectangle_blue",
    "target": "table_drop_zone"
  },
  "episodes": 1
}
```

See `examples/rectangle_to_table_50.taskinfo.json` and `examples/catalog/`. Before running, set the catalog's `scene_file` to your prepared Stack5 scene and set `LISIM_ASSETS_PATH` to the mounted Aloha assets root. The referenced `logs/codegen_assets` scene, USDs, robot assets and textures are external inputs; they are not shipped with the code. This rectangle example illustrates configuration and is not a claim of physically validated grasps. The public taskinfo pipeline currently accepts Aloha; the lower-level ARX5 adapter retains its basic relative-IK compatibility path.

Run inside the project Docker container:

```bash
/isaac-sim/python.sh -m code_gen \
  --taskinfo code_gen/examples/rectangle_to_table_50.taskinfo.json \
  --catalog code_gen/examples/catalog \
  --output logs/codegen/rectangle_trial \
  --recording online_rgb --visual-feedback
```

Add `--check-only` to validate configuration without Isaac Sim or API calls. `--phase develop` stops after multi-seed validation; `--phase collect --resume` uses the frozen program. Set `episodes` to the desired number of successful collection cases and `collection.max_cases` to a finite trial budget (0 means continue until the quota). Failed cases do not count toward that quota.

An external `--catalog` supplies scenes/objects and can reuse built-in robot, goal, and run profiles. Full `schema_version: 1` taskinfo remains supported. Randomization, HDR/tablecloth pools, robot configuration, and support contacts belong in scene/run profiles; grasp frames belong in object profiles.

## Direct collection with an existing program

To skip program development, supply an existing restricted task program:

```bash
/isaac-sim/python.sh -m code_gen \
  --taskinfo my_tasks/taskinfo.json --output logs/codegen/direct_trial \
  --direct-collect --program my_tasks/candidate.py --recording state_only
```

Add `--resume` for an existing output. This optional mode makes no model calls and
has no three-layout development gate. Each new episode reserves a unique random
31-bit seed before execution; configured scene and visual randomization use that
seed. You do not need to choose seeds, and `collection.start_seed` is ignored.
Randomization must still be enabled in the scene/task configuration. Arm/grasp
retries within an episode reset to the **same** layout.

Every physically successful, fully audited demonstration counts immediately.
Existing development demonstrations in this output also count when their source
hash and success evidence match the supplied program. Task/asset locks remain
unchanged. Missing or inconsistent evidence stops collection; it is not silently
replaced. A different program/configuration requires a different output directory.
All existing contact, carried-object collision, joint-margin and final-goal checks
remain active. A kinematic preflight alone never counts as successful data.

Direct episodes use `logs/collection/episode_NNNNNN`, separate from old seed caches.
`logs/direct_collection.json` reserves episode IDs/seeds and records outcomes;
`logs/direct_layouts/episode_NNNNNN/` preserves sampled scenes, HDR/tablecloth
choices and available measured initial states. Invocation records retain framework
hashes. On restart, finished results are recovered without rerunning; unfinished
staging is retained and the next episode samples a fresh layout. Old failed receipts
are never invalidated. Layout failures continue, while infrastructure, program or
unclassified failures stop the object for inspection.

The success target is still `episodes` (expanded: `collection.target_successes`).
`collection.max_cases=0` continues until quota; a finite limit counts all reserved
direct episodes, including interrupted ones, across restarts. The default develop-then-collect
mode remains available, but a direct-mode output must resume with `--direct-collect`.
Use plain program source with exactly one trailing newline so its recorded hash
matches the executed source. Visual choices are sampled independently; they may
repeat even though layout seeds do not.

## Vectorized direct collection

Use one simulator process to step several independent Aloha task episodes:

```bash
/isaac-sim/python.sh -m code_gen --taskinfo my_tasks/taskinfo.json \
  --direct-collect --program my_tasks/candidate.py --num-envs 4 \
  --recording state_only --device cuda:0 --output output/vectorized_trial
```

Each environment owns a program, local controllers, collision/planner state and
public MGG recorder. Completed or failed slots reset independently. Compatible
pose queries and grasp candidates use the shared `BatchPosePlanner` and cuRobo
`plan_batch_env`. Every query retains its own live collision world in the robot
base frame; padding never publishes extra trajectories.

Carried-object moves use robot-only batch proposals with a canonical gripper.
Before execution, every interpolated joint span is checked with that environment's
actual finger geometry, measured attachment spheres, world and self collisions.
A rejected proposal falls back to the original native carried-object planner.
Precision-constrained requests form separate batches with their stricter solver
settings. Native single-environment planners remain the fallback for incompatible
or singleton requests.

A short queue combines carried requests arriving on adjacent physics frames.
The maximum wait is four control steps (0.2 seconds at 20 Hz). Waiting slots hold
the preceding checked joint command and continue physics, recording and joint
limit checks. The current grasp is revalidated before planning; start joints and
collision worlds are read live. Cancelling/resetting a slot closes its pending
controller before reuse.

The per-episode execution timeout charges its own coroutine/validation work plus
participating batch solves and shared physics. Other environments' planning,
initialization and publication time is credited back to its deadline. This does
not raise its active execution budget or relax success tests. Episode wall time
and active/wait accounting remain recorded separately. A captured collision cache
that needs more obstacle capacity is safely retired and rebuilt after CUDA work
completes; cache growth is not classified as geometric unreachability.

Local Cartesian controllers expose differential-IK requests to the same scheduler.
Compatible requests share one Jacobian fetch and batched DLS solve, retaining
individual load compensation, speed/acceleration limits and joint-limit guards.
Scene observations share one packed GPU-to-CPU read per physics state; resets and
physics steps invalidate the cache. Per-episode phase, grasp and target metadata
remain live. Collision previews own FK/self-collision buffers rather than resizing
buffers captured by native planning CUDA graphs.

`progress.json` includes batch/fallback counts, carried batch/query counts,
Cartesian batch/query counts, deferred physical hold steps and shared state
readback/cache-hit counts. Single-environment collection and MGG retain their
synchronous execution and existing default planning behavior. Grasp endpoint IK
screens and home joint-space planning still use their per-environment services;
this is not a fully batched implementation of every planner operation. The
16-environment collector remains experimental: a matched circle benchmark reached
the native home graph search and encountered a CUDA illegal-memory-access error.
Use eight environments for bounded trials rather than assuming sixteen is stable.

Automatic arm fallback resets the same layout before trying the other arm.
Physical success and the 540-degree
cumulative EE rotation limit are required before publishing a demo.

This initial mode requires a new output directory and does not support resume
or online RGB. Geometry/layout seeds are independent, while all environments
share the first sampled visual configuration because dome lighting is global.
Collection and offline rendering remain separate. Read per-environment state
and global physics timings in progress.json. Per-episode planning timers include
waiting for coalesced results and must not be summed as process CPU time.

## MGG recording contract

`recording/recorder.py` only adapts task observations to the public
`RoboChallengeRecorder`; the public recorder owns serialization and replay bundles.
`recording/artifacts.py` owns task diagnostics, success gates and render manifests.
Robot base-link indices are resolved once per episode, while poses are read live.

Aloha collection uses the same measured arm observations, camera calibration,
serializer and Path Tracing state bundle as MGG. Control, trajectory sampling and
video playback are 20 Hz (50 ms per step). Configured scenes retain 240 Hz
physics with twelve substeps per control step for thin-part contact stability.

A successful demo contains `meta.json`, `trajectory.json`, `output.json`,
`scene_states.npz` and `path_tracing_bundle.pkl`; rendering adds `videos/`
and `.path_tracing_complete`. Raw arm arrays retain MGG's batch dimension,
metres, radians and xyzw EE quaternion. The trajectory serializer exports
joint angles in degrees, jaw opening and link6-origin EE position in millimetres.
Planning continues to use the jaw-centre TCP. Object poses use metres and wxyz.

Validation receipts are stored in `logs/demo_NNNN.success.json`, with scene,
contact and planning-frame diagnostics in `logs/demo_NNNN.recording.json`.
Older in-demo `SUCCESS.json` receipts remain readable on resume. These changes
apply to new recordings; existing 48 Hz datasets are not resampled.

## Shared MGG execution layer

| Capability | Shared implementation | Task-specific adapter |
| --- | --- | --- |
| Grasp/lift candidate batches, fixed-size padding and attachments | `ActionPlanner.evaluate_pose_candidates` | Task grasp-frame conversion and cheap workspace filtering |
| Transport and arm choice | `ActionPlanner.plan_move_to_pose / nearest_arm` | Translate task goals into measured TCP goals |
| Cartesian alignment, descent and collision checks | `ActionPlanner.create_*_controller / create_cartesian_collision_session` | Contact declarations, observations and physics-step callback |
| Joint-name ordering and arm action packing | `grasping.joint_actions.write_arm_action` | Supply the active arm's command and opening width |
| Recording and offline rendering | `RoboChallengeRecorder / SceneStateBuffer / save_path_tracing_bundle` | Task evidence and robot-specific observation metadata |

The MGG executor and code generation call the shared action writer. MGG grasp
and lift searches and code generation use the same candidate-batch boundary.
The shared layer does not import `code_gen`. Task retries, LLM repair, insertion-action checks and final
task validation stay in `code_gen`; MGG's lift/drop workflow remains its
own task policy. Sharing primitives does not by itself guarantee the same
throughput or success rate for precision insertion.

For a part held by a round handle, an object profile can declare
`grasps_by_robot.aloha.follow_anchor_object_m` as the handle contact point in
the authored object frame. Follow verification then tracks that point rather
than assuming the object's origin remains rigidly fixed to the TCP. The point
must be finite and inside the authored bounds. This optional calibration keeps
the existing contact, lift and slip thresholds; it does not relax placement
position/orientation checks. Profiles without it retain origin-based tracking.

## Generated task API

The agent may generate `class GeneratedTask` with `play_once(self)`, or the existing `run(ctx)` entry point. The restricted interpreter accepts supported control flow and task APIs, not arbitrary Python or SAPIEN code. This follows RoboTwin's task-level structure but is not source-compatible with its simulator.

- Queries: `snapshot`, `get_object_pose`, `get_target_pose`, `get_grasp_pose`. Queries use the current measured scene.
- Skills: `grasp_actor`/`pick`, `place_actor`/`place`, `move_to_pose`, `move_object_to_pose`, `align_object`, `insert_object`, `move_by_displacement`, `open_gripper`, `close_gripper`, `observe`, `wait`, `home`.
- Calling pick(object_id=...) samples at most 64 candidates from the asset-local grasp_frames library for the current layout and searches them through MGG ActionPlanner/cuRobo batches of eight. A failed physical attempt resets the scene and excludes its chosen candidate before retrying the same Python program (up to three attempts per seed). A default grasp ID initializes the motion but does not lock selection. An explicit grasp_id pins an asset-local TCP pose. The runtime transforms it with the object's current position and orientation; bounded yaw/tilt candidates are checked by the planner.
- Every action is yielded and its result checked. Global transport uses cuRobo. Pick descent/lift, constrained place segments and explicit vertical moves use shared collision-checked Cartesian control; a failed global plan does not silently fall back to IK.

A generated program can remain short because the skill backend handles approach, contact, lift, transport, placement, and retreat:

```python
class GeneratedTask:
    def play_once(self):
        result = yield self.grasp_actor(object_id="rectangle_blue")
        if not result["success"]:
            return
        result = yield self.place_actor(target_id="table_drop_zone")
        if not result["success"]:
            return
```

Automatic Aloha grasp search includes a 180-degree finger swap and prioritizes approaches within 45 degrees of vertical. For upright thin parts resting on the table, `grasps_by_robot.aloha.table_clearance_m` adds a calibrated vertical grip offset (default 3 mm, configurable from 0 to 3 mm). Candidate contact width determines the pregrasp opening; finger contact with the selected object is allowed during descent, while arm, table and other-object collisions remain checked.

Insertion endpoints that match a declared table-placement goal use its XY/Z tolerances, capped at 1 mm, to decide when to hold for release. Above-target alignment retains the controller's tighter precision. Final acceptance still checks the released object against the original task goals.

For Aloha, `yield ctx.home()` returns the selected arm to the episode's initial joint configuration through ActionPlanner/cuRobo joint-space planning. Call it after release and retreat, and check `success`. Completion checks measured joints within 0.02 rad as well as the original TCP pose; it does not mean assigning literal zero to every joint.

Closing the gripper establishes only a pending grasp. Object-specific contact on both fingers, nonzero jaw opening, and measured object following during a proof lift are required before declaring it held. An explicit `close_gripper` must be followed by a vertical proof lift before transport. Missing contact sensors or a stationary object fail the attempt.

For initially touching objects, scene contact declarations describe the permitted object/support relationship. The shared Cartesian collision session checks each proposed joint command, keeps robot/support checks active, and applies only the declared carried-object contact allowance. PhysX collisions remain active. These declarations are asset metadata; generated programs cannot disable collisions.

## Model configuration, validation, and artifacts

Edit api_config.py directly: API_KEY, BASE_URL, MODEL configure the code agent;
VISION_API_KEY, VISION_BASE_URL, VISION_MODEL configure the vision agent.
The vision model must accept image input. An empty VISION_API_KEY reuses the code agent key.
TIMEOUT_SECONDS applies to both agents.

Explicit keys in api_config.py take priority. Empty values fall back to DEEPSEEK_API_KEY
for the code key and CODEGEN_VISION_API_KEY for the vision key. If neither a code key
nor its environment variable is set, the client reads the existing Git-ignored .codegen_api_key file. Literal keys
filled into api_config.py are local credentials and should not be committed.

Use `--recording online_rgb --visual-feedback` to enable the vision agent during development. The outer development loop receives execution errors and visual observations after bounded fixed-program retries. A failed version can then be repaired; every required seed is then tested again for the new version. Identical failed code is rejected without repeating simulation. Vision cannot override failed physical validation.

Outputs under the selected log directory include:

- `logs/development/version_*/candidate.py`, generation request/response records, and per-seed diagnostics.
- Execution summaries and vision feedback under `logs/`; transient attempt directories are removed after their case is finalized.
- `task.py` and `program.json` only after every required seed passes.
- Successful collection indexes and MGG-format scene-state bundles for offline rendering. The default validation mode is execution-only; action replay remains an explicit worker mode. Collection makes no code-agent or vision-agent calls.

A failed grasp caused by invalid asset calibration, unreachable geometry, or collisions may still require fixing the asset profile; code repair cannot guarantee success.

The cuRobo changes in `submodules/IsaacLab` are part of the runtime dependency. Commit them in that repository before recording an updated submodule pointer in this repository.

## Generate reusable grasp files

`generate_grasps.py` builds object-local Aloha virtual-TCP poses from the actual mesh surface. It independently implements antipodal ray/contact-pair sampling, inspired by [NVlabs/GraspDataGen](https://github.com/NVlabs/GraspDataGen)'s geometric-proposal/physics-validation workflow. It does not use pretrained weights. The sampler keeps holes and authored scale, varies contact height, lateral position and approach tilt, and checks contact-normal alignment and jaw opening. Geometric candidates do not establish gripper clearance or physical stability.

For every object referenced by a final goal or milestone, preparation uses this order:

1. An explicit scene-object `grasp_library` selects a code_gen library, checked against the asset hash, local frame and Aloha calibration. It takes priority over sibling discovery and sampling.
2. Otherwise, look beside the runtime USD for an associated Isaac Grasp JSON. A unique export with `format=isaac_grasp`, `format_version=1.0`, `created_with=grasp_sim` and `perturbation_validation_status=validated` takes priority over unvalidated proposals. Two equally preferred matches are ambiguous; filenames and timestamps are not ranking criteria.
3. Only when no associated annotation exists, generate up to 128 geometric candidates. Invalid or ambiguous annotations do not silently fall back to sampling. Existing inline catalog frames do not override this preparation for USD assets.

Imported labels require object_scale=1 and metre-scale source/runtime USD stages; reference-frame transforms must be rigid, without scale or shear. Normalized labels need an explicit inverse-normalization adapter. Association must identify the actual source USD, normally through `object_file`'s basename; without that field, a same-stem sibling can identify it. Finding a same-stem `.usda`/`.usdc` variant is only a discovery step. Import must prove the runtime frame from the exact source asset, its USD prim-stack composition, or identical mesh topology and coordinates in the body-local frame. Similar shapes or bounds are insufficient. A missing declared source file or an unproven mapping stops preparation. Single-rigid-body `reference_link` frames are resolved from that source; articulated assets are not flattened into one rigid object.

Piper imports require `piper.usd`/`piper.usdc` and `gripper_frame_link=link6`. Each candidate needs `joint7` and its per-row `bite_point` in the calibrated source link6 convention. An omitted `joint8` means `-joint7`; an explicit second jaw must be symmetric. The supported bite location has separation along X, Y approximately zero, and Z approximately 129.998 mm. The importer uses the measured bite midpoint to convert the source pose to the Aloha virtual TCP and rejects apertures above 80 mm. Missing calibration evidence is recorded as an invalid candidate; an entirely unusable library stops preparation. There is no guessed full-close fallback.

The Panda adapter remains separate: it accepts the supported `franka_panda.usd`/`franka_panda.usdc`, `panda_hand`, +Z-approach convention with per-row finger and bite geometry. Both adapters use canonical `position/orientation`, not pregrasp or post-perturbation poses. Nonpositive-confidence candidates are excluded. Source validation/confidence is provenance, not proof of success with Aloha; normal planning, contact and task checks still apply.

Objects without a USD may retain explicit inline frames. Original assets, annotations and catalog files are never rewritten. Prepared libraries are reused only while their source, annotation, importer and calibration checksums match; changing them requires a new output directory.

Generated files live under OUTPUT/grasps/OBJECT_ID/ (candidates.json, candidates.npz, surface.obj, preview.png and a reusable object_profile.json). OUTPUT/grasp_preparation.json records provided_verified/imported/generated/cached libraries. All seeds share the generated library; --resume verifies its checksum, source-asset and annotation dependency hashes, TCP calibration and preparation version before reuse. Changed inputs require a new output directory. Source assets and catalog profiles are never rewritten.

Use the normal task command to run preparation and the complete pipeline. To inspect preparation alone, add --prepare-only; this generates missing libraries and writes taskinfo.resolved.json / scene.resolved.json without LLM calls or simulation. --check-only remains read-only and lists grasp_import_pending and grasp_generation_pending. An empty/unusable library stops before execution; generated candidates remain unvalidated until the normal physical task checks pass.

Manual generation from the exact asset referenced by a task (inside Docker):

```bash
/isaac-sim/python.sh -m code_gen.generate_grasps generate \
  --taskinfo code_gen/examples/rectangle_to_table_50.taskinfo.json \
  --catalog code_gen/examples/catalog --object rectangle_blue \
  --count 64 --output logs/codegen_grasps/rectangle
```

Alternatively use `generate --mesh /path/to/object.usd --output /path/to/output` for a single rigid object, or a mesh already expressed in metres. USD export uses the rigid-body frame, converts stage units, and preserves scale. Composed multi-body assets are rejected rather than merged into a fake rigid object. The lightweight sampler currently assumes local +Z is up and targets parallel-jaw tabletop grasps within the task skill's 5 cm origin-offset bound. Aloha contact-to-TCP compensation defaults to the existing calibrated 5.165 mm and can be changed with `--contact-to-tcp-m`; that value is gripper-specific.

For a single-object lift or ordinary pick/place task, validate candidates across seeds:

```bash
/isaac-sim/python.sh -m code_gen.generate_grasps validate \
  --taskinfo code_gen/examples/rectangle_to_table_50.taskinfo.json \
  --catalog code_gen/examples/catalog --object rectangle_blue \
  --library logs/codegen_grasps/rectangle/candidates.json \
  --seeds 0 1 2 --limit 12 --stop-after 1 \
  --output logs/codegen_grasps/rectangle/validation
```

Validation makes no model calls. This helper accepts exactly one final goal for the selected object, with no milestones or declared insertion goal; it probes pick plus lift/ordinary placement. Multi-object, ordered or insertion tasks must use the normal task pipeline. The helper runs Aloha through the shared planning/control layer and PhysX, with object-specific contact/lift checks and the declared final goal. It reuses one environment/planner session and resets before each frame/seed. A frame must pass all requested seeds; execution/planning success alone cannot promote it. Existing task collision policies, including declared support departures, are retained and recorded. Results establish suitability only for that robot, task, asset version and tested seeds.

Files:

- `candidates.json`: candidate IDs, object-local TCP `[x,y,z,qw,qx,qy,qz]`, jaw widths, surface contact pairs, asset hashes and `physics_validated: false`.
- `candidates.npz`: poses, widths and IDs; load with `allow_pickle=False`.
- `surface.obj`, `preview.png`: exported surface and a preview of the first 16 candidates.
- `validation/validation.json`: seed results and paths to complete physics/planning diagnostics.
- `validation/validated_grasps.json`: only frames that passed every requested seed; empty when none passed.
- `validation/object_profile.json`: generated only when at least one frame passes; its explicit `grasp_library` references `validated_grasps.json`, preventing preparation from replacing the accepted frames.

To reuse accepted grasps, copy `object_profile.json` into your catalog's `objects/` directory and reference its filename stem in the scene's `objects` mapping. Keep its referenced library accessible; when relocating files, update that path. The generated task can call `pick(object_id=..., grasp_id=...)`; the runtime composes the stored local frame with the current measured object pose. New collection seeds still undergo full physical validation.

Runtime grasp evidence requires initial bilateral object contact and an actual lift. During motion only, force gaps shorter than 80 ms are tolerated while the measured object follows the TCP within 6 mm; persistent loss, missing sensors, or slip fails the attempt. Final pick verification still requires bilateral contact. Candidate IDs, selected local frame IDs and retry exclusions are retained in planning reports.

For custom placement of a held object, use
`move_object_to_pose(object_id=..., pose=[x,y,z,qw,qx,qy,qz], vertical=False)`.
The pose is the desired world pose of the **object origin**. Each call converts it
to a TCP goal using the current measured grasp transform, including rotation.
The object must have passed the physical grasp/proof-lift checks. This action uses
the same collision-aware planner and does not release the object.
Support-surface poses must first be converted to the intended object-origin pose.
For short precision insertion, transport with cuRobo to the final object pose plus
the approach height in world Z. Then call
`insert_object(object_id=..., pose=final_object_pose, approach_height=0.02)`.
It calls ActionPlanner.create_insertion_controller(), which continuously aligns
and descends using one shared IsaacLab differential IK controller,
preserving load compensation across the transition. Do not add a separate
`align_object` or planned descent around this action. The separate alignment API
remains available for standalone alignment through cuRobo.

The local controller requires Aloha/cuRobo transport and a physically verified
grasp; approach height must be 5–30 mm, initial above-goal error <=20 mm/0.15 rad.
Choose an above pose with clearance for lateral/rotational alignment.
A quintic pose reference limits nominal translation to 4 mm/s during alignment
and 10 mm/s during descent; angular alignment is limited to 0.05 rad/s. Joint
commands are limited to 0.4 rad/s and 1.5 rad/s²; integral compensation is capped
at 1 mm per axis. Completion requires measured object error <=0.5 mm/0.005 rad
and a stable 0.25 s window. Grasp loss, joint limits, tracking error, unexpected
finger contact and movement of other objects stop the action.

PhysX collisions remain active, but this explicitly requested local controller
does not perform cuRobo path collision certification. It is never an automatic
fallback for global planning failures. It uses simulation poses, not image-based
hole detection. Per-step references, measured poses and results are saved under
`cartesian/insertion_*.json`. It does not open the gripper: check success, release,
then allow independent task/hold validation. A successful action alone does not
certify hole clearance, collision-model coverage or final assembly. Existing final placement goals check position, release and stability, not a target quaternion. Precise orientation is checked by the insertion action before release; strict post-release orientation acceptance requires an additional goal type or validator.

The implementation is shared under
`isaaclab_arena/grasping/cartesian_insertion.py`, exposed through
`ActionPlanner.create_insertion_controller(env_id, arm, ...)`.
`code_gen/runtime/cartesian.py` only adapts task observations, arm commands,
grasp verification, recording and the return to planned motion. The shared
controller has no dependency on code_gen and does not step an environment itself:
its caller supplies `step(arm_joint_command) -> CartesianObservation`.
It supports selecting a single arm/environment; simultaneous batch control is
not implied. Existing MGG planning stays unchanged until it explicitly calls this API.


### Staged pick and insertion

Grasp selection follows MGG: cheap geometric/workspace filtering, a maximum
of 64 sampled candidates, batches of eight and two native planning attempts.
Both call ActionPlanner.iter_grasp_candidate_batches and stop after the first
successful batch, choosing a successful candidate with the planner's seeded RNG.
Failed batches are not rescanned
in extra search rounds. The selected approach trajectory is reused for execution.

Ordinary picks validate the collision-aware approach path. For insertion tasks,
a scene entry under `grasp_insertion_goals[object_id]` declares `target_id`,
`approach_height_m` and optional `retreat_height_m`. Before planning an approach,
each batch is screened with the shared ActionPlanner endpoint IK service for
approach, grasp, above-slot, insertion and declared retreat poses. The destination
TCP is composed from the live object/target poses and each candidate grasp frame.
Only survivors enter MGG's trajectory search; the first successful batch wins.

Insertion candidates are grouped by nominal contact aperture (rounded up within
1 mm) so destination IK checks the occupied hand rather than fully closed fingers.
The lightweight IK model omits unused attachment slots, caches finger models and
shares the live world checker. This filter does not certify the
carried object's path, actual finger opening, continuous IK branch or physical
success. Those remain checked during execution with the measured grasp transform.
Both arms exhausting their bounded search rejects this layout, not all possible
grasps. A destination IK failure after pickup tries the other arm immediately
when an automatic-arm retry remains. The task program is reused without model calls.

After a verified physical grasp, cuRobo transports the measured object above the
live socket pose. One Cartesian controller aligns and descends, preserving load
compensation. Each proposed command retains robot and carried-object collision
checks; release, retreat and final physical success are verified when executed.
No predictive insertion rollout is required before starting this controller.

Gripper dwell is a minimum: closing and insertion release wait up to two seconds
for measured contact/aperture convergence. Release compensates static tracking
bias by at most 0.4 mm of total aperture (0.2 mm per finger, inside the existing
0.5 mm clearance reserve). The final 0.2 mm aperture tolerance and contact
validation stay unchanged. Release searches up to the full aperture and checks
the intervening finger sweep against neighboring objects. If only a partial
opening is safe at the socket, the next explicit move more than 6 cm away from
the released object finishes opening at the reached retreat pose. This final
finger sweep is collision-checked with no object exemptions, and blocked jaws
fail the action. The original final placement checks remain in effect.


Failed-layout diagnostics keep one settled, pre-action head-camera image in
logs/failed_layouts/ and the reason in logs/cases.json. The state-only runner
renders this image once per attempt without recording RGB video or rendering
each physics step. Completed successful cases do not retain failed-layout images.
After both arm budgets are exhausted, collection samples another seed. Candidate
exclusions remain scoped to the current layout and arm.
Failures before a scene exists record image unavailability explicitly.

### Selecting a scattered stack member

Assembly randomization optionally accepts `select_from_scattered`, a list of
possible manipulated object IDs. The runtime scatters the scene first, chooses
uniformly among those candidates already on the table, and exposes
`ctx.snapshot()["task_selection"]` with `object_id` and `target_id`. Layouts
without an eligible candidate are rejected. Grasp preparation covers every
candidate, and the single placement goal template follows the chosen object.

For layered assemblies, `stack_groups` contains ordered bottom-to-top `parts`
and a positive `layer_pitch_m`. Members declare the same bottom-layer target
reference pose. Retained layers are packed in their original order and the
selected piece targets the next free layer. Target poses follow the live base
transform; they are not fixed world coordinates. These per-episode metadata
changes do not force reconstruction of an unchanged simulation scene.
