# 新任务生成 50 条成功数据

本教程对应当前 code_gen。一次任务开发生成一份可复用的 task.py，再用不同 seed 改变场景布局，累计保存 50 条通过验证的执行轨迹。不是让 LLM 独立生成 50 份 Python 代码。

## 已有程序：直接随机采集，不经过开发门槛

如果已经有能运行的任务程序，可以在采集容器中执行：

~~~bash
/isaac-sim/python.sh -m code_gen \
  --taskinfo my_tasks/taskinfo.json \
  --output logs/codegen/direct_trial \
  --direct-collect --program my_tasks/candidate.py --recording state_only
~~~

恢复同一输出目录时加 `--resume`。这种模式不调用 LLM，也不要求先通过三个开发布局。
每轮自动抽取并持久化一个未使用过的内部 seed，再按配置随机摆放、选择桌布/HDR；
用户不用管理 seed，`start_seed` 不再参与采样。场景必须配置了随机化，否则不同 seed
也不会使固定场景自动变化。单轮内抓取或换臂重试仍恢复相同布局。

**每条完整验证成功的数据立即计数，满目标数量结束。** 同一目录中以前开发阶段留下的
成功 demo，只要任务锁、程序哈希和成功证据一致，也直接计入配额，不要求重新采集。
抓取接触、物体跟随、碰撞、关节余量和最终放置成功检查均不放宽。

- 旧失败记录保留；新轮次使用独立的 `collection/episode_NNNNNN`，不会命中旧 seed 失败缓存。
- 中断后保留未完成文件，下一轮换新布局；已经完成的结果恢复登记，不重复执行。
- 内部 seed、场景位姿、桌布/HDR 和可获得的实测初态保存在 `logs/direct_layouts/` 等日志中。
- 布局失败继续采样；基础设施错误、程序错误或未分类错误会停下检查。
- `episodes` 是成功配额；有限的 `max_cases` 统计跨重启的所有已预留轮次，包括失败与中断；0 表示不限轮次。
- 程序需为普通源码且以一个换行结尾。更换程序/任务配置应使用新输出目录，不能改旧记录冒充同一版本。

下面各节描述的是**默认的先开发、后采集模式**，其中关于三个验证 seed 和开发成功不计配额
的说明不适用于 `--direct-collect`。桌布/HDR 独立随机抽样，允许重复，并不保证每轮使用不同图片。

## 1. 准备环境和 API

先启动项目 Docker 环境（参见仓库的 Docker 说明），再在宿主机终端选择实际运行的容器：

~~~bash
CODEGEN_CONTAINER=isaaclab_arena-latest  # 换成你的实际容器名
docker exec -it "$CODEGEN_CONTAINER" bash
~~~

当前部署方式是把完整仓库挂载进 Docker 后运行；不是单独安装 wheel 就能启动和渲染的发行包。完整流程还依赖仓库中的 scripts，Dockerfile 也不会把 code_gen 复制到镜像中。

后续命令都在这个容器终端执行：

~~~bash
cd /workspaces/isaaclab_arena
export LISIM_ASSETS_PATH=/path/in/container/to/aloha_assets
~~~

把上述占位路径替换为容器内的真实资产目录，其中需要存在 LiAuto/Robots/aloha。机器人、桌子、任务 USD、场景 JSON 和桌布/HDR 都是外部输入，必须自行准备并挂载。示例引用的 logs/codegen_assets 不随代码提交；先修改 examples/catalog/scenes/stack5_rectangle.json 中的 scene_file，或复制 catalog 后修改自己的配置，才能继续检查和运行。当前公开 taskinfo 流程使用 Aloha。

编辑 [API 配置](api_config.py)：

~~~python
API_KEY = ""                         # 填代码生成服务的 key
BASE_URL = "https://api.deepseek.com"
MODEL = "YOUR_CODE_MODEL"             # 替换为服务实际可用的模型名

VISION_API_KEY = ""                   # 留空复用代码生成的 key
VISION_BASE_URL = "https://api.deepseek.com"
VISION_MODEL = "YOUR_VISION_MODEL"    # 开启视觉检查时，必须支持图片输入

TIMEOUT_SECONDS = 90.0
~~~

显式填写的 key 优先生效；不填则继续使用环境变量或已有的 .codegen_api_key。真实 key 不要提交到 Git。视觉服务和代码服务可以独立配置；相同 key 是否可用取决于服务地址和模型权限。

默认启动命令只使用代码 agent 和物理验证。只有加上 --visual-feedback 才调用视觉 agent，不能仅凭配置了 VISION_MODEL 就认为启用了视觉检查。

## 2. 写 taskinfo：已经有场景和物体配置时

可以直接使用 [50 条模板](examples/rectangle_to_table_50.taskinfo.json)：

~~~json
{
  "task_name": "rectangle_to_table_50",
  "scene": "stack5_rectangle",
  "robot": "aloha_auto",
  "instruction": "取下顶部蓝色长方形片，放到桌面空闲区域，松开夹爪并撤离，稳定保持一秒。不要扰动其他物体。",
  "success": {
    "type": "table_placement",
    "object": "rectangle_blue",
    "target": "table_drop_zone"
  },
  "episodes": 50,
  "development": {
    "generation_seed": 0,
    "validation_seeds": [1, 2],
    "max_versions": 3
  },
  "collection": {
    "start_seed": 1000,
    "max_cases": 0
  }
}
~~~

这个例子复用 examples/catalog 中的矩形片场景描述。真实 Stack5 场景和资产需另行配置；模板中的候选抓取帧不是物理成功证明，也不代表所有随机布局均可抓取。

| 字段 | 如何填写 |
| --- | --- |
| task_name | 新任务的英文标识，只用字母、数字、下划线或连字符。 |
| scene | catalog/scenes 中 JSON 的文件名，不含扩展名。 |
| robot | aloha_auto 自动按当前物体到末端的距离选左右臂；也可以用 aloha_left 或 aloha_right。 |
| instruction | 描述抓什么、如何移动、放哪里、何时结束及约束；不要写死某个 seed 的世界坐标或抓取点编号。 |
| success | 程序实际检查的成功条件，必须与描述一致；不是由 LLM 自行决定。 |
| episodes | 采集阶段需要的成功数据数。开发阶段通过的 seed 不计入这 50 条。 |
| generation_seed / validation_seeds | 默认同一份候选代码必须通过指定的三个布局才能发布；至少两个独立验证 seed。 |
| max_versions | 开发阶段最多生成或修复几版任务代码，可设 1–10；不是每个 seed 的抓取重试次数。 |
| max_layout_attempts | 可选：开发最多尝试几个不同布局。省略时保留固定 seed 验证；例如设30，只对明确的布局采样失败或抓点搜索耗尽换seed，同一份代码仍须通过3个不同布局。失败全部保留，不计成功。 |
| start_seed | 采集从哪个 seed 开始，之后逐个增加；与开发 seed 分开。 |
| max_cases | 允许尝试多少个不同采集 seed。0 表示不设上限，直到成功满 50 条；例如 200 表示最多尝试 200 个 seed。 |

不要把 max_cases 设成 50 后期待必然得到 50 条成功数据，失败 seed 也会占用这个预算。不要在简版 taskinfo 的 collection 中填写 target_successes，成功数量由 episodes 指定。

换成其他已配置任务时，修改 task_name、scene、instruction 和 success 的物体/目标 ID 即可。物体和目标 ID 必须真实存在，修改描述不会自动创建资产或目标区域。

当前成功判定支持 table_placement、placement 和 lift。单纯拿起并保持需要 lift 的 reference、reference_point_m 等参考几何信息，不能只把 type 改成 lift。开抽屉等不属于现有技能和成功判定的任务，需要先扩展运行时能力。

## 3. 新物体或新场景：先补 catalog

如果新任务仍使用已有场景，不必重做资产。全新资产需要先完成 USD、刚体/碰撞和场景布局配置，LLM 不会自动把任意模型适配成可抓取场景。

内置 `aloha_auto`、`aloha_left`、`aloha_right` 和 `default` 运行预设由代码提供，
不需要复制 robots/default JSON。自己的 catalog 中同名文件可以覆盖默认值。
机械臂定义、抓取点转换、规划和录制实现由 Arena 公共层提供，CodeGen 负责
任务生成、验证和采集编排。MGG 与 CodeGen 共用记录时钟、物体观测和保存入口。

建议单独建立自己的目录，复用内置机器人、运行参数和成功条件：

~~~text
my_tasks/
  new_task.taskinfo.json
  catalog/
    scenes/my_scene.json
    objects/my_object.json
    runs/my_run.json          # 可选：桌布/HDR 等运行配置
~~~

可以先复制示例作为起点：

~~~bash
mkdir -p my_tasks
cp -r code_gen/examples/catalog my_tasks/catalog
cp code_gen/examples/rectangle_to_table_50.taskinfo.json my_tasks/new_task.taskinfo.json
~~~

场景配置参考 [stack5_rectangle.json](examples/catalog/scenes/stack5_rectangle.json)，主要字段是：

- scene_file：实际场景 JSON，包含 objects、targets 等内容。
- objects：把场景中的物体 ID 映射到 catalog/objects 的配置文件名。
- default_robot、default_run：如 aloha_auto、default。
- overrides：放置目标、物体补充信息等。
- randomization：哪些物体一起随机平移、旋转。

所有资源路径相对于仓库根目录，或使用容器内的绝对路径；不是相对于 taskinfo 文件的位置。宿主机路径只有挂载进容器后才能使用。

对象配置可以先写成：

~~~json
{
  "description": "待抓取物体，使用 scene_file 中的实际 USD。",
  "overrides": {},
  "instruction_hint": "使用当前物体局部抓取帧，先抓起，再放到目标区域；检查动作结果。"
}
~~~

最终目标或里程碑涉及的物体按以下顺序准备抓取库：

1. 若场景对象显式填写 `grasp_library`，先验证并使用它。库必须是 code_gen 支持的格式，源资产哈希、坐标系和 Aloha 标定都要匹配；准备阶段不会用自动采样覆盖它。
2. 否则查找 USD 同目录下关联的 Isaac Grasp JSON。如果有且只有一个 GraspDataGen 仿真验证版本，优先于未验证提案；同一优先级有多个文件仍报歧义。
3. 没有关联标注才调用几何生成器。标注存在但格式错误、坐标来源无法证明或没有可用候选时，会明确失败，不会改走自动生成。USD 物体原有的内联手写帧不覆盖这项准备流程。

“已验证版本”根据内容识别：`format=isaac_grasp`、`format_version=1.0`、
`created_with=grasp_sim`、`perturbation_validation_status=validated`，不是看文件名或时间戳。
只有没有 USD 的程序化物体，才直接使用内联帧。

例如资产旁可以同时保留：

~~~text
my_object/
  part.usda
  part_piper_guess.json
  part_piper_sim.json
~~~

导入要求 object_scale=1，源/运行时 USD 以米为单位，引用坐标变换不能带缩放或剪切。归一化后的标注需另行适配，不能直接套用。
两个 JSON 必须关联真实源 USD。优先使用 `object_file` 的文件名；缺省时可以按
同名文件关联。发现同 stem 的 `.usda/.usdc` 不等于坐标已经兼容：导入器还会检查
实际源文件、USD 组合关系，或逐网格拓扑及刚体局部坐标完全一致。类似外形和包围盒
不构成证明。声明的源 USD 缺失、不能证明来源时会停止，请保留原始资产文件。

Piper 标注使用 `piper.usd/piper.usdc`、`link6`，每条需有 `joint7` 及对应的
`bite_point`。没有 joint8 时按 -joint7 计算；显式双指必须对称，开口不能超过
Aloha 的 80 mm。当前适配的 bite 点分离轴为 X，Y 接近零，Z 约 129.998 mm。
导入器据此转换到 Aloha 的虚拟 TCP，不再把缺少这些字段当作“完全闭合”。
不满足条件的候选记录到 invalid_records，全部无效则报错。

另外保留独立的 Panda/panda_hand 标定转换；不能把任意夹爪姿态直接当作 Aloha 姿态。
导入使用正式 position/orientation，不用 pregrasp 或 post_perturbation 替代。
原夹爪上的 confidence 或仿真验证状态不代表 Aloha 任务已成功。
完整格式和标定边界见 [抓取库说明](README.md#generate-reusable-grasp-files)。

导入或生成的库放在 OUTPUT/grasps/OBJECT_ID/；显式库的校验副本放在
OUTPUT/grasps/provided/。grasp_preparation.json 记录 provided_verified、imported、
generated 或 cached。不会改写原资产、标注和 catalog。改变源资产、标注、适配代码
或标定后，旧准备缓存可能不再匹配，需要新的输出目录重新验证。

`generate_grasps validate` 仅适用于单个目标物体的 lift 或普通 pick/place，没有
milestones 或插入目标。它不调用 LLM，也不能验证任意完整装配程序；多物体、取出再
放回、精确插入等任务使用正常 pipeline。生成的 object_profile.json 会显式指向
validated_grasps.json；移动文件后需要同步修改这个路径。新 seed 仍需完整物理验证。

Stack5 这类堆叠组合需要保留各个物体的刚体身份，并作为一组随机移动。support_contacts 属于具体资产的接触建模，应按实际接触关系配置，不要把示例的蓝片/黄片关系直接照搬给新物体。

## 4. 随机位置、旋转、桌布和 HDR

改变 seed 只有在配置了随机范围时才会改变布局。示例场景已经配置整个 Stack5 组合的 XY 和 yaw 随机化：

- x_range_m: [0.15, 0.65]
- y_range_m: [-0.4, 0.4]
- yaw_range_deg: [-180, 180]

这些是原始 scene 坐标系中组合 pivot 的绝对 XY 范围，不是对初始位置再加一次偏移，也不是最终机器人世界坐标。程序会检查桌面边界和放置区域间距；过大范围可能导致布局采样失败或机械臂不可达。objects 列出一起移动的物体，targets: [] 表示目标区域保持固定。

如果还要随机桌布和 HDR，在自己的 catalog/runs/my_run.json 写：

~~~json
{
  "development": {
    "generation_seed": 0,
    "validation_seeds": [1, 2],
    "max_versions": 3
  },
  "collection": {
    "start_seed": 1000,
    "max_cases": 0,
    "target_successes": 50
  },
  "hold_seconds": 1.0,
  "visuals": {
    "hdr_dir": "my_tasks/assets/visuals/hdr",
    "tablecloth_dir": "my_tasks/assets/visuals/tablecloth",
    "hdr_intensity": 1800.0
  }
}
~~~

这里 target_successes 是运行预设的结构字段，最终仍由 taskinfo 的 episodes 覆盖。把路径替换成自己的素材池：HDR 文件夹至少两张 .hdr，桌布文件夹至少两张 .png，文件直接放在该目录下。

然后在 taskinfo 顶层加：

~~~json
"run": "my_run"
~~~

这是加到已有 JSON 对象里的字段，不是单独的完整 taskinfo。启动时使用 --catalog my_tasks/catalog。素材按 seed 抽取，允许不同 seed 抽中同一张。未配置 visuals 时，默认示例不会随机替换桌布和 HDR。

## 5. 检查配置，然后启动 50 条采集

以下命令运行附带模板。自己的任务改 --taskinfo、--catalog、--output 三个参数即可。

先做只读配置检查，不调用 API、不启动仿真：

~~~bash
/isaac-sim/python.sh -m code_gen \
  --taskinfo code_gen/examples/rectangle_to_table_50.taskinfo.json \
  --catalog code_gen/examples/catalog \
  --check-only
~~~

确认 collection.target_successes 为 50，max_cases 为 0；使用素材池时 hdrs、tablecloths 应至少为 2。grasp_import_pending 列出将读取的同级标注；grasp_generation_pending 列出的物体会在真正运行时生成抓取库。ready: true 只表示配置加载成功，不代表规划和抓取已验证。

然后启动完整流程，OUTPUT 必须使用尚不存在的新目录：

~~~bash
/isaac-sim/python.sh -m code_gen \
  --taskinfo code_gen/examples/rectangle_to_table_50.taskinfo.json \
  --catalog code_gen/examples/catalog \
  --output logs/codegen/rectangle_to_table_50_run01 \
  --phase all \
  --recording state_only
~~~

程序依次执行：

1. 准备缺失的抓取库，读取随机场景和物体信息。
2. 外层开发循环调用代码 agent 生成一版程序；每个 seed 内先以固定源码重置重试，再把未解决的执行/视觉诊断反馈给模型生成下一版，最多 max_versions 版。每次改代码后重新验证必需布局。
3. 全部开发 seed 通过后，写出固定的 task.py 和 program.json。
4. 从采集 seed 1000 开始复用同一份代码。抓取搜索和大范围搬运调用共享 ActionPlanner/cuRobo；下降、抬升等短距离约束段使用带碰撞检查的共享笛卡尔控制器。
5. 每个 seed 的普通抓取失败最多执行三次尝试，重置布局并排除已证实失败的候选；源码不变，失败动作后不继续执行生成程序。布局采样失败或预算耗尽按记录换下个 seed；基础设施、程序或未分类错误停止检查。
6. 成功数达到 50 时退出；不再继续生成。

采集阶段不会调用代码 agent 或视觉 agent。开发阶段可能调用多次 API，不是固定一次。max_cases: 0 也不保证必然完成：代码开发失败、基础设施错误、或持续不可抓的资产都需要排查。

state_only 保存轨迹、场景状态和后续 Path Tracing 所需数据，不在线渲染视频；仍会为失败诊断拍摄每次尝试的布局首帧。若需要开发阶段的视觉检查，首次启动时把最后一行换成：

~~~bash
  --recording online_rgb --visual-feedback
~~~

确保 VISION_MODEL 的服务实际支持图片输入。这种方式会启用相机记录，但不会自动执行离线 Path Tracing。录制模式和视觉检查开关会被运行配置锁定，续跑时保持一致。

可选：只准备抓取库时使用 --prepare-only；它会创建输出目录，因此后续完整启动同一个输出目录时需要加 --resume。

## 6. 后台运行、查看进度和续跑

若希望终端关闭后继续运行，可从宿主机执行下列后台启动命令；先设置 CODEGEN_CONTAINER，并替换资产占位路径。它与上一节的前台启动二选一；不要同时启动同一个输出目录。

~~~bash
docker exec -d -e LISIM_ASSETS_PATH=/path/in/container/to/aloha_assets "$CODEGEN_CONTAINER" bash -c '
cd /workspaces/isaaclab_arena || exit 1
mkdir -p logs/codegen
/isaac-sim/python.sh -m code_gen \
  --taskinfo code_gen/examples/rectangle_to_table_50.taskinfo.json \
  --catalog code_gen/examples/catalog \
  --output logs/codegen/rectangle_to_table_50_run01 \
  --phase all --recording state_only \
  > logs/codegen/rectangle_to_table_50_run01.console.log 2>&1
status=$?
printf "%s\n" "$status" > logs/codegen/rectangle_to_table_50_run01.exit_code
'
~~~

回到容器终端后查看日志：

~~~bash
tail -f logs/codegen/rectangle_to_table_50_run01.console.log
~~~

Ctrl+C 只会退出 tail，不会停止后台采集。容器关闭或主机重启仍会中断任务。

~~~bash
cat logs/codegen/rectangle_to_table_50_run01/progress.json
cat logs/codegen/rectangle_to_table_50_run01.exit_code
~~~

progress.json 会在开发验证时记录 validation_successes、validation_attempted_seeds 和 layout_failures；生成或初始化尚未完成时看控制台日志。只有 successes: 50 且 target_reached: true 才算采集完成。不要只凭进程退出码0判断成功：当前 SimulationApp.close 可能覆盖原返回码。phase=development_failed 或 development_layout_budget_exhausted 均未通过开发，不能作为成功数据。

中断后，在容器里用相同参数加 --resume：

~~~bash
/isaac-sim/python.sh -m code_gen \
  --taskinfo code_gen/examples/rectangle_to_table_50.taskinfo.json \
  --catalog code_gen/examples/catalog \
  --output logs/codegen/rectangle_to_table_50_run01 \
  --phase all --recording state_only --resume
~~~

已完成的 seed 会读取保存结果，不会重复执行；未完成的尝试保留，随后开启新的 run。已通过开发验证时，会直接复用 task.py 继续采集。先前已终结的失败 seed 不会因 --resume 被无限重试。

也可以首次用 --phase develop 只生成验证代码，完成后以同一目录、相同录制参数运行 --phase collect --resume。没有通过验证的 task.py 时不能直接 collect。

不要修改 taskinfo、catalog、源场景、素材池或 task.py 后继续沿用旧输出目录；改变任务或配置后换一个新输出目录。续跑也不会自动重置已经耗尽的开发版本/采集预算。

## 7. 找到成功数据，并确认位置确实变化

以输出目录为根：

~~~text
rectangle_to_table_50_run01/
  task.py                       # 验证通过后冻结的任务代码
  program.json                  # 代码哈希及开发 seed 验证记录
  taskinfo.input.json            # 用户输入
  taskinfo.resolved.json         # 展开的完整配置
  grasp_preparation.json         # 抓取库使用/生成记录
  grasps/                       # 自动生成的库（需要时）
  logs/                         # API 日志、任务候选、执行摘要和断点记录
    cases.json                  # seed 结果汇总，包含失败原因
  progress.json                 # 采集进度
  successful_cases.json         # 成功 seed 的结果
  successful_data.json          # 成功数据路径索引
  demo_0000/ ...                 # 成功 MGG 记录的实体目录（开发验证也可保存）
  path_tracing_jobs.json        # 成功数据的离线渲染清单
~~~

demo_* 是实体目录，直接交给 render_path_tracing.py --demo_dir。失败 seed 和 attempt 的临时目录在 case 结束后删除；失败原因在 logs/cases.json，能获取到的布局首帧在 logs/failed_layouts/。logs/demo_NNNN.success.json 的 phase 区分 development 与 collection；正式采集数量以 successful_cases.json 为准。运行中的工作目录暂存在 logs 下，完成后自动整理。旧版输出需迁移后才具有此结构。

每个成功记录包含 MGG 的 meta.json、trajectory.json、output.json、scene_states.npz、path_tracing_bundle.pkl；不再额外保留多层 simulation/run/attempt 数据副本。Aloha 控制、轨迹采样和视频均为 20Hz（每步 50ms），末端记录使用 link6 原点，规划仍使用夹爪中心 TCP。配置场景保留 240Hz 物理子步。视频取决于录制/后续渲染方式，state_only 结束后没有视频属于预期。已有 48Hz 数据不会自动重采样。

具体路径从 successful_data.json 的 cases 中读取。采集应使用同一份 task.py；
成功证据中保存的 source_sha256 应一致。要比较不同布局，可查看：

- `demo_NNNN/trajectory.json` 首条 sample 的目标物体世界位姿。
- `demo_NNNN/output.json` 首条 save 的 obs.mgg_objects_pose，其中包括底座和其他物体的世界位姿。
- `logs/demo_NNNN.recording.json` 的 scene_visuals，核对抽中的 HDR/桌布。
- 直接采集模式额外保留 `logs/direct_layouts/` 的采样场景和布局记录。

已完成 case 的 run/attempt 输入目录会被清理，不要把它们当作长期保留的布局记录。
所有位姿应比较记录值；桌布/HDR 允许重复，不能仅凭背景判断 seed 是否生效。

## 8. 后续再做 Path Tracing

仓库提供 scripts/render_mgg_bundles.py，可在采集完成后单独执行：

~~~bash
/isaac-sim/python.sh scripts/render_mgg_bundles.py \
  --manifest logs/codegen/rectangle_to_table_50_run01/path_tracing_jobs.json \
  --device cuda:0 --spp 16
~~~

这个命令会真正开始渲染。需要当前容器具备渲染所需的 GPU/OptiX 环境和原始资产，渲染失败时查看对应 demo 的 path_tracing_render.log；渲染结果与物理任务是否成功是两回事。

本教程和配置模板经过配置加载检查，编写教程不会自动启动这 50 条仿真或调用模型。

## 自定义搬运：区分物体位姿与夹爪位姿

普通拿起放桌面的任务仍优先使用 grasp_actor / place_actor。需要自定义搬运、
对齐或下降时，使用 move_object_to_pose(object_id=..., pose=..., vertical=False)。
pose 是物体原点在世界坐标系中的目标 [x,y,z,qw,qx,qy,qz]，不是夹爪位姿。
运行时会在每次动作开始时读取实测夹爪和物体位姿，换算完整的位置和旋转偏移，
再交给共享运动层；大范围搬运用 cuRobo，显式竖直段用局部笛卡尔控制。不能对尚未实际抓住的物体使用此接口。

get_target_pose 返回支撑面位姿。应根据资产原点、尺寸和最终朝向先得到物体目标；
只有原点在底面且方向约定一致等条件满足时，才能直接作为物体最终位姿。
精密短程插入使用一个连续动作：
先用 move_object_to_pose 搬运到最终物体位姿上方 approach_height（默认 2 cm），
再调用 insert_object(object_id=..., pose=最终物体位姿, approach_height=0.02)。
不要再单独调用 align_object 或额外的竖直下降；内部使用同一个笛卡尔控制器
连续完成精对齐、下降，保留跟踪补偿。已满足对齐精度时只检查稳定性，不做多余修正。

该动作要求 Aloha、cuRobo 搬运和已验证的实际抓取。approach_height 为 5–30 mm；
开始时距上方目标不超过 2 cm、0.15 rad，且必须留出横向/旋转修正的净空。
参考轨迹采用平滑插值，关节指令限制速度和加速度。物体位置误差不超过
0.5 mm、角度误差不超过 0.005 rad，并稳定 0.25 秒才进入下一阶段或返回成功。
抓取丢失、关节超限、跟踪受阻、手指意外接触或其他物体明显移动都会停止。

物理碰撞保持开启，但局部控制不执行 cuRobo 的整段碰撞认证，不是规划失败时
的自动兜底。这里读取仿真实测位姿，不通过图像猜测孔位。轨迹诊断保存到
cartesian/insertion_*.json；最终任务仍由独立的物理目标和稳定保持检查验收。现有 placement/table_placement 最终目标检查位置、释放和稳定性，不检查目标四元数；插入动作会在松爪前验收角度。若任务要求松爪后的严格朝向，还需扩展成功判定，不能把当前最终成功标志当作已验证该约束。

示例（goal 已按资产原点换算成最终物体位姿，抓取已成功）：

```python
above = [goal[0], goal[1], goal[2] + 0.02, goal[3], goal[4], goal[5], goal[6]]
result = yield ctx.move_object_to_pose(object_id="part", pose=above)
if not result["success"]:
    return
result = yield ctx.insert_object(object_id="part", pose=goal, approach_height=0.02)
if not result["success"]:
    return
result = yield ctx.open_gripper()
if not result["success"]:
    return
result = yield ctx.wait(seconds=1.0)
if not result["success"]:
    return
```

insert_object 本身不会松手，也不承诺装配已经成功；必须检查结果，再松手和验收。
需要撤离时仍应提供经过验证的撤离动作。旧 align_object 保留用于独立的上方对齐，
其 cuRobo 多次规划行为不变。


共享实现位于 isaaclab_arena/grasping/cartesian_insertion.py，通过
ActionPlanner.create_insertion_controller(env_id, arm, ...) 创建。
code_gen/runtime/cartesian.py 只负责任务观测、动作打包和记录，不包含差分 IK
或插入控制算法。其他调用方（包括 MGG）提供物理步进回调及
CartesianObservation 后可复用；现有 MGG 流程不会自动切换为此控制器。

### 接触装配与有顺序的任务

搬运仍由 ActionPlanner/cuRobo 规划。抓取下降、证明抓稳的抬升、放置下降和撤离使用
共享的短距离笛卡尔控制器。抓取点搜索只规划接近路径，后续动作在执行到该阶段时处理。每次提交关节指令前
检查机器人、自碰撞及携带物体碰撞，PhysX 资产碰撞保持开启。insert_object 的共享
物体位姿反馈控制器使用相同的关节指令碰撞检查。

贴着支撑物的零件可在场景中显式声明接触关系：

    "contact_surfaces": [
      {"object_id": "piece", "support_id": "base", "target_id": "socket", "xy_radius_m": 0.01}
    ],
    "attachment_collision": {"method": "contact", "sphere_count": 400}

仅在目标附近启用声明的接触对，只允许携带物体与支撑物之间最多 0.3 mm 的规划接触，
机器人与支撑物仍正常检查。静态支撑物用 obstacle_paths 代替 support_id，填写
planning/world_geometry.json 中的真实完整障碍路径，不使用模糊名称或全场景豁免。

contact 拟合适用于闭合凸网格，限制向外突出至 0.05 mm，并检查采样表面覆盖。
它不是精确网格碰撞；球数不足会报错，非凸网格使用原生拟合。未显式配置时保留
原生拟合和 80 个槽位。凸零件的验证结果不能直接推广到带孔、非凸或复合资产。

目标可用 reference_object_id、reference_position_m 和可选的
reference_quaternion_wxyz 声明相对支撑物的局部位姿。运行时按支撑物实际位姿
转换到世界系，不依赖初始世界坐标。

“取出、放桌上、再放回”不能只检查最终位置。完整 taskinfo 的 task.milestones
可添加按顺序验证的稳定目标：

    "milestones": [{
      "name": "released_on_table",
      "goals": [{
        "kind": "table_placement", "object_id": "piece", "target_id": "table_drop",
        "xy_tolerance_m": 0.02, "z_tolerance_m": 0.003
      }],
      "hold_seconds": 0.5
    }]

最终目标仍写在 task.goals。所有里程碑及最终稳定检查均通过，任务才算成功。
milestones.json 记录通过的步骤；planning/local_*.json 和
cartesian/insertion_*.json 保留局部轨迹与检查结果。LLM 仍只生成任务逻辑，
无需为每个 seed 重写控制器或重新生成同一任务的代码。

场景对象还可用 grasp_library 显式指定 code_gen 导出的 candidates.json。准备阶段会
校验源资产哈希、局部坐标和 Aloha TCP 标定，再复制到本次运行的 grasps/provided 目录。
显式库优先；未指定时才按第 3 节的关联与验证优先级选择资产旁标注，没有关联标注时自动生成。
已有输出目录的库发生变化会报错，需使用新的输出目录重新验证。

### 减少重复运动

`pick` 包含接近、夹紧和抬升；`place` 包含下降、松手和撤离。生成任务时不要在这些完整动作旁重复添加相同动作。需要验证独立放稳时，仍应保留任务要求的等待及 milestone。

运行时会跳过位置、姿态和夹爪状态已经满足的起始 `raise` 阶段；已经完全张开且没有持物、临时接触或不确定抓取时，重复 `open_gripper` 立即返回。真实释放和抓取验证仍正常执行。

桌面目标的 `approach_height_m` 同时影响放置接近和撤离高度。应根据场景净空验证设置，不能仅为衔接下一次抓取就调高：这也会增加放置下降和撤离的行程。大范围搬运仍由 cuRobo 规划，局部直线动作保留碰撞检查。

### 多物体装配和槽位

多物体任务可以声明最多 32 个最终目标。已声明要移动的积木按各自目标验收；底座和未声明要操作的物体仍按原有位移、旋转限制保护。milestone 使用父任务的操作对象集合，避免把之前正常归位的物体误判为被扰动。

有原始装配关系时，在拆散前保存物体相对底座的目标变换：

`T_base_goal = inverse(T_world_base_original) @ T_world_object_original`

槽位配置使用 `reference_object_id`、`reference_position_m` 和 `reference_quaternion_wxyz`。运行时根据实际底座位姿计算 `T_world_goal = T_world_base_current @ T_base_goal`，LLM 用 `get_target_pose` 读取。这里表示物体正确装入时的目标位姿，并不要求目标原点等于孔的几何中心。没有原始装配关系的资产需要先标注或估计该变换。

生成程序可以用有界循环逐块 pick、搬运、insert_object、松开和撤离。插入上方搬运高度必须与 approach_height 一致。失败反馈中的 `phase_position_error_m` 对应当前阶段；`position_error_m` 始终相对于最终插入位置，因此在上方对齐阶段相差 approach_height 是正常的。


抓取搜索与 MGG 使用同一个 ActionPlanner 搜索循环：

1. 先根据当前物体位姿生成候选，做便宜的几何和工作空间过滤。
2. 按优先/备用候选规则随机抽样，最多送入 64 个候选。
3. 每批 8 个，每批最多两次原生规划尝试。
4. 首个有成功候选的批次即停止，从成功候选中随机选一个，并直接复用接近轨迹。
5. 抓住后再规划搬运；插入、松爪和撤离在实际执行阶段检查。

插入任务可以在场景的 grasp_insertion_goals[物体 ID] 中声明 target_id、
approach_height_m 和可选的 retreat_height_m，必须与任务程序使用的目标和距离一致。
运行时先对每批候选检查接近、抓取、孔上方、插入和声明的撤离终点 IK，再让通过的
候选进入同一个 MGG 接近路径搜索。目标末端位姿由当前物体、抓取点和实时孔位组合，
底座随机旋转后仍使用相对装配关系。

放置预筛按抓取点的名义接触开度分组（向上取整不超过 1 mm），接近和下降抓取
使用张开夹爪，避免把完全闭合的手指模型当成实际夹持间隙。预检缓存不同开度的
轻量机器人模型，不反复重建搬运规划器。

这只是名义夹持的终点预筛；不提前预演整条任务，也不代表实际携物路径一定可行。
抓取后的实际夹持变换、碰撞、连续下降和最终成功仍由执行阶段检查。
自动选手先试近手，候选预算耗尽或抓取后目标 IK 失败时，有重试额度就重置换手；
两只手都失败则换布局。普通抓取任务没有这项声明时沿用原有 MGG 搜索。
各对象可以共用同一个任务循环，只替换对象 ID、槽位 ID 和对应抓取库。

闭合和插入后松爪会在最长 2 秒内等待实际接触/开度到位，固定停留时间仅作为最短
等待时间。松爪补偿最多 0.4 mm 的总开度指令（每指 0.2 mm，小于已有 0.5 mm
碰撞余量），最终仍按原来的 0.2 mm 开度误差和接触力条件验收。

松爪候选覆盖到完全张开的开度，并检查张开途中是否碰到邻近积木。如果孔边只能
小幅张开，下一次显式移动到距已释放物体超过 6 cm 的位置后，会在撤离终点补齐
完全张开。补开爪前检查整个手指运动范围，此时不豁免任何物体；开口受阻或实际
未到位仍会失败。最终放置验收标准不变。


### 多物体连续装配的运行时检查

LLM 仍只生成可复用的任务循环。运行时针对当前物体位姿逐个筛选抓取并执行，不能把几何候选、cuRobo 返回成功或单块插入成功当作整条记录成功。

- 自动采样器对闭合网格记录均匀密度重心估计。候选排序综合接触法向和重力绕夹持线的力矩；这只是排序依据，不是力闭合证明，也不替代物理验证。没有这些字段的旧抓取库仍可读取。
- 抓取搜索保留 cuRobo 接近路径的碰撞和关节限位检查。后续执行继续检查机器人、附着物、目标接触面和支撑关系，抓取点的规划成功不能替代物理抓持验收。
- 不再执行候选路径的额外预演；实际笛卡尔控制器的关节指令限制、逐步碰撞检查及跟踪误差验收保持开启。
- `move_object_to_pose` 在末端到位后等待物体稳定 250 ms，再验收物体的实际位姿（2 mm / 0.02 rad）。必要时最多进行两次经过原 ActionPlanner/cuRobo 的修正；超出 2 cm / 0.25 rad 修正范围或无法收敛时明确失败。`insert_object` 原有精度要求不变。
- 重试报告保留失败动作、参数和阶段。对应抓取失败或明确失去夹持时排除该候选；若已证实当前持有对象和候选一致，`INSERTION_RELEASE_NO_CLEARANCE` 也会排除无法放置后松爪的候选。对于 `insert_object` 的明确 `CARTESIAN_COLLISION`，只有动作开始时的持有对象、动作参数中的对象、最新选中候选的对象三者一致，才记为 `placement_path_collision` 并在当前场景内排除该候选；这表示该次实测夹持关系与放置路径不兼容，不是把它永久标成坏抓取点。普通松爪接触失败、撤离、控制器前置条件或基础设施错误不会一概归咎于上一个抓取点。
- 完整完成抓取、搬运、插入、松爪的候选可以在相同场景和抓取库的后续尝试中优先使用，仍须重新通过全部现场检查。换场景、任务或抓取库后重置这类局部经验。

执行和回放共用步数上限 `max(15000, 2500 × 任务目标数)`，避免长装配轨迹通过执行后又被固定回放上限拒绝。

默认数据生成使用 execution_only：必须通过完整动作执行、所有里程碑、最终稳定目标和关节检查；不会自动再仿真一遍关节指令。底层保留显式动作回放能力；后续 Path Tracing 则回放保存的场景状态进行渲染，两者不同。局部子任务的诊断成功不等于九块整组成功；诊断配置和产物应放在忽略目录中，避免混入提交与正式数据集。

### 随机布局在开发阶段失败

若使用大范围随机布局，可在 development 中增加 `"max_layout_attempts": 30`。优先执行 generation_seed 和 validation_seeds，再从这些seed的最大值加1顺序尝试，开发seed池必须与采集seed范围不重叠。

只对明确的布局采样失败、抓取候选搜索预算耗尽或无剩余候选换seed；代码错误和基础设施错误不按布局问题跳过。program.json 记录实际通过的 validated_seeds 与包含失败项的 validation_attempts，恢复时核对代码哈希和验证证据。此验证说明代码在已接受布局上通过，不保证每个随机布局可行；采集仍保留失败计数并只发布成功数据。


Aloha 导入抓取标注时，标注关节位置表示接触物体后的实际开度，不直接作为闭合指令。
碰撞预检仍使用 `contact_span_m`；执行时默认发送“接触开度减去 4 mm”的总夹爪开度，
保持到松爪阶段。可以在物体的 `grasps_by_robot.aloha.contact_preload_m` 中配置预压
（单位米，范围 0–0.01）；动作显式指定的 `close_width_m` 优先。自定义非导入抓取帧
沿用自己的闭合开度。预压不改变物理摩擦、碰撞几何或物体跟随/最终放置的验收阈值。

### 随机选取桌面片与堆叠目标

`randomization.assembly_scatter.select_from_scattered` 可声明多个目标候选 ID。
系统先随机散放，再从已经位于桌面的候选中均匀选一个；没有候选时换布局。
任务代码在入口读取 `ctx.snapshot()["task_selection"]` 中的 `object_id` 和
`target_id`，后续抓放步骤继续使用这两个 ID。抓取库会为所有候选准备，单个
放置目标模板也会同步替换，不需要为每个颜色重新调用 LLM。

可选 `stack_groups` 为列表，每项包含从下往上排列的 `parts` 和以米为单位的
`layer_pitch_m`。同组目标声明相同的底层参考位姿；未散放的片按原顺序紧凑
堆叠，放回目标位于下一空层。目标仍通过底座实时位姿与局部偏移合成。

### 放置后回到初始姿态

若任务要求放置后回零，在松爪、撤离之后加入：

```python
result = yield ctx.home()
if not result["success"]:
    return
```

Aloha 通过共用 ActionPlanner/cuRobo 关节目标规划回到本轮初始关节姿态，
同时检查实际关节误差不超过 0.02 rad 和 TCP 位姿。这里的零位是初始姿态，
不是强制所有关节角为零；回程碰撞检查仍然有效。
