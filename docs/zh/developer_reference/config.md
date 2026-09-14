# 配置

SGLang-Omni 使用声明式配置作为模型专属流水线定义与模型无关运行时之间的契约。`PipelineConfig` 描述整条流水线：模型路径、阶段列表、端点以及 relay 后端。`StageConfig` 描述一个逻辑阶段：如何构造它、它运行在哪里、普通结果发往何处，以及它是否参与扇入或流式边。

配置层有意保持静态。它应当在运行时启动之前就让拓扑、放置与阶段构造一目了然；请求期的行为属于阶段、调度器、model runner 与模型局部的负载逻辑。

## 声明式配置

流水线在模型的 `config.py` 中用 `PipelineConfig` 与 `StageConfig` 声明。阶段拓扑——存在哪些阶段、它们如何路由、请求从哪里进入——只在这里定义；配置文件与 CLI 标志可以覆盖这些阶段上的设置，但永远不会增删阶段。

示例：

```python
# Every non-TP stage must declare `process` explicitly — there is no implicit
# default. Each stage below runs in its own OS process; multiple stages can
# share an OS process by giving them the same `process` value (see
# `Qwen3OmniSpeechColocatedPipelineConfig` for that pattern).
stages = [
    StageConfig(
        name="preprocessing",
        process="preprocessing",
        factory_path="...create_preprocessing_executor",
        next=["image_encoder", "audio_encoder", "mm_aggregate"],
        project_payload={
            "image_encoder": "...project_preprocessing_to_image_encoder",
            "audio_encoder": "...project_preprocessing_to_audio_encoder",
            "mm_aggregate": "...project_preprocessing_to_mm_aggregate",
        },
    ),
    StageConfig(
        name="mm_aggregate",
        process="mm_aggregate",
        factory_path="...create_aggregate_executor",
        wait_for=["preprocessing", "image_encoder", "audio_encoder"],
        merge_fn="...merge_for_thinker",
        next="thinker",
    ),
    EngineStageConfig(               # drives an SGLang engine, so engine.* exists
        name="thinker",
        process="thinker",
        factory_path="...create_sglang_thinker_executor_from_config",
        factory=FactoryArgs(max_seq_len=8192),
        gpu=0,
        next=["decode", "talker_ar"],
        stream_to=["talker_ar"],
    ),
    StageConfig(
        name="decode",
        process="decode",
        factory_path="...create_decode_executor",
        terminal=True,
    ),
]
```

## 消费者分组

阶段设置按消费它们的模块分组：

| 分组 | 消费者 | 示例 |
| --- | --- | --- |
| 阶段顶层 | 父进程：放置、进程规划、接线 | `gpu`、`tp_size`、`process`、`gpu_memory_fraction` |
| `engine.*` | SGLang `ServerArgs`（仅存在于 `EngineStageConfig` 阶段） | `mem_fraction_static`、`max_running_requests`、`disable_cuda_graph` |
| `factory.*` | 阶段工厂的函数签名 | `dtype`、`max_seq_len`、`max_concurrency`、`enable_async_decode` |

每个分组声明自己常被调节的字段，这些字段会被即时校验。**其余任何键都原样透传**：分组的词汇表属于它们的消费者，因此入口一侧从不对未知键做格式检查。自由形式的 `factory.*` 键会以自己的名字到达阶段工厂；自由形式的 `engine.*` 键会随 `server_args_overrides` 映射传给 SGLang，由它来裁决。工厂专属旋钮就是这样传递的——过去写在 `factory_args` 里的项现在写在 `factory.*` 下：

```yaml
stages:
  latent_engine:
    factory:
      num_steps: 4          # not a declared FactoryArgs field; passed to the
                            # factory as num_steps=4, validated by its signature
```

工厂不接受的值会在阶段构造时报错，而不是静默无效。

## 设置取值：YAML 与 CLI

同一种路径语言恰好有两种面向用户的写法。

**YAML** —— `stages:` 映射，以阶段名为键：

```yaml
config_cls: MossTTSPipelineConfig
model_path: OpenMOSS-Team/MOSS-TTS

stages:
  tts_engine:
    tp_size: 2
    engine:
      mem_fraction_static: 0.7
  vocoder:
    factory:
      dtype: bfloat16
```

条目按名字合并：文件没有写的字段保留模型默认值。写出配置类未定义的阶段名是一个错误，错误信息会列出真实的阶段名。

**CLI** —— 点分标志，隐含 `stages.` 前缀；标志从阶段名开始，与映射一致：

```bash
sgl-omni serve --config omni.yaml \
    --tts_engine.tp_size 2 \
    --tts_engine.engine.mem_fraction_static 0.7 \
    --vocoder.factory.dtype bfloat16 \
    --vocoder.process vocoder
```

CLI 文本按声明的字段类型做强制转换；自由形式的分组键回退到 YAML 标量解析（`true` → bool，`7` → int）。在同级优先级下把同一条路径写两次会被拒绝，绝不会静默地以后写者为准。命令行优先于配置文件；显式的点分路径优先于下文的广播标志。

**共享取值** —— `shared:` 选择器列表把一个值一次写入多个阶段：

```yaml
shared:
  - select:
      stages: [preprocessing, latent_engine]   # or engine: true, exclude: [...]
    factory:
      num_steps: 4
```

该条目在解析前展开为每个命中阶段一个补丁。显式的按阶段条目（写在 `stages:` 下或作为点分标志）覆盖展开结果；两个 `shared` 条目写同一个叶子则冲突。

**广播标志** —— `--mem-fraction-static 0.7` 把一个值扇出到每个 SGLang 引擎阶段的 `engine.mem_fraction_static`。这是保留下来的唯一一个便利标志；点分的按阶段写法可以覆盖它而不算冲突。

**检查** —— `sgl-omni config resolve` 打印用相同参数启动时将会使用的配置（`--show config|diff|provenance`）；`sgl-omni config explain PATH` 指出某个值的设置来源以及它覆盖了什么。两者都运行与 `serve` 相同的合并逻辑。

## `StageConfig` 参考

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `name` | `str` | 必填 | 唯一的阶段标识符。由配置类设置；配置文件按名寻址阶段，且永不重命名。 |
| `factory_path` | `str` | 必填 | 指向阶段工厂的点分导入路径。 |
| `engine` | `EngineArgs` 或 `None` | `None` | SGLang ServerArgs 覆盖。只存在于 `EngineStageConfig` 阶段；写在别处是路径错误。 |
| `factory` | `FactoryArgs` | 空 | 阶段工厂的构造 kwargs，按字段名传递。未知键透传。 |
| `next` | `str`、`list[str]` 或 `None` | `None` | 普通结果路由的静态下游阶段，可为单个或列表。 |
| `terminal` | `bool` | `False` | 将阶段标记为终态；终态结果发送给 coordinator。 |
| `route_fn` | `str` 或 `None` | `None` | 请求感知的结果路由函数的点分路径。函数接收 `(request_id, stage_output)`，返回一个下游阶段名或阶段名列表。 |
| `gpu` | `int`、`list[int]` 或 `None` | `None` | 阶段的 GPU id。`None` 表示 CPU 放置。列表用于张量并行的各个 rank。 |
| `tp_size` | `int` | `1` | 张量并行 rank 数。`gpu` 为列表时必须等于 `len(gpu)`。 |
| `gpu_memory_fraction` | `float` 或 `None` | `None` | 按阶段、按 rank 的显存预算，占总物理显存的比例。多个进程共享一块 GPU 时每个阶段都必须声明。 |
| `process` | `str` 或 `None` | `None` | OS 进程组标识符。`process` 值相同的非 TP 阶段共享一个 OS 进程；每个非 TP 阶段都必须显式声明。对 TP 阶段，`process` 可选，作为派生 rank 进程名（`{process}_tp{rank}`）的前缀；未设置时以阶段名作为前缀。 |
| `env` | `dict[str, str]` | `{}` | 在该阶段的 worker 进程启动时应用的按阶段环境变量默认值；绝不覆盖 `os.environ`。 |
| `wait_for` | `list[str]` 或 `None` | `None` | 本阶段执行一个请求之前所需的上游阶段。 |
| `wait_for_fn` | `str` 或 `None` | `None` | 请求感知的扇入来源选择函数的点分路径。 |
| `merge_fn` | `str` 或 `None` | `None` | 扇入合并函数的点分导入路径。设置了 `wait_for` 时必填。 |
| `stream_to` | `list[str]` | `[]` | 流式块（如隐藏状态或 codec code）目标的静态超集。 |
| `stream_done_to_fn` | `str` 或 `None` | `None` | 请求感知的流完成目标函数的点分路径。 |
| `project_payload` | `dict[str, str]` | `{}` | 可选的"目标阶段 → 点分投影函数"映射，在写下游负载之前使用。 |
| `comm` | `CommConfig` 或 `None` | `None` | 按阶段的通信池与 Mooncake 选项。 |

路由规则：`next` 与 `terminal=True` 恰好设置其一。`route_fn` 是对已声明 `next` 的阶段可选的请求感知覆盖。扇入遵循同样的静态超集模式：把 `wait_for` 保持为可能上游阶段的完整集合，只用 `wait_for_fn` 来选择当前请求生效的子集。使用 `stream_done_to_fn` 时，请保持 `stream_to` 为静态超集，因为运行时准备会从它推导流接收方。

## `PipelineConfig` 参考

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model_path` | `str` | 必填 | Hugging Face 模型 id 或本地 checkpoint 路径。 |
| `stages` | `list[StageConfig]` | 必填 | 阶段定义。配置文件按名覆盖其上的字段，不能增删阶段。 |
| `name` | `str` 或 `None` | `model_path` | 流水线名称。用于报告与运行时识别。 |
| `entry_stage` | `str` 或 `None` | 第一个阶段 | 当第一个阶段不是入口时由配置类声明。不能从配置文件或 CLI 设置。 |
| `fused_stages` | `list[list[str]]` | `[]` | 要并置到同一运行时进程中的相邻线性阶段组。 |
| `env_defaults` | `dict[str, str]` | `{}` | 在阶段工厂导入之前应用的环境默认值。已存在的进程值优先。 |
| `mps` | `off`、`on` 或 `auto` | `off` | 面向符合条件的同 GPU worker 进程的原生 CUDA MPS 策略。 |
| `endpoints` | `EndpointsConfig` | IPC 默认值 | 端点分配设置。 |
| `placement` | `PlacementConfig` | 默认值 | 放置规划限制，例如 `max_total_gpu_memory_fraction_per_gpu`。 |
| `terminal_stages_fn` | `str` 或 `None` | `None` | 请求感知的终态阶段解析函数的点分路径。 |
| `config_cls` | `str` | 类名 | 自动存储，在加载已保存的配置文件时使用。 |

模型配置类可以声明的类级钩子：

- `stage_config_types: ClassVar[dict[str, type[StageConfig]]]` —— 命名阶段编译路径时所依据的 `StageConfig` 子类；把某个阶段映射到 `EngineStageConfig` 才使 `engine.*` 在它上面存在。
- `stage_factory_kwargs(stage_name)` —— 流水线作者在代码中传给工厂的构造 kwargs（接线，而非配置）。用户设置的同名 `factory.*` 值覆盖该钩子的值。
- `tensor_parallel_server_args_overrides(stage_name, tp_size)` / `topology_gated_custom_all_reduce_stages()` —— 启动时从解析出的 TP 拓扑推导的引擎覆盖。

派生值由阶段计算而来，无需手工维护：`resolved_entry_stage`、`terminal_stages`、`gpu_placement`。

### 阶段融合

`fused_stages` 是框架级的并置提示。它把列出的每个逻辑阶段都保留为普通的 `Stage`；它不会创建合成调度器。在运行时准备阶段，每个融合组添加一条进程并置约束，随后普通 Stage 路由就可以对符合条件的跳转使用进程内分发。一个组必须相邻、线性、非 TP，且至多落在一块 GPU 上。

## 取值如何到达工厂

两条通道为阶段工厂供值，在父进程中解析、在 worker 中按工厂签名应用：

```text
PipelineConfig.stage_factory_kwargs(name)      # author channel: code wiring
stage.factory.*                                # config channel: by field name
stage.engine.*  ->  server_args_overrides      # config channel: one dict to SGLang
```

按键而言，配置通道胜过作者通道；`server_args_overrides` 同样按键合并。工厂不接受的一个已配置键会在构造时抛错。标准 kwargs（`model_path`、`gpu_id`、`total_gpu_memory_fraction`）只在工厂签名声明了它们时才注入；`gpu_id` 归放置所有，从作者通道传入会被拒绝。

### 设备与 GPU 放置契约

一个阶段用哪块卡由放置决定：在父进程中根据 `stage.gpu`（或复制进程的 `processes.<name>.replica_devices`）做出，并作为 `gpu_id` 交给工厂。工厂的 `device` 参数只表示设备类型：`cpu` 表示把阶段留在宿主上，或者一个必须与宿主匹配的平台类型，如 `cuda`/`npu`。它从不携带索引。配置里写 `cuda:1` 会被拒绝，`factory.gpu_id` 以及 `PLACEMENT_OWNED_FACTORY_KWARGS`（`sglang_omni/config/schema.py`）中列出的两个显存比例 kwargs 同样被拒绝，因为它们由放置注入。

在工厂内部，这两个值在一个辅助函数中汇合：

1. 同时声明两个参数：`device: str | None = None` 与
   `gpu_id: int | None = None`。一个放在 GPU 上的阶段若其工厂没有 `gpu_id` 参数，会在加载任何权重之前按名字被拒绝。
2. 调用 `sglang_omni.utils.device.resolve_concrete_device(device, gpu_id)`。
   它会对照宿主平台检查请求的类型，从 `gpu_id` 取索引，只有当双方都没有给出索引时才询问宿主该进程当前坐在哪块卡上。不要在工厂里调用 `resolve_device_spec`，也不要手工拼接这两个值。
3. 自行构建 SGLang `ServerArgs` 的工厂，通过
   `sglang_omni.scheduling.sglang_backend.pin_resolved_device_type` 把解析出的类型写入
   `server_args_overrides["device"]`，该函数同样会拒绝指定了不同设备的运维覆盖。共享的
   `SGLangGenerationEngineBuilder` 已经这样做了。

`tests/unit_test/test_stage_device_contract.py` 会按这些规则扫过仓库中的每个模型、拓扑与阶段；新模型无需注册即被覆盖。由于每个 GPU 工厂都遵守这些规则，任何 GPU 进程都可以被赋予 `num_replicas`/`replica_devices`（`basic_usage/process_topology.md`）。

## 运行时准备与 Runner

运行时准备构建 runner 使用的解析后状态：

- 校验阶段名与静态拓扑
- 计算入口阶段与终态阶段
- 分配 ZMQ 端点
- 把点分的工厂、合并、路由与投影路径带入 worker 规格
- 不导入阶段工厂就解析两条 kwargs 通道
- 从阶段放置与 relay 后端构建 relay 配置
- 接线流目标与同 GPU 流式快速路径

服务同时为单进程与多进程拓扑使用 `MultiProcessPipelineRunner`。运行时准备先解析 GPU 放置，再解析进程拓扑：每个非 TP 阶段都必须显式声明 `process`，显式的 `stage.process` 以声明式方式给非 TP 阶段分组。一个进程组可以包含 CPU 阶段与至多一块 GPU 上的阶段。多个进程组共享同一块 GPU 的前提是每个阶段的 `gpu_memory_fraction` 预算都是显式的，且符合配置的放置上限。

```text
pipeline/
|-- stage_workers.py    # StageLaunchConfig, subprocess entrypoint, StageGroup
|-- runtime_config.py   # endpoint/runtime-dir/placement prep
`-- mp_runner.py        # Cross-stage orchestration and coordinator ownership
```

子进程不会重新编译流水线。主进程构建完全解析、可 pickle 的阶段/进程规格；子进程导入阶段工厂、构建调度器、构造 `Stage` 对象、报告就绪，并在同一个事件循环中运行一个或多个非 TP 阶段。

## 张量并行

阶段内部的张量并行与阶段之间的流水线并行是正交的。

```bash
sgl-omni serve --model-path ... --thinker.tp_size 4 --thinker.gpu "[0, 1, 2, 3]"
```

当 `tp_size > 1` 时，runner 为每个 TP rank 派生一个进程。每个进程运行阶段调度器与模型 worker，持有不同的 `tp_rank` 与 GPU。模型前向内部的 NCCL 集合通信保持各 TP rank 步调一致。TP 阶段的 `StageConfig.process` 是可选的；若设置，它作为派生的按 rank 进程名（`{process}_tp{rank}`）的前缀。TP rank 永远独占自己的 OS 进程。

只有 rank 0 拥有对外的阶段 IO：

- rank 0 接收来自 coordinator 或上一阶段的 ZMQ 消息
- rank 0 把工作与中止分发给 follower rank
- 所有 rank 做出相同的调度决策
- 只有 rank 0 发送下游结果或终态完成

每个 TP 阶段获得自己的 NCCL 端口分配，因此一条流水线内可以存在多个 TP 组。
