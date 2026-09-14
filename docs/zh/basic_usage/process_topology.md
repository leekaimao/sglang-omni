# 进程拓扑、副本与 GPU 共享

四种机制决定了 SGLang-Omni 流水线如何占用其 GPU。它们可以组合使用，每一种回答不同的问题：

1. **进程拓扑**定义哪些阶段在一起运行。
2. **进程副本**定义运行多少个副本。
3. **放置**定义它们在哪里运行。
4. **CUDA MPS** 改善一个 GPU 上进程之间的 kernel 调度。
5. **CUDA IPC 权重共享**消除一个 GPU 上重复的权重副本。

| 机制 | 配置位置 | 它改变什么 | 它不做什么 |
|---|---|---|---|
| 进程拓扑 | `StageConfig.process` | 非 TP 的逻辑 Process 成员关系；TP 按每个 rank 物化 | 副本数量、放置 |
| 进程副本 | `PipelineConfig.processes[<name>]` | 复制整个逻辑 Process，按请求保持固定 | 单个请求的模型并行、MPS |
| 放置 | `replica_devices` | 每个副本或 rank 落位的 GPU | GPU 上下文调度 |
| CUDA MPS | `mps` | 共置 CUDA 上下文之间的 kernel 重叠 | 路由、权重或 KV 共享 |
| CUDA IPC 权重共享 | `weight_share` | follower 以别名方式引用 leader 的不可变权重 | KV、CUDA Graph、采样器、请求状态 |

你想要哪一种：

| 你的问题 | 机制 | 它不能解决什么 |
|---|---|---|
| 哪些阶段共享一个 OS 进程及其本地状态？ | 进程拓扑 | 副本数量和 GPU 放置 |
| 哪个瓶颈 Process 需要更多容量？ | 进程副本 | 单个请求的模型并行 |
| 每个副本运行在哪个 GPU 上？ | `replica_devices` | CUDA 上下文调度 |
| 共置的进程如何利用空闲算力？ | CUDA MPS | 副本创建和请求路由 |
| 完整副本在单个 GPU 上放不进显存？ | CUDA IPC 权重共享 | KV、CUDA Graph 和请求状态共享 |

## 阶段与进程拓扑

阶段是流水线 DAG 的一个逻辑执行单元。它声明自己的工厂、接线（`next`、`stream_to`、`wait_for`）、GPU、TP 大小、运行时资源以及进程名称。

对于非 TP 阶段，`StageConfig.process` 定义 Process 成员关系。共享同一 Process 名称的阶段共享一个 OS 进程、Python 堆、asyncio 事件循环和本地分发路径；其中的 GPU 成员还共享一个 CUDA 上下文。TP 阶段独占其逻辑 Process，并为每个 rank 物化一个 OS 进程。

逻辑 Process 是分组、进程生成（spawn）和放置的边界。它不是恢复边界：任何子阶段进程死亡仍会使整个流水线停止。

```yaml
stages:
  talker_ar:
    process: talker_ar
  code2wav:
    process: code2wav
```

使用 `sgl-omni config resolve --config <config.yaml> --show config` 查看解析后的结果。

## 进程副本

副本复制的是整个逻辑 Process，而不是单个阶段。同一个 Process 内部的阶段在相同的副本索引下一起被复制。运行时将物理实例命名为 `<name>@rN`，而模型和路由仍然引用逻辑名称。

在准入时，协调器会为请求涉及的每个已复制 Process 选择一个副本。该绑定随消息传递，并在请求的整个生命周期内保持固定，覆盖 payload、流、完成和中止路径。默认策略是按 Process 的线程安全轮转（round robin），并且每个 Process 独立选择。

```yaml
processes:
  talker_ar:
    num_replicas: 2
    replica_devices: [1, 2]
  code2wav:
    num_replicas: 2
    replica_devices: [1, 2]
```

## 放置

`replica_devices` 设置每个副本使用的 GPU（包括副本 0），并覆盖该副本内每个 GPU 阶段的放置。CPU 阶段不受影响，仍留在主机上。

一个 `num_replicas > 1` 的 GPU Process 必须声明 `replica_devices`：非 TP Process 需要 `N` 个设备 id，TP 大小为 `T` 的 Process 需要 `N x T` 个。每个 GPU 阶段工厂都声明 `device: str | None = None` 和 `gpu_id: int | None = None`，并通过 `sglang_omni.utils.device.resolve_concrete_device` 解析它们，因此任何 GPU 阶段都可以被复制；缺少 `gpu_id` 的工厂会在启动时按名称被拒绝，而 `tests/unit_test/test_stage_device_contract.py` 确保每个模型都遵守这一契约。配置绝不设置 `factory.gpu_id`；`factory.device` 只命名设备类型（例如 `cpu` 表示让阶段留在主机上），且不得携带索引——具体的显卡来自 `stage.gpu` 或 `replica_devices`。不同副本可以重复使用同一个设备 id，同 GPU 数据并行正是通过这种方式表达的：

```yaml
processes:
  code2wav:
    num_replicas: 2
    replica_devices: [1, 1]
```

这会把两个非 TP 副本放在 GPU 1 上。这并不意味着一个 TP 副本的两个 rank 可以共享一个 GPU。

当 `replica_devices` 把多个 Process 组共置到一个 GPU 上时，所涉及的每个 GPU 阶段都必须声明 `gpu_memory_fraction`，无论通用的共置检查是否启用。该值是放置时的预算，而不是运行时内存限制：

```yaml
stages:
  code2wav:
    gpu_memory_fraction: 0.014
```

## CUDA MPS

MPS 只是在一个 GPU 上调度多个 CUDA 上下文。它不创建副本、不选择路由，也不共享权重、KV 或 CUDA Graph。当单个上下文已经让 GPU 饱和时，MPS 带来的可能是争用和尾延迟，而不是吞吐量。

运行时自行管理守护进程。模式（CLI 上的 `--mps` 或配置中的 `mps:`，默认为 `off`）：

* `off`：绝不触碰 MPS。
* `auto`：在每一个承载了本流水线两个或更多单 GPU、非 TP CUDA 进程（包括进程副本）的 GPU 上启用。
* `on`：只需一个符合条件的进程即可启用，并且在不支持 MPS 的平台上会直接报错而非仅给出警告。

```bash
sgl-omni serve --config <config.yaml> --mps auto --port 8091
```

守护进程按物理 GPU 共享，以设备 UUID 为键。完整的生命周期、验证和运维说明请参阅 [使用 CUDA MPS 的同 GPU 数据并行](mps_dp.md)。

## CUDA IPC 权重共享

权重共享消除重复的权重显存占用；它不改变调度。在一个共享组内，副本索引最低的是 leader：它加载 checkpoint 并发布 CUDA-IPC 句柄。Follower 用 dummy 权重构建相同的模块树，并通过赋值以别名方式引用 leader 的不可变参数和缓冲区。KV cache、CUDA Graph、采样器状态、请求状态，以及架构的共享策略标记为副本私有的任何张量，都保留在各自副本中。

用它来让目标副本数量放得下，或者为 KV、CUDA Graph 和更多副本腾出显存。它是一种容量机制，而不是吞吐量优化。

```bash
sgl-omni serve --config <config.yaml> --mps on --weight-share on --port 8091
```

`weight_share` 的取值为 `off` 或 `on`（默认 `off`）。当为 `on` 时，其副本重复使用同一 GPU id 的每个逻辑 Process 都会成为该 GPU 上的一个共享组，运行时会自行分配角色；环境中不得设置 `SGLANG_OMNI_WEIGHT_SHARE`。独自占用其 GPU 的同一 Process 副本仍会加载自己的权重，并正常服务请求。

以下要求全部在任何进程生成之前检查：

* 共享的 Process 恰好包含一个 SGLang 引擎阶段，且 `tp=pp=1`；
* 该阶段固定 `max_total_tokens`，因为 follower 在其 dummy 权重释放之后才接入，内存分析无法推导出稳定的 KV 预算；
* 该架构在共享策略允许列表中，引擎会在 leader 加载时强制检查。

由于共享建立在进程副本之上，只有当模型的 GPU 阶段工厂接受 `gpu_id`（如"放置"一节所述）时，该模型才能使用共享。工厂不支持 `gpu_id` 的允许列表模型，仍可通过 `examples/mps_dp/launch.sh` 的多服务启动方案共享权重。

Follower 只会在每个 leader 就绪之后才生成，并会在其 leader 之前关闭。共享是一个整组生命周期：leader 必须比其 follower 存活更久；共享处于活动状态时拒绝在线权重更新；leader 死亡会使流水线失败，而不是继续基于别名内存提供服务。请将流水线整体重启。

权重共享不需要 MPS，MPS 也不需要权重共享。`--mps on --weight-share on` 是常见的同 GPU 数据并行（DP）组合：MPS 为副本提供 kernel 重叠能力，权重共享为它们腾出容纳空间。

## 组合运用

按以下顺序配置：

1. 决定哪些阶段共享一个进程；
2. 选择副本数量；
3. 把每个副本放置到一个 GPU 上；
4. 当多个进程共享一个 GPU 时启用 MPS；
5. 当重复权重成为容量上限时启用权重共享。

### 跨 GPU 的副本

```yaml
config_cls: Qwen3OmniSpeechPipelineConfig
name: qwen3-omni-speech-replica2
model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct

stages:
  talker_ar:
    gpu_memory_fraction: 0.123
  code2wav:
    gpu_memory_fraction: 0.014

processes:
  talker_ar:
    num_replicas: 2
    replica_devices: [1, 2]
  code2wav:
    num_replicas: 2
    replica_devices: [1, 2]
```

解析后的布局：

```
GPU 0: image_encoder + audio_encoder + thinker
GPU 1: talker_ar@r0 + code2wav@r0
GPU 2: talker_ar@r1 + code2wav@r1
```

使用 `sgl-omni serve --config examples/configs/qwen3_omni_speech_replica2.yaml --port 8091` 运行它；完整文件是 [`qwen3_omni_speech_replica2.yaml`](https://github.com/sgl-project/sglang-omni/blob/main/examples/configs/qwen3_omni_speech_replica2.yaml)。

### 单 GPU 上的副本

重复设备 id，并为该卡上的每个 GPU 阶段声明内存预算。不加 `--mps` 时，副本会对 GPU 分时复用。这里使用 MOSS TTS local，是因为它的引擎工厂接受 `gpu_id`，而且它的架构在权重共享允许列表中，因此同一个文件也可以用于下一个示例：

```yaml
config_cls: MossTTSLocalPipelineConfig
name: mossl
model_path: OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5

stages:
  preprocessing:
    gpu: 0
    gpu_memory_fraction: 0.05
  tts_engine:
    gpu: 0
    gpu_memory_fraction: 0.35
    engine:
      mem_fraction_static: 0.30
      max_total_tokens: 30000
  vocoder:
    gpu: 0
    gpu_memory_fraction: 0.15

processes:
  pipeline:
    num_replicas: 2
    replica_devices: [0, 0]
```

### 单 GPU 上带 MPS 和权重共享的副本

同样的配置，再加上两个运行时标志。上面的 `max_total_tokens` 正是权重共享对引擎阶段的要求：

```bash
sgl-omni serve --config <config.yaml> --mps on --weight-share on --port 8091
```

leader 持有共享权重；follower 通过 CUDA IPC 接入，只携带自己的 KV、CUDA Graph 和请求状态。

## 性能与正确性

副本主要在排队和较高并发的场景下有帮助。在低并发下它们可能略慢，因为并行化收益覆盖不了路由和进程开销。MPS 只有在共置进程存在可重叠的 GPU 工作时才有帮助。权重共享节省内存，其本身并不会改善调度。

验证一项拓扑变更时需要检查：串行输出一致性、并发路由与请求隔离、中止与恢复、进程/端口/GPU 内存的干净退出，以及（在启用时）MPS 挂接和权重共享生命周期。

## 迁移

已移除的接口及其替代方案见 [进程拓扑迁移](process_topology_migration.md)：用 `StageConfig.process` 表达进程成员关系，用顶层 `processes` 块表达副本数量和放置。
