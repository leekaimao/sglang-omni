# 请求级性能分析器

`sglang-omni` 提供两个互补的性能分析器，它们共享同一个 `run_id`，并由同一个 HTTP 面控制：

- 一个**请求级事件记录器**，写入由逐请求里程碑（准入、预处理、编码器、prefill、首个 token / 首个代码块、hop、终止响应）组成的 JSONL 流——用于重建单个请求的端到端时间线，并在一个批次内聚合 stage / hop 成本；
- 一个 **torch profiler**，产生内核级 CPU / CUDA 活动的 Chrome trace——用于在事件记录器找出时间去向之后，深入某个具体窗口。

大多数诊断使用事件记录器。torch profiler 需要显式开启（opt-in），用于更深入的内核调查。

## 事件模型

每个插桩点向一个逐进程的 JSONL 文件追加一行 JSON。其形状如下：

```jsonc
{
  "request_id": "req-123",
  "stage": "thinker",
  "event_name": "scheduler_first_emit",
  "timestamp_ns": 1717000000123456789,
  "run_id": "demo-run",
  "pid": 42,
  "metadata": {"chunk_id": 0}
}
```

文件写在 `<event_dir>/events_<stage>_<pid>.jsonl` 之下。同一 OS 进程内多个共置的 stage 共享**一个** JSONL 文件——文件名使用第一个启动的 stage，而逐事件的 `stage` 字段标识其归属。视图层按 `request_id` 合并来自每个进程的文件。

### 标准事件名

记录器总是把当前 `stage` 名附加到每个事件上，因此同一个 `scheduler_prefill_start` 从 thinker 进程发出时是 "thinker prefill start"，从 talker 进程发出时是 "talker prefill start"。`scheduler_queue_enter` 标记一个已构建的请求进入调度器队列；`scheduler_prefill_start` 稍后发出，此时该请求的第一个可执行 prefill / extend 批次被选中。

| 流水线里程碑 | 具体事件 | 来源 |
|---|---|---|
| 请求准入 | `request_admission` | `Coordinator._submit_request` |
| 预处理开始 / 结束 | `preprocess_start` / `preprocess_end` | 模型预处理器 `__call__` |
| 编码器开始 / 结束 | `encoder_start` / `encoder_end`（metadata `modality`、`batch_size`） | 图像 / 音频编码器执行器 |
| 聚合就绪 | `stage_aggregate_ready` | `InputHandler.receive` 返回合并后的负载之后的 `Stage._on_data_ready` |
| Thinker prefill 开始 | `scheduler_prefill_start`（stage = thinker） | `OmniScheduler.run_batch` |
| Thinker 首个 token | `stage_first_stream_chunk_sent`（stage = thinker） | `Stage._send_stream_to_target` / `_send_stream_to_coordinator` |
| 发往客户端的首个流式块 | `stage_first_stream_chunk_sent`（terminal stage → coordinator） | 同上 |
| Talker 请求构建执行开始 / 结束 | `scheduler_request_build_start` / `_end`（stage = talker） | `OmniScheduler._run_request_builder` |
| Talker prefill 开始 | `scheduler_prefill_start`（stage = talker） | 同上 |
| 首个代码块 | `stage_first_stream_chunk_sent`（stage = talker） | `Stage._send_stream_to_target` |
| Code2Wav 首段音频 | `code2wav_first_audio` | `Code2WavScheduler.decode_delta` / `_flush_pending` / `run_step` |
| 终止响应 | `terminal_response` | `Coordinator._handle_completion` |

用于更细粒度分解的辅助事件：

| 层 | 事件 | 说明 |
|---|---|---|
| Coordinator | `coordinator_stream_received` | coordinator 上收到的每个 `StreamMessage` |
| Stage | `stage_input_received` | 被接受的提交或中继负载（metadata `from_stage`） |
| Stage | `stage_dispatch` | 调度器收件箱入队 |
| Stage | `stage_complete` | 调度器结果被路由到后续（metadata `terminal`、`next`） |
| Stage | `stage_hop_sent` | 发往下一个 stage 的负载 `DataReadyMessage` |
| Stage | `stage_stream_chunk_sent` | 每个流式块（metadata `to_stage`、`chunk_id`、`modality`） |
| Stage | `stage_stream_chunk_received` | 每个被物化并可供接收方调度器使用的流式块，包括 coordinator 的终止块 |
| AR 调度器 | `scheduler_queue_enter` | 已构建请求进入调度器队列 |
| AR 调度器 | `scheduler_first_emit` | 每个请求的首次 `stream_output_builder` 发射 |
| Code2Wav | `code2wav_decode_start` | 串行解码开始：触发原因、start/end/new/context/window 帧数、活跃与达到阈值的请求数、收件箱深度 |
| Code2Wav | `code2wav_decode_launched` | 声码器（vocoder）工作与异步 D2H 拷贝均已入队的流水线化串行窗口；包含执行模式与 window/new 帧计数 |
| Code2Wav | `code2wav_decode_end` | 重复 start 的 metadata 并加上当前解码的 `audio_samples` 与执行 metadata；output-overlap 运行还包含 `pipelined` 以及上一窗口 EOS 后扫描的 `d2h_wait_ns` |
| Code2Wav | `code2wav_batch_start` | 合并步骤开始：批次与桶形状、new/window 帧数、活跃请求、收件箱深度、最久等待、触发原因、到期桶数量，以及子批次分解 |
| Code2Wav | `code2wav_batch_end` | 重复 start 的 metadata 并加上音频样本数、执行模式、graph key 与回退原因 |

要狭义地解读 `d2h_wait_ns`。在输出重叠开启时，惰性编解码器（codec）EOS 扫描会对已暂存的帧头调用 `.tolist()`，这本身就是一个主机同步点，而且它发生在 `code2wav_decode_start` 之前。因此 `d2h_wait_ns` 只测量该扫描之后残余的等待，而不是整个重叠区间。要判断阻塞同步是否真的被移除，请以全进程的 CUDA API trace 作为承重测量。

自定义调用点可以调用 `sglang_omni.profiler.event_recorder.emit(...)` 来添加领域特有的事件。来自未激活记录器的事件是空操作，因此插桩点不需要针对禁用情况做保护。

### 活动 stage 归因

`emit(...)` 接受显式的 `stage=...` 参数；当调用方无法把 stage 名一路传下去时（预处理器 `__call__`、编码器可调用对象、`OmniScheduler` / `Code2WavScheduler` 内部），它可以传 `stage=None`，由记录器从**逐线程 / 逐任务的活动 stage** 补齐。

`Stage._run_scheduler` 在调用调度器之前，先在调度器线程上绑定 `set_active_stage(self.name)`。该绑定同时使用一个 `threading.local` 槽位（面向普通 `threading.Thread` 工作线程）和一个 `contextvars.ContextVar`（因此它能穿越 `asyncio.to_thread` / `loop.run_in_executor` 传播，这两者复制 contextvars 但不复制 thread-local）。emit 上显式的 `stage=...` 永远优先；只有当调用方传 `stage=None` 时才查询活动 stage 绑定。

要在你自己的线程中手动绑定 / 解绑：

```python
from sglang_omni.profiler.event_recorder import set_active_stage, reset_active_stage

token = set_active_stage("my_stage")
try:
    ...
finally:
    reset_active_stage(token)
```

`reset_active_stage(None)` 是"擦除"形式（由测试夹具使用），会同时清除 thread-local 槽位和 contextvar。

## 生命周期

记录器是进程局部的。当 `POST /start_profile`（或 `POST /start_request_profile`）被命中时，它在每个 stage 和 coordinator 上启动：

1. 启动器接收该 HTTP 请求。
2. coordinator 启动其指向 `<event_dir>` 的本地记录器。
3. 启动器通过 ZMQ 向每个 stage 广播 `ProfilerStartMessage`，同时携带 torch trace 模板和 `event_dir`。
4. 每个 stage 加入逐进程记录器。在共享进程拓扑中，第一个调用 `start()` 的 stage 赢得文件名；同一进程中后续的每个 stage 都写同一个文件，由逐事件的 `stage` 字段消歧。
5. 在 `POST /stop_profile` 时，记录器在所有地方被关闭；文件保留在磁盘上的 `<event_dir>` 之下。

`POST /stop_profile` 和 `POST /stop_request_profile` 接受可选的 `run_id` 字段。**省略**时，该请求是通配的：每个 stage 停止当前活跃的任何性能分析会话。**设置**时，只有活跃 run 匹配的 stage 停止。这让常见情形（调用方在启动或停止时都没有指定 run_id）无需额外仪式即可工作。

torch profiler 与事件记录器共享同一个 `run_id`。在启动请求上设置 `enable_torch=false` 可以记录 JSONL 事件而不付出内核 trace 的代价。

## 生成报告

直接使用 views 模块：

```python
from sglang_omni.profiler.views import build_report
report = build_report("/tmp/profiles/demo-run/events")
print(report["request_count"], len(report["stage_breakdown"]))
```

……或者通过 CLI：

```bash
python -m sglang_omni.profiler /tmp/profiles/demo-run/events --format table
python -m sglang_omni.profiler /tmp/profiles/demo-run/events --format json --out report.json
```

CLI / `build_report` 返回从同一事件流派生的三个视图：

1. **Timeline（时间线）** —— 逐请求事件列表，`t_rel_ms` 以准入为锚点。
2. **Stage breakdown（stage 分解）** —— 按 stage 聚合的 `(open_event, close_event)` 区间时长（count、total、avg、p50、p95、max）。同一个开启事件可以参与多对配对（例如 `scheduler_prefill_start` 既对着 `scheduler_first_emit` 关闭，也对着 `stage_first_stream_chunk_sent` 关闭）；每对配对有各自的待处理栈，因此 A 对的关闭事件不会消耗 B 对的开启事件。
3. **Hop breakdown（hop 分解）** —— 按 (source, destination, kind) 统计的 `stage_hop_sent` / `stage_input_received` 与 `stage_stream_chunk_sent` / `stage_stream_chunk_received` 时长。终止 stage 的流式块以相同方式与目的地 `coordinator` 配对。

hop 配对按 `(request_id, source_stage, dest_stage, chunk_id?)` 跨进程匹配，因此即使每个 stage 运行在各自的进程中，也能重建单个请求穿越子进程的路径。

## Torch profiler

当 `enable_torch=true`（`/start_profile` 的默认值）时，torch profiler 与事件记录器一同运行。它在 `start()` 与 `stop()` 之间持续记录——没有 `schedule(...)`，也没有 `step()` 要求——并在停止时导出 Chrome trace `*.trace.json.gz`。

昂贵的自省标志通过环境变量按需开启，使默认 trace 保持足够小，能够加载进 `chrome://tracing` 或 [`ui.perfetto.dev`](https://ui.perfetto.dev)：

| 环境变量 | 效果 |
|---|---|
| `SGLANG_TORCH_PROFILER_RECORD_SHAPES=1` | 记录每个算子的输入张量（tensor）形状 |
| `SGLANG_TORCH_PROFILER_PROFILE_MEMORY=1` | 跟踪 CUDA 缓存分配器的每次分配 / 释放 |
| `SGLANG_TORCH_PROFILER_WITH_STACK=1` | 记录每个算子的 Python（以及 C++）调用栈 |
| `SGLANG_TORCH_PROFILER_WITH_FLOPS=1` | 估算每个算子的 FLOPs |

四个全部关闭（默认）时，一次典型的 10 样本 MMMU 运行产生的 trace 在几十 MB 量级。四个全部开启时，同一负载可能产生多 GB 的 trace——只在需要那类特定信息时才开启。

## HTTP 接口

| 方法 | 路径 | 请求体 | 说明 |
|---|---|---|---|
| POST | `/start_profile` | `{"run_id": ?, "trace_path_template": ?, "event_dir": ?, "enable_torch": true \| false, "config": ?}` | 启动 torch trace + 事件记录器。省略 `run_id` 时自动生成。 |
| POST | `/stop_profile` | `{"run_id": ?}` | 停止 torch trace + 事件记录器。省略 `run_id` 即为通配（"停止当前活跃的任何会话"）。 |
| POST | `/start_request_profile` | `{"run_id": ?, "event_dir": ?}` | 仅事件记录器——没有 torch trace。开销更低；更适合保持开启。 |
| POST | `/stop_request_profile` | `{"run_id": ?}` | 与 `/stop_profile` 相同的通配语义。 |

示例：在不产生内核 trace 的情况下为每个请求记录廉价事件：

```bash
curl -X POST http://localhost:8000/start_request_profile \
     -d '{"run_id":"demo","event_dir":"/tmp/profiles/demo/events"}'
# … run traffic …
curl -X POST http://localhost:8000/stop_request_profile -d '{}'
python -m sglang_omni.profiler /tmp/profiles/demo/events --format table
```

## 纪律

- **性能分析绝不能破坏服务。** 发射器吞掉写入错误并统计丢弃数；第一次失败只记录一次日志。
- **张量和大块数据不得进入事件 metadata。** metadata 只保留小标量（id、计数、时长、modality、错误字符串）。记录器对此做了防御性强制：如果张量 / numpy 数组最终出现在 metadata 中，`_json_default` 会序列化一个摘要（`{"__tensor_summary__": true, "type": ..., "shape": [...], "dtype": "...", "device": "..."}`）而不是物化其内容。0 维张量 / numpy 标量仍序列化为普通标量。
- **事件命名。** 小写 snake_case，以拥有该事件的层为前缀（`stage_*`、`scheduler_*`、`encoder_*` 等）。用 stage 名（而不是事件名）区分 "thinker prefill start" 与 "talker prefill start"。
