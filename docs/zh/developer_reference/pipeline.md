## 流水线总览

### Coordinator

`Coordinator` 是全局请求路由器。它注册阶段端点，把新请求发送到入口阶段，接收 `CompleteMessage` 与 `StreamMessage` 事件，并解析（resolve）客户端的 future 或流。

关键职责：

- 将新请求路由到 `entry_stage`
- 跟踪请求状态：pending、running、completed、failed、aborted
- 收集终态阶段的完成事件
- 当流水线有多个终态阶段（如 `decode` 和 `code2wav`）时合并结果
- 向所有阶段广播中止消息

Coordinator 与阶段的具体实现无关。在张量并行阶段组中，它只与 rank 0 通信。同组的其余 rank 保持在阶段组内部。

### Stage

`Stage` 是一个 IO 外壳。它处理所有阶段间通信：接收控制消息，读写 relay 负载，在需要时执行扇入（fan-in），并把所有可执行的工作压入 `scheduler.inbox`。

```python
class Stage:
    def __init__(
        self,
        name,
        control_plane,
        relay,
        get_next,
        input_handler,
        scheduler,
        stream_targets,
        same_gpu_targets,
    ):
        self.scheduler = scheduler
```

Stage 的职责：

- 通过 ZMQ 接收 `SubmitMessage`、`DataReadyMessage`、`ShutdownMessage` 以及 profiler 控制消息
- 通过 coordinator 广播通道接收 `AbortMessage`
- 通过 relay 读写完整的 `StagePayload` 对象
- 用 `AggregatedInput` 为扇入阶段聚合输入
- 将普通结果路由到下游阶段或 coordinator
- 路由流式块，包括同 GPU 的 CUDA IPC 与跨 GPU 的 relay
- 排空 `scheduler.outbox` 并把调度器输出转换为控制面消息

重要的不变式是：`Stage` 不会按调度器类型分支。`SimpleScheduler`、`OmniScheduler` 与各种流式调度器对外呈现的接口完全一致。

### Scheduler

所有调度器实现同一接口：

```python
class Scheduler:
    inbox: Queue[IncomingMessage]
    outbox: Queue[OutgoingMessage]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def abort(self, request_id: str) -> None: ...
```

调度器消息用于与阶段层通信：

```python
class IncomingMessage:
    request_id: str
    type: Literal["new_request", "stream_chunk", "stream_done"]
    data: Any

class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error"]
    data: Any
    target: str | None
    metadata: dict[str, Any] | None
```

#### OmniScheduler

`OmniScheduler` 用于自回归阶段。它是对 SGLang 上游调度器的组合。目标是在复用 SGLang 的 batch 选择、KV cache 管理、prefill/decode 调度以及树缓存的同时，把 SGLang-Omni 自己的传输、请求对象与流式行为保留在上游调度器之外。（明确不支持 overlap scheduling：`OmniScheduler._event_loop_overlap` 拒绝运行，因为在该循环上 `Req.inflight_middle_chunks` 的递减会滞后一次迭代。）

#### SimpleScheduler

`SimpleScheduler` 面向非 AR 阶段，例如预处理、编码器、聚合与解码。它没有 KV cache，也没有 SGLang 批处理。其循环为：

```text
inbox.get() -> compute function -> outbox.put(result or error)
```

对于本地批处理有价值的阶段，它支持批量计算函数。

#### Code2WavScheduler

`Code2WavScheduler` 是流式声码器（vocoder）调度器。它处理：

- `new_request`：初始化按请求的状态
- `stream_chunk`：累积并解码 code 块
- `stream_done`：冲刷剩余音频并发送最终结果

### Model Runner

model runner 层掌管 AR 前向路径。设计目标是：

```text
ForwardBatch -> before/custom forward hooks -> model forward -> post hook -> output processing
```

共享的基类 runner 掌管通用机制：`ForwardBatch` 构建、采样、logit 处理、重复惩罚处理、输出处理，以及向调度器输出的转换。

#### ThinkerModelRunner

`ThinkerModelRunner` 面向 Qwen-omni thinker 风格的 AR 模型。它的模型专属工作是在模型前向之前，通过注入多模态 embedding（如图像、视频、音频与 deepstack 输入）来准备前向 batch。

#### FeedbackARModelRunner

重构设计为这类 AR 模型划出了一个共享的 `FeedbackARModelRunner` 角色：它们的下一个解码步依赖于同一个 model runner 内上一步产生的反馈。Qwen3-Omni talker 与 Fish Audio S2-Pro 都符合这一形态；Qwen3 目前在其 talker runner 中实现了该模式。

该抽象只覆盖自包含的反馈回路：

- 前向之前把上一步的反馈写入模型缓冲区
- 在模型 `forward()` 内运行 AR 骨干与次级头
- 前向之后抽取 codebook 输出与反馈张量
- 将流式或结果输出压入调度器发件箱

跨阶段反馈——生产者与消费者位于不同调度器、通过 relay 通信——不在该 runner 的范围内。

该设计把模型专属的反馈行为收拢为一个很小的策略：

```python
class FeedbackStrategy:
    def write_buffers(self, model, schedule_batch, requests) -> None: ...
    def extract_output(self, model, schedule_batch, requests, outbox) -> None: ...
    def prefill_forward(self, tp_worker, forward_batch, ...) -> object | None: ...
```
