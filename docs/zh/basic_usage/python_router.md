# Omni Router 使用方法

SGLang-Omni Router 是面向 Omni V1 部署的外部 HTTP 路由器。它前端代理多个完整的
Omni V1 API 服务器，并向客户端暴露一个 OpenAI 兼容端点。

当你启动多个 `sgl-omni serve` 进程、并希望用一个稳定端点完成请求分发、健康
跟踪与 worker 池控制时，请使用该 router。

## Router 拓扑

router 是一个外部 HTTP 进程：

```text
client
  |
  v
sgl-omni-router-py
  |
  +-- sgl-omni serve worker A
  +-- sgl-omni serve worker B
```

每个 worker 都是一个完整的 Omni V1 HTTP 服务器。router 不加载模型权重，也不会
把单个请求拆分到多个 worker。它为每个请求选择一个可路由的 worker，转发原始
请求字节，并返回带 router 诊断头的 worker 响应。

## 从 YAML 启动 worker 与 Router

对本地同构池，`sgl-omni-router-py` 可以先启动 worker 副本，并在所有受管
worker 通过 `/health` 之后再启动 router：

```bash
sgl-omni-router-py \
  --host 0.0.0.0 \
  --port 8008 \
  --launcher-config examples/configs/qwen3_omni_router.yaml \
  --policy round_robin \
  --health-failure-threshold 2 \
  --health-success-threshold 1 \
  --health-check-interval-secs 10 \
  --log-level info
```

launcher 配置示例：

```yaml
launcher:
  backend: local
  model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
  model_name: qwen3-omni
  num_workers: 2
  num_gpus_per_worker: 1
  worker_host: 127.0.0.1
  worker_base_port: 8011
  worker_extra_args: "--config examples/configs/qwen3_omni_colocated_h20.yaml --colocate"
  wait_timeout: 600
```

`backend: local` 表示 router 进程在同一台机器上启动并管理 worker 子进程。被
启动的 worker 是用 `sgl-omni serve` 启动的完整 Omni V1 服务器，不是部分的
流水线阶段。router 会等待每个受管 worker 通过 `/health` 才开始接收客户端
流量，并在 router 退出时停止这些受管 worker。

`num_gpus_per_worker` 控制自动 GPU 分组。默认的 Qwen3-Omni router 示例使用
共置 worker：每个完整的语音 worker 通过
`examples/configs/qwen3_omni_colocated_h20.yaml` 运行在一块 GPU 上。在
`num_workers: 2` 且 `num_gpus_per_worker: 1` 时，当两块 CUDA 设备可见，
launcher 会把 GPU `0` 分配给第一个 worker，GPU `1` 分配给第二个 worker。

单 H200 worker 请改用 `examples/configs/qwen3_omni_colocated_h200.yaml`。

只有需要显式放置时才设置 `worker_gpu_ids`。每个条目把一个
`CUDA_VISIBLE_DEVICES` 值映射到一个 worker，例如两个单 GPU 共置 Qwen3-Omni
worker 使用 `worker_gpu_ids: ["0", "1"]`。只有在有意要文本输出 worker 而不是
语音输出 worker 时，才使用 `worker_extra_args: "--text-only"`。

worker 进程专属的公开 Omni V1 serve 选项（如 `--mem-fraction-static`、
`--thinker.tp_size` 或 `--text-only`）请通过 `worker_extra_args` 传入。这些
参数会在 launcher 自有的标志之后传给 `sgl-omni serve`。未提供内存标志时，
Omni V1 使用其正常的自动定容路径。

当受管 worker 有意只暴露 Omni API 表面的一部分时，请使用
`worker_capabilities`。例如，text-only worker 不应声明语音或音频输出支持：

```yaml
launcher:
  backend: local
  model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
  model_name: qwen3-omni
  num_workers: 2
  num_gpus_per_worker: 1
  worker_extra_args: "--text-only"
  worker_capabilities:
    - chat
    - streaming
    - image_input
    - audio_input
    - video_input
```

如果省略 `worker_capabilities` 而 `worker_extra_args` 含有 `--text-only`，
router 会以上面同样的 text-only 能力集注册受管 worker。

对短的音频输入/文本输出 MMSU 风格工作负载，请用融合文本路径的 Qwen3-Omni
配置代替默认的语音共置 worker：

```yaml
launcher:
  backend: local
  model_path: Qwen/Qwen3-Omni-30B-A3B-Instruct
  model_name: qwen3-omni
  num_workers: 2
  num_gpus_per_worker: 1
  worker_extra_args: "--config examples/configs/qwen3_omni_mmsu.yaml --text-only"
```

这让预处理、编码器、聚合、thinker 与 decode 留在一个 worker 进程内，同时
不改变通用的语音共置拓扑。

## 手动启动 worker 服务器

分别启动每个 Omni V1 worker。下面的示例在不同 GPU 与端口上启动两个共置的
Qwen3-Omni 语音 worker：

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --host 0.0.0.0 \
  --port 8011
```

```bash
CUDA_VISIBLE_DEVICES=1 sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --host 0.0.0.0 \
  --port 8012
```

传给 router 的 worker URL 必须是形如 `http://127.0.0.1:8011` 的 base URL。
不要包含端点路径、查询字符串或 fragment。

## 启动 Router

带上 worker URL 启动 router：

```bash
sgl-omni-router-py \
  --host 0.0.0.0 \
  --port 8008 \
  --worker-urls http://127.0.0.1:8011 http://127.0.0.1:8012 \
  --policy round_robin \
  --health-failure-threshold 2 \
  --health-success-threshold 1 \
  --health-check-interval-secs 10 \
  --log-level info
```

## Router 参数

下表列出 router 的命令行参数。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | router HTTP 服务器的主机接口。 |
| `--port` | `8000` | router HTTP 服务器的端口。 |
| `--worker-urls` | 未设置 | 空格分隔的 Omni V1 worker base URL，构成同构 worker 池。 |
| `--worker-config` | 未设置 | 定义 worker 及可选的每 worker 模型/能力元数据的 JSON 文件。 |
| `--launcher-config` | 未设置 | 用于受管本地 worker 池的 YAML 文件。不要与 `--worker-urls` 或 `--worker-config` 同时使用。 |
| `--policy` | `round_robin` | 路由策略：`round_robin`、`least_request` 或 `random`。 |
| `--model` | 未设置 | 使用 `--worker-urls` 时分配给每个 worker 的模型名。不要与 `--worker-config` 同时使用。 |
| `--request-timeout-secs` | `1800` | 代理 worker 请求的超时时间。 |
| `--max-payload-size` | `536870912` | router 接受的最大请求体大小，单位为字节。 |
| `--max-connections` | 自动：`128 x workers`，上限 `4096` | 准入界限：router 以 `503` 快速拒绝前允许的最大并发在途模型请求数。上游连接池至少按该值定容。显式值低于 `64 x workers` 时会记录欠喂警告。 |
| `--max-inflight` | 等于 `--max-connections` | 高级覆盖项，把准入界限与 `--max-connections` 解耦。上游池按两者中较大的定容。 |
| `--health-failure-threshold` | `3` | 连续健康检查失败或被路由请求失败达到该次数后，worker 变为不健康。 |
| `--health-success-threshold` | `2` | 不健康或未知 worker 连续健康检查成功达到该次数后变为健康。 |
| `--health-check-timeout-secs` | `5` | 单次 worker 健康检查请求的超时。 |
| `--health-check-interval-secs` | `10` | 后台 worker 健康检查的间隔。 |
| `--health-check-endpoint` | `/health` | 后台健康检查使用的 worker 端点。 |
| `--voice-owner-worker-url` | 第一个同时具备 `speech` 与 `audio_input` 的 worker | 仅单进程模式。拥有已上传 TTS voice 的 worker。使用已上传 voice 的 voice 管理与合成请求都留在该 worker 上。没有这样的 owner 时，内置 voice 仍可路由到合格的语音 worker，但 voice 管理不可用。 |
| `--router-state-dir` | `$SGLANG_OMNI_ROUTER_STATE_DIR`，否则 `$XDG_STATE_HOME/sglang-omni-router`，否则 `~/.local/state/sglang-omni-router` | 存放[持久 Router 状态](durable-router-state)（权重更新 journal）的目录，必须在主机重启后仍然存在。容器中请挂载到持久卷。 |
| `--log-level` | `info` | router 与 Uvicorn 的日志级别。 |
| `--strict-limits` | off | 当 `nofile` 软限制低于解析出的上游池大小（`max(--max-connections, --max-inflight)`）时，启动失败而不是警告。 |
| `--router-processes` | `1` | 数据面转发进程数。`1` 保持下述单进程 router；`N >= 2` 启用[多进程 Router](multi-process-router-controldata-plane-split)（仅限 x86-64 Linux，且与显式 `--policy least_request` 一起在启动时被拒绝）。 |
| `--shutdown-drain-secs` | `--request-timeout-secs` | 仅多进程：正在停止的数据面进程在取消剩余请求前等待在途请求的时间。默认值与请求超时一致，因此常规关停绝不会截断一个本可被自身超时放行的请求。 |

路由策略：

- `round_robin`：按顺序在可路由 worker 之间轮转。
- `least_request`：选择活跃数据面请求最少的可路由 worker，打平时轮转。
- `random`：随机选择一个可路由 worker。

`--launcher-config`、`--worker-urls`、`--worker-config` 三者恰好传入一个。
当 worker 服务不同模型、或只暴露 Omni 能力的一个子集时，使用
`--worker-config`：

```json
{
  "workers": [
    {
      "url": "http://127.0.0.1:8011",
      "model": "qwen3-omni",
      "capabilities": ["chat", "image_input", "video_input"]
    },
    {
      "url": "http://127.0.0.1:8012",
      "model": "qwen3-omni",
      "capabilities": ["chat", "audio_input", "audio_output", "speech"]
    }
  ]
}
```

然后这样启动：

```bash
sgl-omni-router-py \
  --host 0.0.0.0 \
  --port 8008 \
  --worker-config workers.json \
  --policy least_request
```

(check-router-and-worker-state)=
## 检查 Router 与 worker 状态

router 暴露相互独立的进程与 worker 池健康面：

```bash
curl -i http://127.0.0.1:8008/live
curl -i http://127.0.0.1:8008/ready
curl -i http://127.0.0.1:8008/health
curl -s http://127.0.0.1:8008/workers
curl -s http://127.0.0.1:8008/v1/models
```

各端点含义不同：

- `GET /live`：router 进程正在运行。不会等待 worker 变为健康。
- `GET /ready`：至少一个 worker 可路由。当所有 worker 不健康、已死、被禁用或
  仍未知时返回 `503`。
- `GET /health`：worker 池健康摘要，外加准入统计（`inflight`、
  `max_inflight`、`peak_inflight`、`rejected_total`）。没有可路由 worker 时
  返回 `503`。
- `GET /workers`：详细 worker 状态，包括 `health_state`、`disabled`、
  `routable`、`active_requests`、聚合与按服务类的请求计数器以及最近错误。
- `GET /v1/models`：来自可路由 worker 的合并模型列表。

## 通过 Router 发送请求

让客户端指向 router 端口而不是 worker 端口。请求 schema 与每个 worker 服务器
使用的 OpenAI 兼容 schema 相同。

图像输入、文本输出：

```bash
curl -i http://127.0.0.1:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-request-id: router-image-1" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "How many cars are there in the image? Answer briefly."}
    ],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

流式文本：

```bash
curl -N http://127.0.0.1:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-request-id: router-stream-1" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say hello briefly."}],
    "stream": true,
    "max_tokens": 16
  }'
```

router 保留原始请求体。对普通 JSON 请求，它只解析有限数量的请求元数据用于
worker 选择，并把原始字节转发给选中的 worker。

## 管理 worker

运行时添加一个 worker：

```bash
curl -s http://127.0.0.1:8008/workers \
  -H "Content-Type: application/json" \
  -d '{"url":"http://127.0.0.1:8013","model":"qwen3-omni"}'
```

禁用一个 worker 而不删除它：

```bash
curl -s -X PUT http://127.0.0.1:8008/workers/http%3A%2F%2F127.0.0.1%3A8013 \
  -H "Content-Type: application/json" \
  -d '{"disabled":true}'
```

把一个 worker 标记为 dead 做手动隔离：

```bash
curl -s -X PUT http://127.0.0.1:8008/workers/http%3A%2F%2F127.0.0.1%3A8013 \
  -H "Content-Type: application/json" \
  -d '{"is_dead":true}'
```

恢复一个被手动标记为 dead 的 worker：

```bash
curl -s -X PUT http://127.0.0.1:8008/workers/http%3A%2F%2F127.0.0.1%3A8013 \
  -H "Content-Type: application/json" \
  -d '{"is_dead":false}'
```

删除一个 worker：

```bash
curl -s -X DELETE http://127.0.0.1:8008/workers/http%3A%2F%2F127.0.0.1%3A8013
```

worker 更新请求是原子的。如果更新返回 `400`，线上 worker 状态不会被部分
更改。

(routing-behavior)=
## 路由行为

router 只选择健康、未被禁用、且有能力服务该请求的 worker。

默认的 worker 能力集代表一个完整的 Omni V1 副本：

- `chat`
- `speech`
- `streaming`
- `image_input`
- `audio_input`
- `video_input`
- `audio_output`

router 从每个请求推断所需能力：

- `/v1/chat/completions` 要求 `chat`
- `stream: true` 要求 `streaming`
- `images`、`image` 或图像消息片段要求 `image_input`
- `audios`、`audio_inputs` 或音频消息片段要求 `audio_input`
- `videos`、`video` 或视频消息片段要求 `video_input`
- `modalities: ["audio"]` 或 `audio` 输出字段要求 `audio_output`
- `/v1/audio/speech` 与 `/v1/audio/speech/batch` 要求 `speech`；
  `/v1/audio/speech` 在 `stream: true` 时还要求 `streaming`（批量语音不支持
  流式）
- 使用 `ref_audio` 或携带音频的 `references` 的语音请求还要求 `audio_input`
- `/v1/audio/speech/stream` WebSocket 会话要求 `speech` 与 `streaming`，
  配置了参考音频时还要求 `audio_input`
- `/v1/audio/voices` 的管理以及使用已上传 voice 的合成要求 owner worker，
  它同时具备 `speech` 与 `audio_input`
- `/v1/audio/transcriptions` 与 `/v1/audio/translations` 要求 `audio_input`
  （`stream` 表单字段为 true 时也要求 `streaming`）。二者是 multipart 上传，
  因此 router 用一次跳过上传文件的线性扫描读取 `model` 与 `stream` 表单字段
  以避免 CPU 开销，这样在混合 ASR 模型的池中，请求会落到以该模型名注册的
  worker 上。当 router 无法读取某个字段时，回退到
  `X-SGLang-Omni-Route-Model` 与 `X-SGLang-Omni-Route-Stream`；当字段与其
  对应头同时存在时二者必须一致，否则 router 返回 `400`。翻译支持因模型而
  异，不支持翻译的 worker 会返回 `400`。

只有当 worker 确实无法服务上述某类请求时，才为其注册更窄的能力集。

下文描述的完整 TTS 路由集目前由单进程 router（`--router-processes 1`）支持。
多进程 router 数据面保留[多进程 Router](multi-process-router-controldata-plane-split)
中记录的路由表面。

已上传的 TTS voice 是可变的 worker 本地状态。Router 把 voice 的列出、上传与
删除操作发送到 `--voice-owner-worker-url`，并把使用已上传 voice 名称的语音、
批量与 WebSocket 请求固定到同一个 worker。该 owner 不可用时返回 `503`，而不
是把请求静默路由到没有该 voice 的 worker。一旦已上传 voice 的注册表可用，
内置 voice 仍按配置的路由策略负载均衡。默认情况下，第一个同时具备 `speech`
与 `audio_input` 能力的可路由 worker 成为 owner，并在该 router 进程生命周期内
保持 owner 身份。当 owner 需要在 router 重启之间保持稳定时，请配置
`--voice-owner-worker-url`。没有此类 owner 的池仍可把内置 TTS 路由到合格的
语音 worker，但 voice 管理返回 `503`。router 运行期间，无法通过其 worker API
移除 owner，也无法移除其任一必需能力。

Voice 归属假设池中只有一个 router 作为写入者、客户端通过该 router 执行 voice
变更、且 owner 在重启之间保留其 speaker 目录。多个独立 router 或对 worker 的
直接变更不做协调，对已上传 voice 不受支持。router 在 voice 流量开始后于后台
加载其已上传 voice 注册表，并在 voice 变更后对其做对账。在初次加载成功之前、
或某个变更结果悬而未决期间，每个非默认 voice 名称都被固定到 owner，因为
router 无法安全区分内置名称与已上传名称。`GET /health` 在 `voice_routing`
之下报告被选中的 owner、owner 可路由性、注册表状态与已上传 voice 数量。
router 通过紧凑的 `GET /v1/audio/voices?names_only=true` 响应为该注册表注水，
而不是传输存储的参考元数据。

大型 JSON 请求不会被 router 完整解析。对由完整 Omni V1 副本组成的同构池，
不需要额外的头。混合模型时，请提供模型提示。混合 worker 能力时，当 router
无法推断出单一安全的 worker 集合时，请提供能力提示：

- `X-SGLang-Omni-Route-Model`：混合模型池中请求的模型
- `X-SGLang-Omni-Route-Capabilities`：逗号分隔的能力，例如 `image_input`、
  `audio_input`、`video_input`、`audio_output` 或 `streaming`
- `X-SGLang-Omni-Route-Stream`：大型流式请求取 `true` 或 `false`

超过 1 MiB 的语音与语音批量 JSON 体会被保守地固定到 voice owner，因为
router 无法完整检查它们以排除已上传 voice 引用。没有合格的 owner 时，router
保守地要求 `audio_input`；它绝不在大小边界处削弱能力选择。route-hint 头不会
覆盖 voice 归属。

这些头是仅 router 可见的提示，不会转发给 worker。

## 请求诊断

被路由的 HTTP 响应包含：

- `X-SGLang-Omni-Worker`：被选中的 worker ID
- `X-SGLang-Omni-Request-ID`：来自请求头或请求体的请求 ID，或 router 生成的
  ID
- `X-SGLang-Omni-Route-Attempt`：目前为 `1`

HTTP 路由在 worker 选择之后发出 `route_completed`，在选择之前发出
`route_rejected`。完成记录包含请求 ID、被选中的 worker、路径、流式标志、推断
出的能力、状态码、时长与最终结果。

WebSocket 会话在升级之后无法添加 HTTP 响应头。Router 的拒绝会以一个 TTS
`error` 事件发送，后跟一个协议对应的关闭码。每个被接受的 WebSocket 都会发出
一条 `tts_websocket_completed` 记录，包括 worker 选择之前的拒绝；这些记录的
`worker=-`。

(overload-behavior)=
## 过载行为

router 约束其并发工作量。一旦有 `--max-connections` 个在途模型请求正在被
转发，额外的模型请求会在读取请求体之前立即被拒绝：

- HTTP 请求收到状态 `503`、OpenAI 风格的错误信封
  （`"type": "overloaded_error"`）、`Retry-After: 1` 头，以及一条
  `reason=router_overloaded` 的 `route_rejected` 日志
- WebSocket 请求收到一个 `error_type=overloaded_error` 的 TTS `error` 事件、
  关闭码 `1013`，以及一条 `outcome=router_overloaded` 的终态日志

健康与管理端点（`/live`、`/ready`、`/health`、`/workers`、管理路由）从不
受限。`GET /health` 报告当前在途水平、启动以来的峰值以及累计拒绝数。

容量建议：

- 自动默认值（`128 x workers`）是防止发散的兜底，不是延迟目标。对单核
  router 上的大响应，过大的界限会拖垮服务本身；请按你的负载形态把
  `--max-connections` 定在 `容量 x 可接受延迟` 附近。
- 每个在途请求占用两个文件描述符（客户端加上游）。当 `nofile` 软限制低于
  `2 x 上游池大小 + 余量` 时（池大小为
  `max(--max-connections, --max-inflight)`），router 在启动时发出警告。请
  提高该限制，或调低决定池大小的那个标志（警告会指名它）；`--strict-limits`
  把该警告变成启动错误。
- 被拒绝的请求会让客户端损失其 keep-alive 连接（router 在读取请求体之前就
  响应），因此客户端在收到 `503` 时应退避，而不是立即用新连接重试。

(failure-handling)=
## 故障处理

worker 存活由后台 `/health` 探测负责。只有当 router 无法从 worker 获得可用
响应时，被转发的请求才会把 worker 标记为不健康：传输层失败（连接错误或读
超时，没有 HTTP 响应），或 worker 返回的网关状态 `502 Bad Gateway` 或
`504 Gateway Timeout`。容量背压以及 worker 自身回答的应用状态——`429 Too
Many Requests`、`503 Service Unavailable`、`408 Request Timeout` 与
`500 Internal Server Error`——在 worker 统计中计为每请求失败，但绝不会驱逐
一个可达的 worker，因此一个过载 worker 或一连串坏输入请求不会把整个池级联
拖垮。离开池的 worker 在达到配置次数的成功健康检查之后可以恢复健康。

检查故障转移行为：

1. 停掉一个 worker。
2. 调用 `GET /workers` 并检查其 `consecutive_failures`、`health_state` 与
   `routable` 字段。
3. 再通过 router 发送一个请求，确认它使用剩余的可路由 worker。
4. 重启被停止的 worker 并等待它恢复健康。

对未安装 console script 的源码检出，用以下命令验证模块入口：

```bash
python -m sglang_omni_router.python.serve --help
```

(durable-router-state)=
## 持久 Router 状态

`--router-state-dir` 存放必须比 router 重启、甚至比 router 所在主机重启存活
更久的控制面状态。目前那就是权重更新 journal：当一次更新在部分 worker 已更新
后被中断，journal 会把这些受影响的 worker 保持禁用，直到运维者核验其权重
版本并重新启用它们（`PUT /workers/{worker_id} {"disabled": false}`）。
worker 运行在各自的主机上且比 router 存活更久，因此丢失 journal 会让一个
新启动的 router 重新启用一个权重已不匹配的池。

解析顺序：

1. `--router-state-dir`
2. `SGLANG_OMNI_ROUTER_STATE_DIR`
3. `$XDG_STATE_HOME/sglang-omni-router`
4. `~/.local/state/sglang-omni-router`

该目录以仅属主权限创建（`0700`，journal 文件 `0600`），并以 router 的
`host:port` 为键，因此同一台机器上的多个 router 保持各自的 journal。如果目录
无法创建或写入，失败方式取决于模式：`--router-processes >= 2` 时启动失败；
单进程默认仍会启动、记录警告，并拒绝权重更新（`503`），直到
`--router-state-dir` 指向一个可写的持久路径。两种模式都没有临时目录回退——
那会在重启静默丢弃记录的同时看起来持久。

如果 journal 本身变得不可读，重新启用无法解决该问题，每次权重更新都以
`409` 保持阻塞。请核验池的权重版本，然后显式丢弃该记录：

```bash
curl -X POST http://127.0.0.1:8000/weight_update_journal/resolve \
  -H "Authorization: Bearer $SGLANG_OMNI_ADMIN_KEY" \
  -d '{"acknowledge": true}'
```

该调用需要管理员认证，要求 `acknowledge` 以防记录被意外丢弃，会报告其丢弃的
worker id，并且在文件无法删除时以 `503` 失败。受影响的 worker 保持禁用，直到
被逐一重新启用。

在容器中，把它挂载到持久卷：

```bash
docker run -v /srv/sglang-omni-router:/var/lib/sglang-omni-router ... \
  python -m sglang_omni_router.python.serve \
    --router-state-dir /var/lib/sglang-omni-router \
    --worker-urls http://127.0.0.1:8011
```

该目录**不**管理的内容：

- 每次运行的运行时文件（序列化配置、内部 socket、worker 快照、准入共享
  内存）留在临时的每运行工作目录中。它们在启动时重建，本来就应随进程树一起
  消失。
- 日志。router 写入 stdout/stderr；请用你的容器运行时、systemd 或日志收集器
  持久化它们。

(multi-process-router-controldata-plane-split)=
## 多进程 Router（控制面/数据面分离）

> 仅限 x86-64 Linux：共享准入 seqlock 依赖 x86-64 的存储顺序，其他机器在
> 启动时被拒绝。用 `--router-processes N` 启用；默认 `1` 保持上文描述的
> 单进程 router。`N = 1` 时唯一的行为新增是权重更新 journal：启动时的恢复
> 可能会把早前一次被中断更新中的 worker 保持禁用，直到运维者重新启用它们。

在 `N >= 2` 时，router 以一棵小型进程树运行：

- 一个 **supervisor** 只绑定一次公共端口，并把监听 socket 传给 `N` 个
  **数据面（DP）**进程；DP 从共享队列 accept，并转发模型路由（`/generate`、
  `/v1/chat/completions`、`/v1/audio/speech`、`/v1/audio/transcriptions`、
  `/v1/audio/translations`）。
- 一个**控制面（CP）**拥有 worker 注册表、健康检查与管理表面。DP 通过 CP
  在每次状态变化时以及固定 keepalive 节奏上重新发布的快照文件获知可路由
  worker 集合。发往公共端口的管理请求被转发给 CP，由它强制管理密钥；
  `/v1/models`、`/live` 与 `/ready` 由 DP 自行回答，`/health` 是 CP 的聚合
  视图。
- supervisor 重启崩溃的子进程（对快速崩溃循环有失败关闭的预算），按代次隔离
  被替换的进程，并在 SIGTERM 时按顺序拆除进程树。正在停止的 DP 在剩余任务
  被取消之前，为在途请求排空至多 `--shutdown-drain-secs`（默认为
  `--request-timeout-secs`）；supervisor 会等完这段排空时间再升级到
  SIGKILL。

需要了解的行为差异：

- **准入界限是共享且软性的。**在途界限通过共享内存计数器数组覆盖所有 DP。
  并发的准入检查最多可能超出界限 `N - 1` 个请求。在 `/health` 中，
  `inflight` 是瞬时和（不是线性一致的值），`peak_inflight` 是尽力而为的。
  只要每个存活槽位仍可读，`rejected_total` 就是精确的；一个在写入中途被杀
  的 DP 会丢失它尚未发布的计数器，因此在崩溃与回收窗口期间该总数是尽力而
  为的。
- **`least_request` 要求单进程。**它读取每进程计数器，因此
  `--router-processes >= 2` 与显式 `--policy least_request` 组合会在启动时
  失败；请使用 `round_robin`（DP 会错开起始偏移以避免扎堆）或 `random`。
- **CP 不可用先降级、后失败。**DP 在过期超时内继续按其最后快照服务，之后用
  快速 `503` 摒弃新请求并翻转 `/ready`；一次重新发布即恢复服务。部分 DP
  缺失时 `/health` 报告 `degraded`，只有在完全无法服务时才返回 `503`。
- **worker 权重更新会等待数据面。**广播之前，CP 先发布禁用 worker 的快照并
  等到每个存活 DP 确认；超时时更新失败关闭，什么也不发送。
- **计数器是聚合的。**`/workers` 的聚合计数与 `*_requests_by_class` 映射跨
  DP 求和，并在 DP 重启之间保持单调；`active_requests` 是对存活 DP 尽力而
  为的 gauge。计数器随 CP 重置，与单进程重启一致。
- **文件描述符随 `N` 增长。**每个 DP 持有全尺寸的上游池（偏斜的 keep-alive
  客户端组合可能把整个界限钉在一个 DP 上，它必须扛得住），keep-alive 按
  `pool / N` 拆分；启动日志会为 nofile 检查打印每进程与集群的 fd 预算。

密集部署的 CPU 亲和性建议：把 `N` 个 DP 进程钉在一个 NUMA 节点的 `N` 个专用
核上；CP 与 supervisor 几乎空闲，可以共用一个核；把模型 worker 与基准客户端
放在其他核（或另一个 NUMA 节点）上，避免转发吞吐被争抢。

## 故障排查

重启任何进程之前，先检查 Router 与 worker 池——端点命令与含义见
[检查 Router 与 worker 状态](check-router-and-worker-state)。

用响应与健康信号决定下一步：

| 信号 | 含义 | 行动 |
|---|---|---|
| `/live` 失败 | Router 进程不可达。 | 检查进程、绑定地址、端口与启动日志。 |
| `/live` 为 `200`，`/ready` 为 `503` | 没有可路由的 worker。 | 在 `/workers` 中检查各 worker 状态。 |
| 模型请求：`413`，`message=payload too large` | 请求体超过 `--max-payload-size`。 | 缩小请求体积，或在评估内存与并发影响后调高配置上限。 |
| 大 JSON 模型请求：`400`，要求 route-hint 头 | 超过 1 MiB 的 JSON 体不会被完整解析，Router 无法推断从混合 worker 池中做选择所需的模型或能力。 | 设置错误消息中指名的头，例如 `x-sglang-omni-route-model` 或 `x-sglang-omni-route-capabilities`；参见[路由行为](routing-behavior)。 |
| 模型请求：`400`，`message` 指名某个 route-hint 头 | route-hint 头格式错误或与 JSON 体不一致：空值、不支持的能力，或与请求体的 `model` 或 `stream` 冲突的值。 | 阅读错误消息；它会指名出问题的头。拒绝日志在 `reason=` 中记录同一消息（空格替换为下划线）。 |
| 模型请求：`503`，`type=overloaded_error`，`Retry-After: 1` | Router 准入已满。 | 退避，并在 `/health` 中检查 `inflight`、`max_inflight` 与 `rejected_total`。 |
| 模型请求：`503`，`message=no eligible upstream` | 没有可路由 worker 匹配该请求。 | 检查 worker 可路由性、模型与能力。 |
| 模型请求：`503` 且带 `X-SGLang-Omni-Worker` | 被选中的 worker 返回了 `503`。 | 检查该 worker 的日志与 `/health` 端点。 |
| 流式响应以 `200` 开始但提前结束 | HTTP 状态发出后上游流失败。SSE 响应以一个 `upstream stream failed before completion` 事件结束，其负载带 `"code": 502`；非 SSE 的流式体会直接截断，没有错误帧。 | 在路由完成日志中检查 `outcome=stream_error`（客户端断开记录的是 `stream_cancelled`），然后检查被选中的 worker。 |

准入限制、拒绝日志与容量建议见[过载行为](overload-behavior)。

选择失败包含 `reason=no_eligible_upstream` 以及推断出的模型与能力。它们发生
在 worker 被选中之前，因此没有 `X-SGLang-Omni-Worker` 头。

worker 返回的 `503` 本身不会驱逐该 worker。

### 区分 `502` 响应

对选择单个 worker 的模型请求，带 `X-SGLang-Omni-Worker` 的 `502` 可能来自
Router，也可能来自被选中的 worker。用响应体区分二者：

- `{"error": {"message": "upstream request failed"}}`：Router 选中了一个
  worker，但连接错误或超时使它无法获得响应。检查被选中的 worker 进程、其
  端口及其 `/health` 端点。
- 其他任何响应体：被选中的 worker 返回了自己的 `502`，由 Router 原样转发。
  排查该 worker 的上游依赖。

该区分不适用于 `/v1/models` 或管理广播路由。这些路由可能返回 Router 生成的
`502`，而不选择单个 worker，因此不带 `X-SGLang-Omni-Worker`。

传输失败与 worker 返回的 `502`/`504` 响应如何影响 worker 健康，见
[故障处理](failure-handling)。

(inspect-worker-state)=
### 查看 worker 状态

打印用于判定可路由性的字段。`worker_id` 是管理路由期望的百分号编码标识符；
`display_id` 是日志中显示的主机与端口：

```bash
curl -s http://127.0.0.1:8008/workers | python3 -c '
import json, sys
for worker in json.load(sys.stdin)["workers"]:
    print(
        "worker_id=" + worker["worker_id"],
        "display_id=" + worker["display_id"],
        "state=" + worker["health_state"],
        "disabled=" + str(worker["disabled"]),
        "routable=" + str(worker["routable"]),
        "failures=" + str(worker["consecutive_failures"]),
        "last_error=" + str(worker["last_error"]),
    )
'
```

| 状态 | 含义与行动 |
|---|---|
| `unknown` | 该 worker 尚未通过 `--health-success-threshold` 次检查。确认其 `/health` 端点可达并等待启动检查。 |
| `unhealthy` | 失败次数达到 `--health-failure-threshold`。检查 `last_status_code`、`last_error` 与 worker 日志。成功的检查会自动恢复它。 |
| `dead` | 该 worker 被手动隔离，健康探测跳过它。先解决问题再清除 `is_dead`；Router 会先检查健康才会再次路由到它。 |
| `disabled: true` | 该 worker 即使健康也被管理性排除。只在它准备好接收新请求时再重新启用。 |

同时确认 `--health-check-endpoint` 与 worker 暴露的端点一致。

对 `no eligible upstream`，把请求与 `/workers` 中每个可路由 worker 的
`model` 与 `capabilities` 逐一对照；能力映射与 route-hint 头见
[路由行为](routing-behavior)。

只有当至少一个候选 worker 声明了 `model` 时，模型过滤才生效。不要通过清除
模型元数据来绕过不匹配：如果没有候选 worker 声明模型，Router 会停止按模型
过滤。

### 排空并移除 worker

先禁用该 worker 使其不再接收新请求，等待 `active_requests` 归零，然后删除它：

```bash
(  # subshell: a failed step exits the procedure, not your shell
worker_id='http%3A%2F%2F127.0.0.1%3A8013'
max_wait_secs=1800
deadline=$((SECONDS + max_wait_secs))

curl -fsS -X PUT "http://127.0.0.1:8008/workers/${worker_id}" \
  -H "Content-Type: application/json" \
  -d '{"disabled":true}' ||
  exit 1

while true; do
  active_requests=$(
    curl -fsS http://127.0.0.1:8008/workers |
      python3 -c '
import json, sys
target = sys.argv[1]
workers = json.load(sys.stdin)["workers"]
worker = next((item for item in workers if item["worker_id"] == target), None)
print(worker["active_requests"] if worker else "missing")
' "${worker_id}"
  ) || exit 1
  case "${active_requests}" in
    0) break ;;
    missing)
      echo "worker ${worker_id} is not registered; check /workers" >&2
      exit 1
      ;;
  esac
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for ${worker_id} to drain" >&2
    exit 1
  fi
  sleep 2
done

curl -fsS -X DELETE "http://127.0.0.1:8008/workers/${worker_id}"
)
```

请从 `/workers` 复制 `worker_id`，不要手工构造；[查看 worker 状态](inspect-worker-state)
中的片段会以可直接粘贴的形式打印它。对响应可能合法超过默认 30 分钟排空窗口
的工作负载，请调大 `max_wait_secs`。

删除 worker 只移除其 Router 注册。排空完成之后，请另行停止该 worker 进程。
