# API Server 设计

本页从最有利于维护的层面解释 API server：它在系统中的位置、哪些文件重要，以及请求如何映射进运行时。

如果你只想启动服务器并调用它，请先看 [API Server 快速上手](../get_started/apiserver_quickstart.md)。

## 在系统中的角色

API server 是叠在 `sglang-omni` 流水线运行时之上的最外层协议层。

概览而言，内置的服务器启动路径是：

`CLI / Python 入口` → `PipelineConfig` → `流水线启动` → `Coordinator` → `Client` → `FastAPI`

启动之后的请求路径是：

`HTTP 请求` → `FastAPI 路由` → `Client` → `Coordinator` → `Stage 流水线` → `Client 聚合` → `HTTP/SSE 响应`

这样的分工让职责保持清晰：

- 流水线运行时负责编排与执行
- `Client` 层提交请求并组装结果
- API server 在 HTTP/OpenAI 风格的负载与这些内部抽象之间做转换

## 关键文件

就当前的服务器实现而言，下面这些文件最重要。

| 文件 | 角色 |
| --- | --- |
| `sglang_omni/serve/openai_api.py` | 定义 FastAPI 应用、路由、请求转换与响应格式化 |
| `sglang_omni/serve/protocol.py` | 定义请求与响应 schema |
| `sglang_omni/serve/launcher.py` | 编译流水线、启动运行时、挂载应用并运行 Uvicorn |
| `sglang_omni/client/client.py` | 向 coordinator 提交请求并聚合文本、音频与流式结果 |
| `sglang_omni/cli/serve.py` | 定义 `sgl-omni serve` 当前的 CLI 面 |

如果你在追查端点行为，`openai_api.py` 与 `client.py` 通常是最好的起点。

## `create_app()` 与 `launch_server()`

这是服务代码中最重要的一个区分。

### `create_app(client, model_name=...)`

`create_app()` 只构建 FastAPI 应用并注册核心路由。

它**不会**：

- 编译流水线
- 启动运行时
- 创建 coordinator
- 挂载 profiling 路由
- 运行 Uvicorn

当你已经持有一个活跃的 `Client` 并想自己嵌入 HTTP 层时使用它。

### `launch_server(pipeline_config, ...)`

`launch_server()` 是完整的内置服务器生命周期。

它会：

- 编译流水线配置
- 启动流水线运行时
- 创建 `Client`
- 创建 FastAPI 应用
- 在单进程路径上挂载 profiling 路由
- 运行 Uvicorn
- 在关闭时停止运行时

当你想要标准的开箱即用服务器路径时使用它。

## 路由面

当前服务器暴露这些主要路由：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 来自 `client.health()` 的健康状态 |
| `GET` | `/v1/models` | 当前活跃流水线的单模型列表 |
| `POST` | `/v1/chat/completions` | Chat completions，含流式与可选音频 |
| `POST` | `/v1/audio/speech` | 文本转语音，`stream=true` 时返回原始音频或原始 PCM chunk |
| `POST` | `/start_profile` | Torch trace +（可选）请求级事件。由内置 launcher 添加 |
| `POST` | `/stop_profile` | 同时停止 torch trace 与请求级事件 |
| `POST` | `/start_request_profile` | 仅请求级事件记录器（不含 torch trace） |
| `POST` | `/stop_request_profile` | 停止请求级事件记录器 |

profiling 路由由单进程的 `launch_server()` 路径挂载。当前的多进程 launcher 路径不会挂载它们。

`/start_profile` 接受：

```jsonc
{
  "run_id": "demo-run",
  "trace_path_template": "/tmp/profiles/demo-run/trace",  // torch trace template
  "event_dir": "/tmp/profiles/demo-run/events",            // request-event JSONL dir (optional)
  "enable_torch": true                                     // set false to skip torch trace
}
```

`/stop_profile` 与 `/stop_request_profile` 都接受可选的
`run_id`。省略它相当于通配：每个阶段停止当前活跃的任何 profiler
会话。

请求级事件以 JSON 行的形式写到
`<event_dir>/events_<stage>_<pid>.jsonl`。使用 `python -m sglang_omni.profiler
<event_dir>` 可推导出 `docs/developer_reference/profiler.md` 所述的时间线 / 阶段 / 跳转报告。

## 请求映射

服务器不会把 OpenAI 风格的请求体直接塞进运行时，而是先将其转换为内部请求对象。

### Chat 请求

`ChatCompletionRequest` 包含标准的 OpenAI 风格字段，例如：

- `model`
- `messages`
- `temperature`
- `top_p`
- `max_tokens`
- `stop`
- `seed`
- `stream`

它还包含 `sglang-omni` 扩展，例如：

- `images`
- `audios`
- `videos`
- 视频处理覆盖，如 `video_fps` 与帧/像素上限
- `modalities`
- `audio`
- `stage_sampling`
- `stage_params`
- talker 专属的生成覆盖
- `request_id`

### 转换为 `GenerateRequest`

`openai_api.py` 中的 `_build_chat_generate_request()` 是关键的转换点。它会：

- 归一化停止序列
- 构建 `SamplingParams`
- 把 chat 消息转换为内部 `Message` 对象
- 映射按阶段的采样覆盖
- 通过 `stage_params` 透传按阶段的运行时参数
- 把媒体输入、音频配置与视频处理覆盖存入请求元数据
- 把 talker 专属的生成覆盖存入 `extra_params`
- 把 `modalities` 复制到 `output_modalities`

路由把这个 `GenerateRequest` 交给 `Client`。客户端随后把它转换为
`OmniRequest`，再提交给 coordinator。

## 响应路径

### 非流式 chat

对非流式 chat，路径大致是：

`chat 请求` → `Client.completion()` → OpenAI 风格的 JSON 响应

`Client.completion()` 聚合：

- 文本片段
- 音频 chunk
- 最终 usage
- 最终 finish reason

如果存在音频，在返回给 API 层之前会做 base64 编码。

### 流式 chat

对流式 chat，服务器发出 SSE 事件。

当前的重要语义是：

- 第一个 chunk 可能只包含 `role="assistant"`
- 文本与音频作为独立的 delta 发出
- 最后一个 completion chunk 包含 `finish_reason`
- 流以 `data: [DONE]` 结束
- `usage` 附在最后一个 completion chunk 上

### Speech / TTS

speech 路由复用同一条内部请求路径，而不是引入另一套服务栈。

`CreateSpeechRequest` 被转换为这样一个 `GenerateRequest`：

- `output_modalities=["audio"]`
- 元数据中的 `task="tts"`
- TTS 专属参数存储在 `tts_params` 之下

对非流式请求，`Client.speech()` 收集音频 chunk、编码，
并向 HTTP 层返回原始音频字节。

对 `stream=true`，路由直接发出 `audio/pcm` 字节。HTTP 响应头从第一个音频 chunk 推导，后续 chunk 必须保持相同的采样率。TTS 的 chunk 节奏旋钮，如
`initial_codec_chunk_frames`，只在客户端提供时才作为请求参数转发，包括显式的 `0`，这样各模型的调度器就能消费它们，而无需改动 Stage、Coordinator 或 Relay。省略该字段时，speech 路由保持其未设置状态，每个模型应用自己的流式默认值。
