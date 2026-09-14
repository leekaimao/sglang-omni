# API Server 快速入门

本页面是从"我拿到了仓库"到"API 服务器响应请求"的最短路径。

`sglang-omni` 在其多阶段流水线运行时之上提供了一个 OpenAI 兼容的 API 服务器。该服务器是以下功能的主要 HTTP 入口：

- 聊天补全
- 流式响应
- 模型列表
- 健康检查
- 文本转语音

如果你希望了解内部设计而非使用流程，请参阅 [API 服务器设计](../developer_reference/apiserver_design.md)。

## 这个服务器是什么

API 服务器是 HTTP 客户端与内部流水线运行时之间的适配器：

`HTTP request` → `FastAPI app` → `Client` → `Coordinator` → `Pipeline stages`

换句话说，它并不直接运行模型逻辑。它的职责是把 OpenAI 风格的请求转换为内部请求，并把结果格式化为 HTTP 响应返回。

## 启动服务器

安装后提供的 CLI 入口是 `sgl-omni`。

启动服务器最简单的方式是提供一个模型路径，让 `sglang-omni` 为你构建流水线配置：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

最有用的标志包括：

- `--model-path`：Hugging Face 模型 ID 或本地模型目录
- `--host`：绑定地址，默认为 `0.0.0.0`
- `--port`：绑定端口，默认为 `8000`
- `--model-name`：覆盖 `/v1/models` 返回的模型名称
- `--log-level`：服务器进程的日志级别

如果你已经有一个流水线配置文件，也可以传入 `--config path/to/config.yaml`。当配置文件中包含 `model_path` 时，`--model-path` 是可选的，可用作覆盖项。

### 统一的 SGLang CLI

当 SGLang-Omni 与支持 serve 后端插件的 SGLang 版本一起安装时，同一个服务器也可以通过核心可执行程序启动：

```bash
sglang serve <model-name-or-path> --model-type omni [additional-arguments]
```

也支持仅用配置文件启动：

```bash
sglang serve --model-type omni --config path/to/config.yaml
```

SGLang-Omni 有意不安装另一个 `sglang` 可执行程序。它向 SGLang 核心注册一个 `omni` serve 后端，同时 `sgl-omni serve` 仍作为向后兼容的别名保留。自动的 Omni 模型检测尚未启用；请使用 `--model-type omni` 显式选择该后端。

## 验证是否正常工作

### 健康检查

```bash
curl -s http://localhost:8000/health
```

示例响应：

```json
{
  "status": "healthy",
  "running": true
}
```

服务器返回：

- `200`：运行时健康
- `503`：HTTP 服务器已启动，但底层运行时报告不健康状态

### 列出被服务的模型

```bash
curl -s http://localhost:8000/v1/models
```

该端点返回一个单模型列表。模型 ID 来自你设置的 `--model-name`，否则来自流水线名称。

## 发送一个最小的聊天请求

核心端点是 `POST /v1/chat/completions`。

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 128,
    "stream": false
  }'
```

响应遵循 OpenAI 聊天补全的结构。在常见情况下，文本位于 `choices[0].message.content`。

除 `model` 和 `messages` 之外，最有用的请求字段包括：

- `temperature`
- `top_p`
- `max_tokens`
- `stop`
- `seed`
- `stream`

## 流式与多模态请求

### 流式

将 `stream` 设为 `true` 以接收 Server-Sent Events（SSE）：

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Write a short greeting."}
    ],
    "stream": true
  }'
```

有几个细节值得注意：

- 响应类型为 `text/event-stream`
- 第一个分块可能只包含 `role="assistant"`
- 流以 `data: [DONE]` 结束
- `usage` 附着在最后一个补全分块上

### 多模态输入与输出

`sglang-omni` 在标准 OpenAI 聊天 schema 之上扩展了几个额外字段：

- `images`
- `audios`
- `videos`
- `modalities`
- `audio`
- `stage_sampling`
- `stage_params`

例如，一个输出文本的视频请求如下所示：

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "What is happening in this video?"}
    ],
    "videos": ["/absolute/path/to/demo.mp4"],
    "modalities": ["text"],
    "max_tokens": 128,
    "stream": false
  }'
```

`videos`、`images` 和 `audios` 字段既接受本地文件路径，也接受 HTTP(S) URL。

## 文本转语音

服务器还提供 `POST /v1/audio/speech`。

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "input": "Hello from SGLang-Omni.",
    "voice": "default",
    "response_format": "wav"
  }' \
  -o speech.wav
```

有两点需要记住：

- 响应体是音频字节，而不是 JSON
- 如果编码器回退到另一种受支持的编解码器（codec），实际输出格式可能与请求的格式不同

## 常见错误

请求失败时，服务器返回标准的 HTTP 错误码：

- `400 Bad Request`：请求体格式错误或参数无效
- `500 Internal Server Error`：生成期间发生运行时错误（详情请查看服务器日志）
- `503 Service Unavailable`：运行时不健康（可通过 `/health` 验证）

如果遇到 500 错误，请查看服务器日志以获取完整的 traceback。常见问题包括：
- 不支持的媒体格式
- 内存不足错误
- 模型文件缺失

## 延伸阅读

- [API 服务器设计](../developer_reference/apiserver_design.md)
- [开发者参考](../developer_reference/main.md)
