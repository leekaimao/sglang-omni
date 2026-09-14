# Omni 模型使用方法

本指南以 [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) 为例，介绍如何在 SGLang-Omni 中配合 OpenAI 兼容 API 使用 omni 模型。Qwen3-Omni 支持多模态输入（文本、图像、音频），并可按模式产生纯文本或文本 + 音频输出。

## 前置条件

按照[安装](../get_started/installation.md)指引安装 `sglang-omni`。

## 纯文本模式

纯文本模式在单块 GPU 上运行 thinker 流水线。它接受多模态输入（文本、图像、音频），仅产生文本输出。

### 启动服务器

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --port 8008
```

多模态预处理（tokenization、图像/视频/音频特征提取）默认串行执行。当 CPU 预处理在并发负载下成为瓶颈时，添加 `--preprocessing.factory.max_concurrency 4` 使其在线程池上运行。线程化预处理会改变请求到达 thinker 的方式，因此 bf16 MoE 的贪心输出可能与串行默认值不同；在比较多次运行的准确率时请保持默认配置。

对于请求较短的 MMSU 风格音频输入 / 文本输出基准测试，请使用融合文本路径（fused text-path）配置，让完整文本路径保持在同一个 worker 进程内：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_mmsu.yaml \
  --text-only \
  --port 8008
```

### 图像和文本输入

发送一张图像并附带一个文本问题，即可获得文本回复。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

### 音频和图像输入

发送一个音频文件并附带一张图像。音频中包含口述的问题（“How many cars are there in the picture?”），模型基于两种输入作答。

> **注意：** 当全部语义内容来自音频、视频或图像而非文本时，请在用户消息上设置 `"content": ""`（空字符串）。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "images": ["tests/data/cars.jpg"],
    "audios": ["tests/data/query_to_cars.wav"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "images": ["tests/data/cars.jpg"],
        "audios": ["tests/data/query_to_cars.wav"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

### 视频和音频输入

发送一段视频并附带口述的音频问题。模型观看视频、听取问题，并以文本回应。

Video-AMME CI 基准测试使用的正是这种模态组合：视频输入加上口述问题/选项的 WAV，文本消息中仅包含路由和答案格式指令。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "videos": ["tests/data/draw.mp4"],
    "audios": ["tests/data/query_to_draw.wav"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "videos": ["tests/data/draw.mp4"],
        "audios": ["tests/data/query_to_draw.wav"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

## 语音模式

语音模式在一块或多块 GPU 上运行完整的八阶段流水线。它同时产生文本（来自 thinker）和音频（来自 talker）输出。

### 编解码器帧合并与首音频延迟

语音流水线在 `stages.talker_ar.factory` 之下设置了 `codec_coalesce_frames=10`、`codec_coalesce_early_frames=10` 和 `codec_coalesce_first_frames=0`。前 10 个编解码器（codec）帧逐个发送；后续帧按每 10 帧合并成组发送。省略 YAML 覆盖项即可保留这些流水线默认值；显式设置 `codec_coalesce_early_frames=0` 可禁用早期前缀。这与默认串行 Code2Wav 的 10 帧阈值保持一致：前三个窗口分别包含 10、20 和 30 帧，后续完整窗口包含 35 帧（含左上下文）。在启用 CUDA Graph 时，这些形状可以使用已捕获的串行窗口。在这种串行配置下改用 12 的早期前缀则会产生 22 帧和 32 帧的窗口，回落到 eager 执行。

在启用 Code2Wav 批处理、`initial_codec_chunk_frames=2` 且 `stream_chunk_size=10` 的情况下，显式设置 `codec_coalesce_early_frames=12`，可使前两个窗口分别在已生成第 2 帧和第 12 帧时满足条件。而均匀的 10 帧分组（`early_frames=0`、`first_frames=0`）则要到第 11 步才发布第一组：发送方会保留最新的一行，直到下一步可以排除 EOS，或请求结束。因此，对于延续到第 10 步之后的请求，首窗口的输入就绪时间会从第 2 步推迟到第 11 步，增加九个 Talker 解码间隔。若该间隔约为 `d` 毫秒，则新增的输入等待约为 `9d` 毫秒。而对于串行 10 帧首窗口，就绪时间则是从第 10 步推迟到第 11 步。默认的 10 帧早期前缀可使就绪时间保持在第 10 步；接下来的两个窗口分别在第 21 步和第 31 步就绪。与不合并时的第 20 步和第 30 步相比，前缀之后的这一额外步骤用于为 EOS 检测保留最新的一行。

这只是一个输入就绪时间的估算，并不是实测的端到端 TTFA 差值，也不是九帧音频的播放时长。实际 TTFA 还取决于传输、排队和声码器（vocoder）执行；整体的合并基准测试并未单独隔离早期前缀设置的影响。

### 启动服务器

语音模式可以作为一个共置（colocated）的单 GPU worker，使用共置配置运行：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --port 8008
```

在单卡 H200 worker 上请使用 `examples/configs/qwen3_omni_colocated_h200.yaml`。

Qwen3-Omni Code2Wav 默认启用精确形状的 CUDA Graph replay。默认阶段配置提供 2% 的带类型 GPU 显存预算；共置示例配置会用其针对具体硬件的预算覆盖该值。

要禁用 replay，请在 YAML 配置中的阶段上设置：

```yaml
stages:
  code2wav:
    factory:
      enable_cuda_graph: false
```

启用 replay 时，自定义的 Code2Wav 阶段必须定义 `gpu_memory_fraction`；启动过程会在加载模型之前拒绝缺失带类型预算的情况。

该功能从 `stream_chunk_size` 和 `left_context_size` 推导出精确的 `B=1` 阈值窗口；默认捕获 `T{10,20,30,35}`。不支持的形状以及最后的流式尾部会以 eager 方式运行。捕获时不兼容的情况也会回落到 eager 执行。

输出重叠（output overlap）在 CUDA 设备上同样默认启用：每个阈值窗口的波形回读作为异步的设备到主机拷贝进入一个锁页中转缓冲区，并在 GPU 计算下一个窗口的同时完成物化；编解码器的 EOS 检查也改为每个窗口执行一次，而不是每帧一次。每个请求的第一个窗口保持同步，因此首音频时间不变。对于成功完成的请求，音频字节和消息边界与同步路径完全一致。失败或被中止的请求在其传输中的中转缓冲区被回收时，可能会丢弃一个尚未发出的待处理窗口。要禁用它：

```yaml
stages:
  code2wav:
    factory:
      enable_output_overlap: false
```

若要手动进行多 GPU 部署，请使用示例脚本：

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 0 \
  --port 8008
```

或者使用不带 `--text-only` 的 CLI 启动标准语音流水线：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008
```

默认情况下，请保持 `mem_fraction_static` 未设置，让 SGLang-Omni 自动确定 SGLang AR 显存预算的大小。如果特定机器需要手动调优，可以全局固定该值，或按 AR 阶段固定：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008 \
  --mem-fraction-static 0.88
```

当 thinker 和 talker 需要不同的预算时，请使用按阶段的标志：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008 \
  --thinker.engine.mem_fraction_static 0.88 \
  --talker_ar.engine.mem_fraction_static 0.88
```

语音服务器启动器提供同样的按阶段控制项：

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 0 \
  --port 8008 \
  --thinker-mem-fraction-static 0.88 \
  --talker-mem-fraction-static 0.88
```

`--mem-fraction-static` 适用于每个 SGLang 引擎阶段。以点号分隔的按阶段路径会覆盖该阶段的全局值。取值必须大于 `0` 且小于 `1`。

thinker 默认最多允许 64 个运行中的请求。在纯文本或语音模式下，均可使用 thinker 专属标志调低或调高该上限：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --thinker.engine.max_running_requests 16
```

如果要改为通过流水线 YAML 文件配置 thinker，请在阶段条目下设置相同的路径：

```yaml
stages:
  thinker:
    engine:
      max_running_requests: 16
```

### 语音阶段放置

在并发 8 时，talker 是最重的语音阶段：它让所在 GPU 保持在约 86% 的中位利用率。请给它一块独享的 GPU。相比之下 code2wav 声码器很轻（9–13% 中位利用率），默认与 thinker 共享 GPU。这一默认仅在 thinker 留在其自身的默认 GPU 上时才成立——`--thinker.gpu` 覆盖只会移动 thinker 而不会移动 code2wav，因此如果你迁移 thinker，请同时显式传入 `--code2wav.gpu`。

这就是不带 GPU 覆盖时 `sgl-omni serve` 的默认拓扑：thinker 独占、talker 独占、code2wav 位于 thinker 的 GPU 上。当 code2wav 与 thinker 共享 GPU 时，thinker 自动确定大小的 KV 池会收缩以便腾出空间——在 H200 上实测约小 4.3 GiB，因为声码器本身大约需要 1.4–1.6 GiB。该调整只发生在自动确定预算的情况下：显式固定的 `--thinker.engine.mem_fraction_static` 不会获得自动划拨，因此紧贴上限的固定比例应为 code2wav 留出余量。

一项并发 8 的实验将 code2wav 固定在 thinker 的 GPU 上，仅改变 talker 的放置方式，从与 thinker 共享 GPU 变为独享 GPU。所有并发指标均有改善：

| 指标（并发 8，两组测量值） | talker 与 thinker 共享 GPU | talker 独享 | 变化 |
|---|---|---|---|
| 墙钟时间 | 22.8–24.1 s | 20.2–20.7 s | 降低 9–16% |
| 首音频时间（TTFA），p50 | 1.43–1.71 s | 0.99–1.20 s | 降低 16–42% |
| TTFA，p90 | 2.66–2.70 s | 1.42–1.46 s | 降低 46–47% |
| 端到端延迟，p50 | 10.04–10.05 s | 8.94–9.44 s | 降低 6–11% |

随后的一组单变量对照实验改为测试将 code2wav 移到自己的 GPU 上，两组中 talker 均已隔离。墙钟时间和端到端延迟基本持平（−0.7% 和 −0.9%，完全在单组交错对照的噪声范围内）——给声码器一块专属 GPU 毫无收益，这正是 code2wav 与 thinker 共享 GPU 而不像 talker 那样被隔离的原因。

单流流量（一次一个请求）则呈现不同的权衡。在上述 talker 放置实验中，隔离 talker——将其移出 thinker 的 GPU——使单流端到端延迟恶化了 14–17%：只有一个在途请求时并无争用可逃避，而该实验中 thinker 到 talker 的交接开始跨越设备边界。单流 TTFA 仍改善了约 30%（从 0.48–0.49 s 降至 0.34–0.35 s），因为 talker 不再在共享 GPU 上排队等待。旧的和新的默认布局本就将 thinker 和 talker 保持在不同的 GPU 上，因此这些数字描述的是该实验的拓扑变化，而不是默认的 code2wav 放置方式。无论如何，流式 TTS 负载都应优先选择这种布局，因为 TTFA 是用户最先察觉到的延迟。

### 服务端轮次检测的实时语音

语音流水线可以通过 `/v1/realtime` 流式输出语音回复。在标准语音流水线上启用该 WebSocket 端点：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008 \
  --enable-realtime
```

连接到 `ws://localhost:8008/v1/realtime` 之后，请求文本和音频输出：

```json
{
  "type": "session.update",
  "session": {
    "modalities": ["text", "audio"],
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "turn_detection": {
      "type": "semantic_vad",
      "eagerness": "medium"
    }
  }
}
```

使用 `input_audio_buffer.append` 流式发送单声道 16 kHz PCM16 输入。轮次检测会自动提交每条话语并开始生成。在请求 `semantic_vad` 之前请先检查 `session.created.capabilities.turn_detection`——较旧的服务器只支持 `server_vad`。

`server_vad`（默认)在固定静音时长后结束一个轮次。`semantic_vad` 在 Silero 语音检测之上增加了一个 GPU Smart Turn v3.2 模型，因此思考中途的自然停顿不会提前结束轮次。`eagerness`（`low`/`medium`/`high`，默认为 `medium`）在延迟与耐心之间权衡；`silence_duration_ms` 仅适用于 `server_vad`。

要启用 `semantic_vad`，请先准备好采用 BSD-2 许可证的 [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) `smart-turn-v3.2-gpu.onnx` 模型，并将 `SGLANG_OMNI_SMART_TURN_MODEL_PATH` 设置为其路径（文件或所在目录）。服务器绝不会自动下载该模型，并在加载时校验其 SHA-256。如果模型缺失或无效，端点仍然可用——语义请求只会回落到 `server_vad`。

文本通过 `response.text.delta` 事件到达；语音输出以 base64 编码的单声道 24 kHz PCM16 形式通过 `response.audio.delta` 事件到达，随后是 `response.audio.done` 和 `response.done`。

音频输出需要显式开启：除非同时请求两种模态，会话保持纯文本。仅含 thinker 的服务器会拒绝音频协商，因为它没有 `code2wav` 阶段。

对于文本加音频会话，服务器侧的打断（barge-in）默认启用。当当前轮次检测器发出 `input_audio_buffer.speech_started` 时，当前响应会以原因 `turn_detected` 被取消；其用户转写仍会完成，并在下一个排队的轮次运行之前进入对话历史。被取消的助手输出不会加入对话历史。

客户端必须在 `speech_started` 时停止缓冲播放，并在该响应的 `response.done` 到来之前拒绝其后针对该响应的每一个 `response.audio.delta`。如果语音在 `response.created` 之前开始，请保留一个待打断标志，并在该响应的 ID 到达时将其拒绝。自动打断以 `response.done.status="cancelled"` 和原因 `turn_detected` 结束；显式的 `response.cancel` 使用原因 `client_cancelled`。

如果助手音频已被排入播放计划，客户端还必须发送 `conversation.item.truncate`，附带取自 `response.audio.delta` 的助手 `item_id`、`content_index: 0`，以及以 `audio_end_ms` 表示的已播放时长。服务器回复 `conversation.item.truncated`，并从对话历史中移除该助手条目。整个助手转写都会被移除，因为该端点无法将文本与已播放的音频对齐。

在 `session.update` 中将 `turn_detection.interrupt_response` 设置为 `false` 即可选择退出。部分更新会保留当前检测器的类型和设置，因此客户端可以在不放弃语义 VAD 或其 eagerness 的情况下更改打断行为。更改检测器行为会重建检测器并清除待处理的输入音频；`interrupt_response` 则独立变化。纯文本响应不会被自动打断。

`playground/qwen-omni/realtime` 中的浏览器示例会采集麦克风输入，按连接协商轮次检测支持情况，并让用户选择仅文本输出或文本加流式 PCM16 音频播放。

## H100/H20 上的单 GPU FP8

SGLang-Omni 也可以服务原生 FP8 的 Qwen3-Omni checkpoint。原生 FP8 在加载 thinker 和 talker AR 阶段时使用 checkpoint 的量化配置，同时保留下文所示的相同 Qwen3-Omni 请求格式。

对于单 GPU H100/H20 的共置启动，请使用 FP8 共置配置：

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

该配置文件中包含 FP8 checkpoint 路径：`marksverdhei/Qwen3-Omni-30B-A3B-FP8`。你仍可以传入 `--model-path` 来覆盖配置值。

FP8 路径让稠密 FP8 GEMM 保持 SGLang `auto`，并在受支持时将原生 FP8 MoE 默认设为 CUTLASS。对于 Qwen3-Omni 流水线启动，除非运维人员已经设置该环境变量，否则 `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` 会作为默认值被设置。这会禁用 SGLang 的全 M DeepGEMM 预编译会话，同时保持 DeepGEMM 可用于稠密 FP8 GEMM。

要重新启用 SGLang 的全 M DeepGEMM 预编译行为：

```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=1 sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

## H100/H20 上的单 GPU AutoRound INT4 Thinker

SGLang-Omni 同样支持 AutoRound INT4 量化的 Qwen3-Omni checkpoint。AutoRound 使用组大小为 128 的 4 比特量化方案，相比 BF16 或 FP8 显著降低显存占用。

公开的 AutoRound checkpoint 量化了 thinker 的 transformer 层。在语音模式下，talker 和 code2wav 阶段从同一 checkpoint 以 BF16 加载。对于单 GPU H100/H20 的共置启动，请使用共置配置并搭配 AutoRound checkpoint：

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --model-name qwen3-omni \
  --model-path Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound \
  --port 8008
```

AutoRound 量化提供：
- 相比 BF16 **约 50% 的显存削减**（从约 60GB 降至约 30GB）
- 相比 FP8 **约 25% 的显存削减**（从约 40GB 降至约 30GB）
- **超低比特宽度下的准确率**：即使在 2–4 比特下也能保持高准确率，得益于其符号梯度下降优化，所需的调优工作量极小。

### 图像和文本输入

发送一张图像并附带一个文本问题，即可同时获得文本和音频回复。设置 `"modalities": ["text", "audio"]` 以启用音频输出。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### 音频和图像输入

发送一个音频文件并附带一张图像。模型听到口述的问题并看到图像，然后同时以文本和音频回应。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "images": ["tests/data/cars.jpg"],
    "audios": ["tests/data/query_to_cars.wav"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "images": ["tests/data/cars.jpg"],
        "audios": ["tests/data/query_to_cars.wav"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### 视频和音频输入

发送一段视频并附带口述的音频问题。模型观看视频、听取问题，并同时以文本和音频回应。

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "videos": ["tests/data/draw.mp4"],
    "audios": ["tests/data/query_to_draw.wav"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "videos": ["tests/data/draw.mp4"],
        "audios": ["tests/data/query_to_draw.wav"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

## 请求参数

下表列出了 `/v1/chat/completions` 端点为 Qwen3-Omni 接受的全部参数。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | `null` | 模型标识符 |
| `messages` | list | （必填） | 聊天消息列表，每条包含 `role` 和 `content` |
| `modalities` | list | `["text"]` | 输出模态：`["text"]` 表示仅文本，`["text", "audio"]` 表示文本和音频 |
| `images` | list | `null` | 图像文件路径列表（本地路径或 URL） |
| `audios` | list | `null` | 音频文件路径列表（本地路径或 URL） |
| `videos` | list | `null` | 视频文件路径列表（本地路径或 URL） |
| `max_tokens` | int | `null` | 可生成的最大 token 数 |
| `max_completion_tokens` | int | `null` | `max_tokens` 的 OpenAI 兼容别名 |
| `temperature` | float | `null` | 采样温度 |
| `top_p` | float | `null` | Top-p 采样 |
| `top_k` | int | `null` | Top-k 采样 |
| `repetition_penalty` | float | `null` | 重复惩罚 |
| `seed` | int | `null` | 用于可复现性的随机种子 |
| `stream` | bool | `false` | 通过 SSE 启用流式输出 |
| `audio` | dict | `null` | 语音响应格式配置，例如 `{"format": "wav"}` |
| `stage_sampling` | dict | `null` | 按阶段的采样覆盖项，例如 `{"thinker": {"temperature": 0.8}}` |
