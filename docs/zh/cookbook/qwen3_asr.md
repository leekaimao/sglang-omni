# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 是通过 OpenAI 兼容的 `/v1/audio/transcriptions` 端点提供服务的音频转写模型。每个请求接受一个上传的音频文件并返回文本。

Qwen3-ASR 不支持 `/v1/audio/translations`；该端点会返回 HTTP 400。请使用 `/v1/audio/transcriptions`。

## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`，然后下载模型：

```bash
MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5
MODEL_PATH="$(
  hf download Qwen/Qwen3-ASR-1.7B \
    --revision "${MODEL_REVISION}" \
    --quiet
)"
```

### Apple Silicon（MLX）

Apple Silicon 路径要求 macOS 14 或更高版本、Python 3.12、Homebrew，以及 SGLang 的 MLX 运行时。音频解码还需要 Homebrew 的带版本号 FFmpeg 7 formula：

```bash
brew install ffmpeg@7
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

不要把 `ffmpeg@7` 换成不带版本号的 `ffmpeg` formula。后者目前安装的是 FFmpeg 9，而 Apple 侧安装的 `torchcodec==0.15.0` 只支持 FFmpeg 4 到 8。由于 `ffmpeg@7` 是 keg-only 的，服务器每次启动时其库目录也必须出现在 `DYLD_LIBRARY_PATH` 中。

当受 SIP 保护的系统可执行文件启动服务器时，macOS 可能会剥离 `DYLD_*` 变量。请把 `DYLD_LIBRARY_PATH` 设置在最终的 `sgl-omni` 进程上；例如把 `/usr/bin/env DYLD_LIBRARY_PATH=...` 放在 `/usr/bin/time` 之类包装器之后。请用 M4A 或 MP3 等压缩格式输入做测试，因为 WAV 解码可能在不加载 FFmpeg 的情况下也能成功。

为两个仓库创建同一个虚拟环境，先从源码安装固定版本的 SGLang tag 及其 `all_mps` 依赖，再安装 SGLang-Omni：

```bash
git clone --branch v0.5.19 https://github.com/sgl-project/sglang.git
git clone https://github.com/sgl-project/sglang-omni.git

uv venv -p 3.12 sglang-omni/.venv-apple
source sglang-omni/.venv-apple/bin/activate

cd sglang
cp python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"

cd ../sglang-omni
uv pip install -e .
```

这样会通过 SGLang 安装 MLX。它不会安装也不会使用 `mlx-audio` 包。下载模型之前，请验证 Metal 与 FFmpeg 都能加载：

```bash
SGLANG_USE_MLX=1 python - <<'PY'
import mlx.core as mx
from torchcodec.decoders import AudioDecoder

assert mx.metal.is_available()
print("MLX Metal and TorchCodec FFmpeg loading are available")
PY
```

使用经 MLX 转换的 Qwen3-ASR checkpoint 并选择 MLX runner：

```bash
export SGLANG_USE_MLX=1
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

sgl-omni serve \
  --model-path mlx-community/Qwen3-ASR-0.6B-4bit \
  --model-name Qwen/Qwen3-ASR-0.6B \
  --asr.engine.max_running_requests 1 \
  --port 8000
```

MLX 路径目前支持单设备（`tp_size=1`）与贪心解码。该路径不使用 Radix cache、分块 prefill 和 CUDA graph。下文的 HTTP 与 SSE 转写接口与 CUDA 上一致；`stream=true` 会在 token 解码时提供伪流式的转写增量。Apple 路径目前不提供采样惩罚或 token logprobs。MLX 可以批处理多个请求，但在意单请求延迟时建议 `max_running_requests=1`；只有在优先考虑吞吐时才调高它。

若改用 Torch MPS 兼容路径，请保持 `SGLANG_USE_MLX` 未设置，并传入官方 PyTorch Qwen3-ASR checkpoint。它目前使用单设备、贪心解码以及 eager 的 `torch_native`/`sdpa` profile：

```bash
unset SGLANG_USE_MLX
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-0.6B \
  --model-name Qwen/Qwen3-ASR-0.6B \
  --asr.engine.max_running_requests 1 \
  --port 8000
```

初始 Torch MPS profile 被限制在 2,048 token 的 KV 预算内，非流式分块使用 `audio_chunking.max_audio_clip_s`（默认 30 秒），并把原生/整段上传请求限制在 60 秒。超过该合格上限的取值会被拒绝。长音频和更大的原生上下文上限请使用 MLX 路径。

## 服务器配置

Qwen3-ASR 在单块 GPU 上运行一个 ASR 阶段。其默认的 `auto` dtype 跟随 checkpoint 配置（Qwen3-ASR-1.7B 为 BF16）；传入
`--asr.factory.dtype float16` 可强制使用 FP16。
所有解码 batch 大小都默认启用异步解码，使得共享的一步前瞻（one-step-lookahead）路径即使在单请求下也能把宿主侧结果处理与下一次 GPU 解码前向重叠。使用
`--asr.factory.enable_async_decode false` 可禁用，或通过
`--asr.factory.async_decode_min_batch_size` 调整切换点。
请求构建器同样使用共享的 LM prefill 准入门控：当 16 个已构建请求就绪，或最早的就绪请求等待超过 40 ms 时，prefill 启动。一旦请求构建工作排空，若解码空闲则立即释放就绪的 prefill；解码活跃时，则继续合并直到达到相同的请求目标或截止时间。

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

对于单张 24 GB 的 RTX 4090（SM89），可使用仓库内的消费级 profile：

```bash
sgl-omni serve \
  --config examples/configs/qwen3_asr_rtx4090.yaml \
  --port 8000
```

该合格 profile 保持模型为 BF16，将阶段限制为 16 个运行中请求，并把 `mem_fraction_static` 设为 `0.65`。其边界是针对已验证的 RTX 4090 布局设定的；在其他 GPU 架构上请使用默认配置或另行验证的 profile。

例如，在对比模式时强制同步解码：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --asr.factory.enable_async_decode false \
  --port 8000
```

## 转写音频

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "Qwen/Qwen3-ASR-1.7B",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## 流式转写

设置 `stream=true` 可通过 SSE 接收增量转写。使用
`curl -N` 禁用客户端响应缓冲：

```bash
curl -N -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json \
  -F stream=true
```

流中包含零个或多个 delta 事件，随后是完整的最终转写与 SSE 哨兵：

```text
data: {"type":"transcript.text.delta","delta":"..."}

data: {"type":"transcript.text.done","text":"..."}

data: [DONE]
```

Qwen3-ASR 默认将 delta 最多缓冲 50 ms。EOS 及其他终止条件会在最终转写事件之前冲刷所有已缓冲文本。

### 实时 PCM 转写

上面的 SSE 模式在完整的 multipart 上传之后才开始解码。若要摄入实时音频，请挂载 realtime WebSocket 端点：

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --enable-realtime \
  --port 8000
```

连接 `/v1/realtime?intent=transcription`，配置会话，然后追加 base64 编码的 16 kHz 单声道 PCM16 包。`input_audio_buffer.commit`
手动结束当前分段；服务器 VAD 也会在配置的静音时长之后结束分段。发送完最后一个包后发送 `transcription.done` 即可收到
`transcription.completed`。`input_audio_buffer.clear` 丢弃当前分段及其派生的部分假设，同时保持 WebSocket 会话开启以接收新音频。

```json
{
  "type": "session.update",
  "session": {
    "language": "English",
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5,
      "prefix_padding_ms": 300,
      "silence_duration_ms": 500
    }
  }
}
```

每次周期性解码都是一个针对当前分段全部音频的普通无状态 Qwen3-ASR 请求。在前两次刷新之后，服务器会从先前假设回滚 5 个 token，并把保留的文本用作下一个提示词前缀。不保留解码器 KV cache，也不保留 worker 亲和性。分段同样会在配置的 `audio_chunking.max_audio_clip_s` 边界（默认 30 秒）处结束。

部分结果是完整替换，而不是只追加的增量：

```json
{
  "type": "transcription.segment",
  "event_index": 7,
  "segment_id": 0,
  "text": "hello wor",
  "is_final": false
}
```

同一 `segment_id` 的后续事件会替换这段文本。带 `is_final=true` 的事件是不可变的。`transcription.completed` 包含所有最终分段拼接后的文本。

追加事件不满足幂等性。传输失败后，请重新连接并重启转写，而不是在旧会话上重试数据包。重连会重置未提交的音频、部分假设、VAD 状态以及 Qwen 回滚状态。

## 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `file` | file | 必填 | 以 multipart 表单数据上传的音频文件 |
| `model` | string | 服务器默认值 | 模型标识符 |
| `language` | string | 无 | 可选的语言提示，接受受支持的代码或规范名称（不区分大小写）；省略则自动检测 |
| `prompt` | string | 无 | 词表偏置（biasing）：音频中可能出现的术语，如名称与行话。见表格下方说明 |
| `response_format` | string | `json` | `json`、`verbose_json` 或 `text` |
| `temperature` | float | `0` | 采样温度；`0` 使用贪心解码 |
| `max_new_tokens` | integer | 服务器阶段上限 | 按请求的生成 token 上限 |
| `stream` | boolean | `false` | 返回 SSE 转写增量；支持 `json` 或 `text` 响应格式 |

偏置会提高模型对所提供术语的偏好。它不是强制：音频中不存在的术语不会被插入，而不相关的列表会把模型偏向从未被说出的词，反而损害准确率。简短且相关的列表效果最好——测试中，超过约 20 个术语后准确率不再提升，而延迟持续增长，因为这些文本会随每个请求一起预填充。

`verbose_json` 使用模型适配器的 verbose 响应 schema，并在时长探测成功时包含基于时长的用量（向上取整的音频秒数）。

### 语言提示

省略 `language` 时，Qwen3-ASR 会在转写前检测口语语言。当语言已知，或自动检测对简短、有歧义的音频不可靠时，请设置显式提示。

Qwen3-ASR 接受以下 30 种显式语言代码及其规范名称：

| 代码 | 规范名称 |
|---|---|
| `ar`, `yue`, `zh`, `cs`, `da`, `nl`, `en`, `fil`, `fi`, `fr` | Arabic, Cantonese, Chinese, Czech, Danish, Dutch, English, Filipino, Finnish, French |
| `de`, `el`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mk`, `ms` | German, Greek, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Macedonian, Malay |
| `fa`, `pl`, `pt`, `ro`, `ru`, `es`, `sv`, `th`, `tr`, `vi` | Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Vietnamese |

例如，`language=es` 与 `language=Spanish` 都会强制添加提示词后缀
`language Spanish<asr_text>`。旧式的 `cn` 与地区性 `zh-*` 拼写也按中文接受。不支持的语言提示会返回 HTTP 400，而不是静默回退到英语。

该模型还覆盖 22 种中文方言的 ASR，但这些方言名称不能作为强制的 `language` 提示；请对它们使用 `Chinese`/`zh`。

## 长音频

当前 Qwen3-ASR 模型每个请求最多接受 1,200 秒音频，因此更长的上传会分块转写：我们切分音频，把每块作为独立的引擎请求运行，再按顺序拼回转写结果。行为由两类取值决定。

调度策略可由你用点分参数或对应的 YAML 键调优：

| 名称 | 默认值 | 含义 |
|---|---|---|
| `--audio_chunking.max_audio_clip_s` | `30` | 单个请求发送给引擎的最长音频片段，即分块长度。它有意远低于模型原生的 1,200 秒：更短的块更利于批处理，且输出 token 预算本身随片段长度伸缩。上限为原生片段限制。 |
| `--audio_chunking.max_concurrent_chunks` | `8` | 一个请求的多少块同时在引擎中运行。按请求设限，避免一个长上传挤占其他人的请求。 |
| `--audio_chunking.max_total_audio_s` | `3600` | 整个上传的上限；超过会得到 HTTP 400。这是显存防护：块运行期间解码出的波形会保存在内存中。 |

模型属性是 `Qwen3ASRPipelineConfig` 上的 ClassVar；没有任何配置路径可以触达它们：

| 名称 | 取值 | 含义 |
|---|---|---|
| `allow_audio_chunking` | `true` | Qwen3-ASR 能正确转写孤立分块，因此分块开启。 |
| `max_native_clip_s` | `1200` | 模型单个请求可接受的最长片段（其原生上限）。流式无法分块，因此这也是流式截断点；Torch MPS 兼容路径将其解析为合格的 60 秒上限。 |
| `min_tail_s` | `0.5` | 值得转写的最短末尾块；若尾块更短，我们会把上一个切点提前以吸收它。这与模型自身的最小输入长度一致。 |

注意：调高 `audio_chunking.max_audio_clip_s` 还会改变编码器 CUDA graph 桶阶梯的尺寸，该阶梯由块长度推导：更长的块意味着更多、更大的捕获 graph，而它们的静态缓冲会在服务器整个生命周期内常驻（阶梯上限约为每 token 6.6 KB；1,200 秒时上限是 124,800 token）。在小显存 GPU 上调高该参数时请为此做预算。

行为说明：

- **`verbose_json` 为每个分块返回一个分段**，带该分块真实的起止时间戳——分块级粒度而非词级（Qwen3-ASR 不输出词级时间戳）。
- 少数特殊音频格式可能无法读出时长；这类上传会回退到非分块路径。
- 流式响应（`stream=true`）尚不支持分块；流式请求作为单个引擎请求运行。MLX 与 CUDA 接受最长到模型原生 `max_native_clip_s`（1,200 秒）的音频，而 Torch MPS 接受最长到其合格 60 秒上限，超过返回 HTTP 400——更长的上传请使用 `stream=false`。

## 基准测试

使用 `benchmarks/eval/benchmark_asr_seedtts.py` 在 SeedTTS 参考音频上通过 `/v1/audio/transcriptions` 扫描 ASR 并发。它默认
`--model-path Qwen/Qwen3-ASR-1.7B`；共享的请求与指标逻辑位于
`benchmarks.tasks.asr`，也通过 `--model-path` 支持 Fun-ASR。
报告包含 RTF（处理时间除以音频时长）与 RTFx（成功输入的音频秒数除以墙钟秒数）。

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port 8000

# Sweep the full SeedTTS EN set (1088 clips), 3 repeats per concurrency:
# Set SERVER_GPU_PID to the server process PID reported by nvidia-smi.
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --gpu-process-pid "${SERVER_GPU_PID}" \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --concurrencies 1,2,4,8,16,32,64 \
  --repeats 3 --warmup
```

结果 JSON 包含所应用的数据集 revision、声明的模型 revision、有效的评测输入内容哈希、归一化方式、仓库与依赖指纹、完整的样本计数，以及延迟/RTF/吞吐。当本地 NVML 与 `psutil` 采样可用时，还包含 CPU 使用率、功耗以及峰值/稳态 GPU 显存。请通过 `--gpu-process-pid` 传入 NVML 报告的每个服务器 GPU PID；没有显式 PID 时，进程级指标保持不可用，而不是把同一 GPU 上的无关负载算进来。在 Docker 容器中，请使用宿主 PID 命名空间（`--pid=host`）来采集进程 CPU 指标。不可用的指标与监控错误都会显式保留。可选的服务器设置与确切的启动命令可通过基准测试的 provenance 参数声明。

ASR CI 门禁在同一基准测试入口上运行选定的 ASR CI 模型预设（`tests/test_model/test_asr_ci_seedtts.py`）。Qwen3-ASR 仍是 TTS 与 talker WER 阶段的转写器。

关于当前主线的并发基线、固定基线对比，以及分阶段瓶颈分解（issue #1324），参见
[Qwen3-ASR 并发分析](../developer_reference/qwen3_asr_concurrency_profile.md)。
基准测试的 `--profile-events`、`--sample-util`、`--save-raw-dir` 与
`--fingerprint` 参数可捕获该报告所用的遥测数据。

## 并发调优

请求构建、准入与 CUDA-graph 策略的默认值来自一次实测扫描（issue #1324 Q-PR5）：`request_build_max_workers` {2, 4, 8} ×
`request_build_max_pending` {16, 32, 64} × `max_running_requests` {16, 32, 64}
并配套相应的 CUDA-graph 覆盖，每种配置在一张 141 GB GPU 上做完整的 SeedTTS EN 并发扫描（1–64，三次重复加预热），启用 pre-LM 编码器并禁用其 embedding 缓存（唯一输入场景）。按客户端并发的请求/秒：

| 配置（workers/pending/running） | c=8 | c=16 | c=32 | c=64 | c=64 时被丢弃 |
|---|---:|---:|---:|---:|---:|
| 2 / 16 / 32 | 39.1 | 47.5 | 52.3 | 51.0 | 704/3264 |
| 4 / 16 / 32 | 47.6 | 60.3 | 70.4 | 55.4 | 301/3264 |
| 8 / 16 / 32 | 48.5 | 75.6 | 89.7 | 64.6 | 173/3264 |
| 8 / 16 / 16 | 57.6 | 75.4 | 42.2 | 46.7 | 250/3264 |
| 8 / 32 / 32 | 57.7 | 76.5 | 87.1 | 65.1 | 0 |
| 8 / 64 / 32 | 55.2 | 76.6 | 87.9 | 64.7 | 0 |
| **8 / 32 / 64（默认）** | 57.4 | 77.0 | 90.2 | 96.8 | 0 |
| 8 / 64 / 64 | 57.0 | 74.3 | 88.8 | 100.3 | 0 |

解读与由此得到的默认值：

- **构建 worker 数在所有 ≥8 的并发下单调扩展到 8**，且在并发 1 时零开销（各处均值都是 0.099–0.101 秒），因此默认为 8。这些 worker 执行 CPU 侧请求构建（音频解码、可选的 mel FFT），并异步提交编码工作。当没有额外的构建排队时，请求构建器等待编码完成并像同步路径一样返回就绪请求；当 pending+积压超过 worker 池时，它返回一个延迟准入，让 worker 去拉取积压。缓存命中仍会完全跳过 mel 提取。
- **Pending 16 → 32 消除了并发 64 下的全部丢弃**，并把并发 8 的吞吐提升约 19%；64 没有进一步收益。默认取 32。
- **`max_running_requests` 为 16 会让并发 32 崩塌**（受队列限制），且对轻载延迟毫无益处，因此不存在为降延迟而调低它的理由。默认取 64，因为它解锁并发 64 区间（请求/秒 +约 50%，零丢弃），代价是更大的 CUDA-graph 与 KV 显存。在显存受限的 GPU 上，请使用保守显存的覆盖配置：

```bash
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B \
  --asr.engine.max_running_requests 32
```

- 每种配置在每个水位下的语料级 WER 都保持在 0.0122。

## 已知限制

- HTTP 端点每个请求接受一个上传文件。实时 PCM 使用 `/v1/realtime?intent=transcription` 且需要 `--enable-realtime`。
- 不超过 `max_total_audio_s`（默认一小时）的非流式上传会通过分块完整转写；见上文"长音频"。流式请求在 MLX/CUDA 上限为 `max_native_clip_s`（1,200 秒）；Torch MPS 把原生与整段上传请求都限制在 60 秒。
- 音频在转写前会被重采样到 16 kHz。
