# ARK-ASR-3B

[ARK-ASR-3B](https://huggingface.co/AutoArk-AI/ARK-ASR-3B)（AutoArk-AI，Apache-2.0）
是一个多语言开放 ASR 模型，通过 OpenAI 兼容的 `/v1/audio/transcriptions`
端点提供服务。每个请求接受一个上传的音频文件并返回文本。在架构上，它是一个
Whisper 风格的音频塔（RoPE 自注意力）加一个 MLP 帧合并适配器，后接稠密 Qwen2
LM，因此它与 Qwen3-ASR 运行在同一条单阶段批处理 ASR 流水线上，并复用 SGLang
原生的 Qwen2 解码器。checkpoint 的分词器与配置以 `trust_remote_code=True`
加载（服务的 `ServerArgs` 也会设置它），因此首次启动会提示执行 checkpoint 携带
的代码。

ARK-ASR 不支持 `/v1/audio/translations`；该端点返回 HTTP 400。请使用
`/v1/audio/transcriptions`。

## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`，然后下载模型：

```bash
hf download AutoArk-AI/ARK-ASR-3B
```

## 服务器配置

ARK-ASR 在单块 GPU 上运行一个 ASR 阶段，默认 `bfloat16`。解码 batch 至少两个
请求时默认启用异步解码，使共享的一步前瞻（one-step-lookahead）路径能把宿主侧
结果处理与下一次 GPU 解码前向重叠。使用
`--asr.factory.enable_async_decode false` 禁用，或通过
`--async-lookahead-min-batch-size` 调整切换点。请求并发与音频编码器批处理分开
控制：

- `max_running_requests` 默认为 `32`，限制 ASR 调度器准入的请求数。
- `encoder_max_batch_size` 默认为 `8`，限制一次编码器前向处理的未缓存音频条目
  数。更大的缓存未命中批次会被拆成顺序的编码器微批次，因此请求并发不会直接造成
  无界的编码器 batch。

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --port 8000
```

编码器激活显存在模型权重与 KV cache 之外。对长片段或高并发，请调低 SGLang 的
静态显存比例以留出运行时余量，例如：

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --mem-fraction-static 0.75 \
  --port 8000
```

`mem_fraction_static` 控制模型权重与 KV-cache 池的 SGLang 显存预算；它不能
替代 `encoder_max_batch_size`。两项设置保护服务路径的不同部分。ARK 不覆盖
SGLang 默认的静态显存比例；当长片段或高并发仍需要额外的编码器余量、且缩小的
KV cache 容量可以接受时，使用 `0.75`。

对比模式时若要强制同步解码：

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --asr.factory.enable_async_decode false \
  --port 8000
```

### Prefill 合并

ARK-ASR 默认会短暂扣住新构建的请求，让调度器准入更大的 prefill batch。调优后的
默认值为 16 个请求 / 32 ms：

```bash
sgl-omni serve \
  --model-path AutoArk-AI/ARK-ASR-3B \
  --prefill-coalesce-requests 16 \
  --prefill-coalesce-wait-ms 32 \
  --port 8000
```

达到请求阈值或等待截止时间后即释放准入；当排队的请求构建工作排空时也可能提前
释放。设置 `--prefill-coalesce-requests 0` 可禁用合并。请根据目标请求分布与
延迟要求调整这些值。

## 转写音频

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=AutoArk-AI/ARK-ASR-3B \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "AutoArk-AI/ARK-ASR-3B",
            "language": "en",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## 流式转写

把 multipart 的 `stream` 字段设为 `true` 并保持 `response_format` 为
`json` 或 `text`，即可接收 Server-Sent Events（SSE）：

```bash
curl -N -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=AutoArk-AI/ARK-ASR-3B \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json \
  -F stream=true
```

响应包含零个或多个 `transcript.text.delta` 事件，随后是一个带完整后处理转写的
`transcript.text.done` 事件，最后是 `data: [DONE]`。流式主要降低首文本时间；
它不会改变最终转写。

请把 `transcript.text.done` 当作权威转写：它与非流式的 `text` 字段是同一个
后处理字符串。增量的 `transcript.text.delta` 事件只是实时预览。把它们拼接起来
可能与 `done` 在首尾空白上有差异（最终适配器对完整解码做了 `.strip()`）。请
持久化 `done.text`，而不是 `"".join(deltas)`。

## 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `file` | file | 必填 | 以 multipart 表单数据上传的音频文件 |
| `model` | string | 服务器默认值 | 模型标识符 |
| `language` | string | `en` | 记录在请求上的语言提示。转写指令是固定的英文提示词（`Please transcribe this audio.`）；ARK-ASR 会自动检测口语语言，该字段不会切换提示词模板。 |
| `response_format` | string | `json` | `json`、`verbose_json` 或 `text` |
| `stream` | boolean | `false` | 发出 SSE 文本增量。流式只接受 `json` 或 `text` 响应格式 |
| `temperature` | float | `0`（贪心） | 采样温度。未设置或为 `0` 时，SGLang 的采样归一化选择贪心解码（`top_k=1`）；不会替换为非零温度。 |

### 响应格式

三个 `response_format` 取值返回不同的形状：

- **`text`** — 以 `text/plain` 返回原始转写。
- **`json`** — `{"text": "...", "usage": {"type": "duration", "seconds": <int>}}`。
- **`verbose_json`** — OpenAI 的 verbose 形状，由默认转写适配器构建（整段转写
  作为单个分段）：

  ```json
  {
    "task": "transcribe",
    "language": "en",
    "duration": 12.34,
    "text": "...",
    "segments": [{"id": 0, "start": 0.0, "end": 12.34, "text": "..."}],
    "usage": {"type": "duration", "seconds": 13}
  }
  ```

`language` 与 `usage` 在未知时省略（`exclude_none`）。如果上传音频的时长无法
探测，`duration` 与分段结束时间为 `0.0`。

## 音频处理

- 音频在特征提取前被重采样到 **16 kHz**。
- 特征使用 128 bin 的 log-mel 前端（原版 `WhisperFeatureExtractor`）。
- 特征提取器在 **30 秒的 Whisper 边界**截断（`n_samples = 480000`）；超过
  30 s 的音频被截断为前 30 s。mel 填充为 `"longest"`，因此短片段不必付出完整
  30 s 的 FFT 代价。
- 送入 LM 的音频 token 数为 `(mel_frames + 1) // 2 // merge_factor`，其中
  `merge_factor = 4`（conv2 步长 2 下采样，再做 4 帧合并）。

## dtype

- 默认服务 dtype 为 **`bfloat16`**。这是经过验证的路径：原生音频编码器重新实现
  已在相同 mel 输入上与参考 `transformers` 实现做过一致性检查。
- 也提供 `float16` 路径。fp16 下编码器层会对残差后激活做钳制（与参考
  `modeling_audio.py` 一致），使大激活保持有限；该钳制在 `bfloat16` 下是
  no-op。

## 标记 token 抑制

原版 checkpoint 不带 `bad_words_ids`，因此单纯的 `skip_special_tokens=True`
解码可能在对抗性 / OOD 音频上泄漏非特殊的附加标记（如 `<tool_call>`、
`<|audio|>`）。请求构建器在采样时通过 `logit_bias` 防御性地抑制每个保留标记
（除 EOS 以外的全部 special 与 `<...>` 附加 id），并在解码时剥离它们。这在干净
语音上已验证为 no-op。

## Pre-LM 音频编码器

音频编码运行在 **LM 准入之前**，而不是 LM 前向内部。音频编码器在请求构建时于
专用的 worker 线程与 CUDA 流上执行，只有当一个请求完整的 LM 就绪 embedding
（`MultimodalDataItem.precomputed_embeddings`）挂好之后才会被准入。没有这一步，
每次准入都会让调度器线程在默认流上为整个编码器前向阻塞正在运行的解码 batch。

编码后的 embedding 缓存在一个有界的 CPU LRU 中，键为音频指纹加编码器流水线的
命名空间摘要（checkpoint 路径、含 `merge_factor` 的模型配置、mel 前端字段、
dtype、注意力后端）。其中任何一项变化都会重新生成缓存键，而不是提供过期的
embedding。相同音频的并发请求会被单飞（single-flight）去重，因此片段只编码
一次。

请求构建不等 GPU 就提交编码工作。调度器把已构建的 LM 请求放在等待队列之外，
直到该请求的编码 future 完成，然后在调度器线程上执行正常准入。有界的编码器
队列在 mel 张量无限累积之前施加背压。

| 旋钮 | 默认值 | 含义 |
|---|---|---|
| `enable_pre_lm_encoder` | `true` | 关闭则回退到 LM 前向内编码。 |
| `pre_lm_cache_max_entries` | `4096` | 缓存 embedding 的最大条目数。 |
| `pre_lm_cache_size_bytes` | `2 GiB` | 字节预算；超过后 LRU 淘汰。 |
| `pre_lm_max_batch_size` | `8` | 一次 `get_audio_feature` 调用最多排空多少排队请求。 |
| `pre_lm_max_batch_wait_ms` | `0` | 批组成窗口。`0` 为贪婪排空：上一组编码期间排队的条目立即取走，因此空闲到达的请求不付出批处理延迟。 |
| `pre_lm_max_pending` | `32` | 活跃批次之后等待的编码条目上限。 |

### 与 `encoder_max_batch_size` 的关系

两个批处理旋钮作用于不同层级，且同时生效：

- `pre_lm_max_batch_size` 决定**多少排队请求被交给一次 `get_audio_feature`
  调用**。
- `encoder_max_batch_size` 决定**该调用如何执行**——它对组做填充与掩码，再拆成
  顺序的微批次以约束编码器激活显存。

两者默认都是 `8`，因此一个排空的组恰好是一个编码器微批次。把
`pre_lm_max_batch_size` 提到 `encoder_max_batch_size` 之上会把一个组变成多次
有界前向；它从不加宽单次前向，因此不改变编码器激活显存峰值。

`request_build_max_workers` 默认为 **2**，`request_build_max_pending` 默认为
**16**。这些 worker 只做 CPU 请求构建；编码器并发与背压由独立的 pre-LM 队列
负责。

## 编码器 CUDA Graph

音频编码器 CUDA Graph 默认启用。启动时它捕获从 `encoder_max_batch_size`
推导的 batch 桶（2 的幂加上上限本身；默认 8 给出 `1/2/4/8`）以及按 64 帧步进、
到约 10 s（1024 帧）为止的 mel 帧桶。更长的片段、未捕获的桶以及捕获或 replay
失败都会使用 eager 编码器；请求绝不会触发捕获。

这些 graph 在 SGLang 的生成 CUDA graph 之后、pre-LM 编码器服务之前捕获。要
profile eager 编码器执行：

```bash
sgl-omni serve --model-path AutoArk-AI/ARK-ASR-3B \
  --asr.factory.enable_encoder_cuda_graph false
```

或在流水线配置中：

```yaml
stages:
  asr:
    factory:
      enable_encoder_cuda_graph: false
```

## 基准测试

使用 `benchmarks/eval/benchmark_asr_seedtts.py` 在 SeedTTS 参考音频上通过
`/v1/audio/transcriptions` 扫描 ASR 并发。传入 `--model-path AutoArk-AI/ARK-ASR-3B`；共享的请求与指标逻辑位于 `benchmarks.tasks.asr`。

```bash
# Download the test set once:
python -m benchmarks.dataset.prepare --dataset seedtts

# Launch ARK-ASR:
sgl-omni serve --model-path AutoArk-AI/ARK-ASR-3B --port 8000

# Sweep the full SeedTTS EN set (1088 clips) at 1..64 concurrency, 3 repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path AutoArk-AI/ARK-ASR-3B \
  --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup

# Quick smoke on a 20-sample subset:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path AutoArk-AI/ARK-ASR-3B \
  --max-samples 20 --concurrencies 2 --repeats 1

# Measure text TTFT and inter-chunk latency through the SSE endpoint:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path AutoArk-AI/ARK-ASR-3B \
  --max-samples 20 --concurrencies 2 --repeats 1 --stream
```

脚本按每个并发级别报告语料级 WER、吞吐与延迟。转写准确率与官方
`transformers` checkpoint 在相同音频上一致。

## 已知限制

- 端点每个请求接受一个上传文件。
- HTTP 端点为兼容 OpenAI 而接受 `prompt`，但 ARK-ASR 目前忽略它（转写指令是
  固定的）。
- 音频在转写前被重采样到 16 kHz 并在 30 s 处截断。
