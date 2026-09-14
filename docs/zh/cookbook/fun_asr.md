# Fun-ASR-Nano

[Fun-ASR-Nano](https://arxiv.org/abs/2509.12508) 是一个多语言音频转写模型，通过兼容 OpenAI 的 `/v1/audio/transcriptions` 端点提供服务。每个请求接受一个上传的音频文件并返回文本。

Fun-ASR 不支持 `/v1/audio/translations`；该端点会返回 HTTP 400。请使用 `/v1/audio/transcriptions`。

## 前置条件

按照[安装](../get_started/installation.md)说明安装 `sglang-omni`，然后下载模型：

```bash
# Use the -hf variant
hf download FunAudioLLM/Fun-ASR-Nano-2512-hf
```

## 服务器配置

Fun-ASR-Nano 在单张 GPU 上运行单个 ASR 阶段。

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
  --port 8000
```

## 转写音频

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=FunAudioLLM/Fun-ASR-Nano-2512-hf \
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
            "model": "FunAudioLLM/Fun-ASR-Nano-2512-hf",
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

将 multipart 的 `stream` 字段设置为 `true`，并将 `response_format` 保持为 `json` 或 `text`，即可接收服务器推送事件（Server-Sent Events，SSE）：

```bash
curl -N -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=FunAudioLLM/Fun-ASR-Nano-2512-hf \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json \
  -F stream=true
```

响应包含零个或多个 `transcript.text.delta` 事件，随后是一个携带完整后处理转写文本的 `transcript.text.done` 事件，最后是 `data: [DONE]`。流式主要降低首文本时间；它不会改变最终转写结果。

## 请求参数

| 参数 | 类型 | 默认值 | 描述 |
|---|---|---|---|
| `file` | 文件 | 必填 | 以 multipart 表单数据形式上传的音频文件 |
| `model` | 字符串 | 服务器默认值 | 模型标识符 |
| `language` | 字符串 | 未设置 | 语言提示。`en`/`english`/`英文` 转写为英文；`zh`/`cn`/`chinese`/`中文`（或未设置）转写为中文；其他值直接作为目标语言传入 |
| `response_format` | 字符串 | `json` | `json`、`verbose_json` 或 `text` |
| `stream` | 布尔值 | `false` | 输出 SSE 文本增量。流式只接受 `json` 或 `text` 响应格式 |
| `temperature` | 浮点数 | `0.0` | 采样温度；`0.0`（贪心）是 Fun-ASR-Nano 正确的解码模式，也是默认值 |
| `max_new_tokens` | 整数 | 基于时长 | 按音频时长缩放的生成预算。显式取值必须在 1 到 200 之间 |

## 基准测试

Fun-ASR-Nano 的 SeedTTS EN/ZH 并发/WER 基准测试位于 `benchmarks/eval/benchmark_asr_seedtts.py`。通过 `--model-path` 传入 Fun-ASR-Nano 模型路径。

```bash
# Download the test set once:
python -m benchmarks.dataset.prepare --dataset seedtts

# Launch Fun-ASR-Nano:
sgl-omni serve --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000

# Sweep the full SeedTTS EN set (1088 clips) at 1..64 concurrency, 3 repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --concurrencies 1,2,4,8,16,32,64 --repeats 3

# Quick smoke on a 20-sample subset:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --max-samples 20 --concurrencies 2 --repeats 1

# Measure text TTFT and inter-chunk latency through the SSE endpoint:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --max-samples 20 --concurrencies 2 --repeats 1 --stream
```

## 基准测试结果

在单张 H100 80 GB（bf16，DP=1）上针对完整 SeedTTS 集进行测量，使用预合并（pre-coalescing）阶段的默认值（`max_running_requests=32`、`request_build_max_pending=16`，prefill 合并关闭）。每行是 3 次运行的均值，每个级别丢弃一次预热运行。RTF 是处理时间除以音频时长（越低越好）。RTFx 是成功处理的输入音频秒数除以墙钟秒数（越高越好）。

SeedTTS EN（1088 条剪辑，平均剪辑长度 4.69 秒）。在直至并发 32 的每个级别上，语料库 WER 均为 0.0171：

| 并发 | 吞吐量（样本/秒） | 平均延迟（秒） | p95 延迟（秒） | RTF 均值 | RTFx |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.44 | 0.038 | 0.047 | 0.0082 | 124 |
| 2 | 42.55 | 0.047 | 0.058 | 0.0102 | 200 |
| 4 | 62.35 | 0.064 | 0.088 | 0.0139 | 293 |
| 8 | 90.24 | 0.088 | 0.121 | 0.0192 | 423 |
| 16 | 127.46 | 0.125 | 0.167 | 0.0270 | 598 |
| 32 | 127.44 | 0.249 | 0.334 | 0.0539 | 598 |
| 64 | 137.98 | 0.453 | 0.542 | 0.0988 | 647 |

SeedTTS ZH（2020 条剪辑，平均剪辑长度 4.68 秒）。语料库 WER（归一化后实际为字符级）在直至并发 32 的每个级别上均为 0.0135：

| 并发 | 吞吐量（样本/秒） | 平均延迟（秒） | p95 延迟（秒） | RTF 均值 | RTFx |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.96 | 0.037 | 0.048 | 0.0080 | 126 |
| 2 | 45.97 | 0.043 | 0.056 | 0.0094 | 215 |
| 4 | 58.28 | 0.069 | 0.093 | 0.0148 | 273 |
| 8 | 79.76 | 0.100 | 0.138 | 0.0216 | 373 |
| 16 | 138.23 | 0.116 | 0.160 | 0.0249 | 647 |
| 32 | 167.42 | 0.190 | 0.264 | 0.0410 | 784 |
| 64 | 165.75 | 0.381 | 0.475 | 0.0825 | 776 |

单个 worker 在请求构建积压队列填满后会按设计以 HTTP 500 丢弃请求；当前默认每个 worker 最多允许 32 个待处理构建。在上面的预合并默认值（16 个待处理构建）下，并发 64 时大约损失 2% 到 5% 的请求。Qwen3-ASR 在该级别也表现出相同的丢弃行为。若客户端并发更高，请在 DP=2 托管路由器之后提供服务，与 ASR CI 拓扑保持一致。

## 已知限制

- 该端点每个请求接受一个上传文件。
- 每段上传的音频不得超过 30 秒，与官方 Fun-ASR VAD 分段限制一致。上传前请切分较长的录音。
- `prompt` 携带上下文偏置：一个逗号分隔的、很可能出现在音频中的术语列表（名称、行话）。每个术语都会成为模型提示中的一个热词。偏置会提高模型对所提供术语的偏好；它不会强制使用这些术语，而且不相关的列表可能损害准确率。
- `itn` 和显式 `hotwords` 仍然对请求构建器的进程内调用方可用；显式 `hotwords` 优先于 `prompt`。
- 音频在转写前会被重采样到 16 kHz。
- 强烈建议使用 bf16；fp16 在适配器路径中可能溢出为 NaN。
