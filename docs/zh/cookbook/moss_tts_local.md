# MOSS-TTS-Local

[MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) 是由 MOSI.AI 和 OpenMOSS 团队开发的文本转语音模型。它与 [MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) 配合生成原生 **48 kHz 立体声** 语音，并支持基于参考音频的零样本语音克隆、无参考合成、长篇语音生成、流式、token 级时长控制、拼音/IPA 发音控制、多语言合成以及语码转换。该模型支持 **31 种语言**，接受语言标签以引导多语言生成，并支持内联停顿标记（如 `[pause 3.2s]`）来实现显式韵律控制。

![MOSS-TTS-Local 架构](../_static/image/moss-tts-arch-local.png)

在架构上，MOSS-TTS-Local-Transformer-v1.5 是 `delay-pattern` 的 [MOSS-TTS-v1.5](moss_tts.md) 的 `local-transformer` 对应版本。它不在时间维度上交错 RVQ 流，而是由 Qwen3-4B 骨干网络为每个对齐的音频帧发出一个全局潜变量，再由一个轻量的帧局部 transformer 将该潜变量扩展为固定的 12-codebook RVQ 块。在 SGLang-Omni 中，它作为 `preprocessing → tts_engine → vocoder` 流水线运行，通过兼容 OpenAI 的 `/v1/audio/speech` 端点提供服务。

| 组件 | 规格 |
|---|---|
| 架构 | `MossTTSLocalModel`（`moss_tts_local`） |
| 骨干网络 | Qwen3-4B 自回归解码器（36 层，hidden=2560，GQA 32/8） |
| 音频 tokenizer | MOSS-Audio-Tokenizer-v2 |
| 音频 token | 固定的 12-codebook RVQ 深度 |
| 输出音频 | 48 kHz 立体声 |
| 语言 | 31 种语言，支持可选语言标签 |
| 控制项 | 声音参考、目标时长 token、拼音/IPA、停顿标记、风格指令 |

## 前置条件

按照[安装](../get_started/installation.md)说明安装 `sglang-omni`，然后下载模型（公开模型，无需 token）：

```bash
hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
```

处理器随 checkpoint 一起提供，因此不需要额外的 TTS 包。解码 base64（data-URI）参考音频还需要 `soundfile`（`uv pip install soundfile`）。

## 服务器配置

默认布局将 AR 骨干网络和编解码器（codec）/声码器（vocoder）放在同一张 GPU 上：

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --port 8000
```

匹配的配置文件位于 `examples/configs/moss_tts_local.yaml`。

语音输入准入遵循文本骨干网络的上下文元数据，而非通用的 4,096 字符预检。超出有效模型上下文的请求会被以兼容 OpenAI 的 HTTP 400 错误拒绝。

### 流式声码器 CUDA Graph

`vocoder_cuda_graph` 流水线设置控制 MOSS-Audio-Tokenizer 流式解码的 CUDA Graph。AR 引擎的 graph 设置在 `tts_engine.engine` 下单独配置。

| 流水线设置 | 默认值 | 用途 |
| --- | --- | --- |
| `vocoder_cuda_graph` | `null` | 使用平台默认值：除 ROCm WSL/DXG 外均启用。设置为 `false` 可使用 eager 流式声码器解码。 |
| `vocoder_cuda_graph_frames` | `null` | 覆盖流式声码器捕获的音频码帧数。 |
| `vocoder_cuda_graph_min_free_gb` | `3.0` | 声码器 graph 捕获前的最小空闲 GPU 显存；`0` 禁用此检查。 |

对于非流式工作负载，在 YAML 中设置 `vocoder_cuda_graph: false` 或传递 `--vocoder_cuda_graph false`，以便为解码留出更多 GPU 显存。完整示例位于 `examples/configs/moss_tts_local_non_streaming.yaml`。

流水线配置和声码器工厂/调度器 API 使用相同的名称。请将 YAML、CLI 覆盖和直接 Python 调用中旧的 `cuda_graph`、`cuda_graph_frames` 和 `cuda_graph_min_free_gb` 拼写替换为带 `vocoder_` 前缀的版本。

## 合成语音

### 基础语音

MOSS-TTS-Local 可以在没有参考剪辑的情况下合成语音：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "voice": "default",
    "input": "SGLang-Omni is a great project!"
  }' \
  --output output.wav
```

### 语音克隆

需要语音克隆时请提供参考剪辑。`references` 字段接受 `audio_path`（本地路径、HTTP URL 或 base64 data URI）和 `text`（该剪辑的文字转写）。提供转写文本可以显著提升克隆质量。

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "voice": "default",
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

`ref_audio` 和 `ref_text` 可作为 `references[0].audio_path` 和 `references[0].text` 的简写形式。

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        "voice": "default",
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his.",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### 参考音频来源

`audio_path` / `ref_audio` 可以是服务器可读取的本地文件系统路径、HTTP(S) URL 或 base64 **data URI**（`data:audio/wav;base64,<...>`，使用 `soundfile` 解码）：

```python
import base64
import requests

reference_url = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
reference_resp = requests.get(reference_url)
reference_resp.raise_for_status()
ref_audio = (
    "data:audio/wav;base64,"
    + base64.b64encode(reference_resp.content).decode("ascii")
)

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        "voice": "default",
        "input": "SGLang-Omni is a great project!",
        "ref_audio": ref_audio,
        "ref_text": "Transcript of the reference clip.",
    },
)
resp.raise_for_status()
with open("output_data_uri.wav", "wb") as f:
    f.write(resp.content)
```

参考编码会被缓存（LRU）并合并为批量的 codec 调用，因此重新发送相同的参考剪辑会跳过重新编码。

### 流式

设置 `"stream": true` 和 `"response_format": "pcm"` 以实时接收原始的 48 kHz 单声道 PCM 块。如果想要可播放的 WAV 文件，请将流通过管道传给 `ffmpeg`：

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "stream": true,
    "response_format": "pcm"
  }' \
  | ffmpeg -f s16le -ar 48000 -ac 1 -i pipe:0 output_stream.wav
```

### 时长控制

MOSS-TTS-Local 以目标**时长 token 数**（codec 帧；数量越大音频越长）为条件。可以在 `input` 上使用内联 `${token:N}` 前缀设置（合成前会被剥离），或使用 `token_count`（别名 `duration_tokens`）参数。该计数必须为正整数。

```json
{
  "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
  "voice": "default",
  "input": "${token:150}A sentence with an explicit duration target.",
  "ref_audio": "..."
}
```

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "voice": "default",
    "input": "${token:150}A sentence with an explicit duration target.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output_duration_tokens.wav
```

如果省略，模型会自行选择时长。

### 文本标记、风格与语言

模型可理解的内联文本标记（例如 `[pause Xs]`、拼音和 IPA）会原样传递。可选的 `instructions` 字段携带自由文本风格指令，可选的 `language` 提示会偏置目标语言（省略它可让模型从文本推断）：

```json
{
  "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
  "voice": "default",
  "input": "今天天气不错 [pause 0.5s] 就该出去晒晒太阳。",
  "ref_audio": "...", "ref_text": "...",
  "language": "Chinese"
}
```

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "voice": "default",
    "input": "今天天气不错 [pause 0.5s] 就该出去晒晒太阳。",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "language": "Chinese",
    "instructions": "Use a natural conversational style."
  }' \
  --output output_markup.wav
```

## 生成参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | 服务的模型 | 所服务模型的标识符 |
| `input` | （必填） | 要合成的文本。可以携带 `${token:N}` 时长前缀和内联标记 |
| `voice` | `default` | 声音标识符 |
| `references` | `null` | 用于克隆的参考剪辑。每项包含 `audio_path` 和 `text` |
| `ref_audio` / `ref_text` | `null` | `references[0].audio_path` / `references[0].text` 的简写 |
| `stream` | `false` | 流式输出原始 PCM 音频块（配合 `response_format: pcm`） |
| `language` | `null` | 可选的目标语言提示。省略以让模型推断 |
| `instructions` | `null` | 可选的自由文本风格指令 |
| `token_count` / `duration_tokens` | `null` | 以 codec 帧数表示的目标时长。必须 `> 0` |
| `max_new_tokens` | `4096` | 最大生成帧数。显式取值必须 `> 0` |
| `temperature` | `1.0` 文本 / `1.7` 音频 | 采样温度。单一 `temperature` 会同时覆盖两个通道 |
| `top_p` | `1.0` 文本 / `0.8` 音频 | Top-p 采样。单一 `top_p` 会同时覆盖两个通道 |
| `top_k` | `50` 文本 / `25` 音频 | Top-k 采样。单一 `top_k` 会同时覆盖两个通道 |
| `repetition_penalty` | `1.0` | 音频重复惩罚 |
| `seed` | `null` | 非负整数。参见[种子可复现性](seed-reproducibility-local) |

这两个默认值反映了模型相互独立的采样通道：`text` 通道是逐帧的继续/停止头，`audio` 通道是 RVQ codebook。请求中单一的 `temperature`、`top_p` 或 `top_k` 会同时作用于两个通道。

(seed-reproducibility-local)=
## 种子可复现性

固定的 `seed` 在**任意并发**下均可复现：每个 token 的采样只取决于它自己的种子和位置，绝不取决于其批次邻居。

- 可复现性仅在**固定的服务器配置和硬件**下成立——骨干网络的浮点非确定性（不同的批次形状、GPU 或 kernel）仍可能在不同部署间改变采样到的 token。
- `seed` 必须是非负整数；负值或非整数值会被拒绝。
- 没有 `seed` 时，每个请求会抽取新的随机种子，在不同运行之间不可复现。

## 基准测试

MOSS-TTS-Local 从每条提示克隆（`--ref-format references`），并使用 `--token-count auto` 估计每个样本的时长。请在 `--max-concurrency 16` 下运行。

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --port 8000 \
    --ref-format references \
    --token-count auto \
    --output-dir results/moss_tts_en \
    --lang en --max-concurrency 16
```

中文切分请使用 `--lang zh`。完整工作流请参阅 `benchmarks/README.md`。

## 评估基准

### Seed-TTS-Eval 参考性能

Seed-TTS-Eval 完整集（EN = 1088，ZH = 2020），2× H100，并发 16，`--token-count auto`。这些是 PR #728 中报告的参考推理性能数据——可复现的参考值，而非 CI 阈值。

| 切分 | 延迟均值 / p95（秒） | RTF 均值 | 吞吐量（req/s） |
|---|---:|---:|---:|
| EN | 1.538 / 1.989 | 0.3682 | 10.355 |
| ZH | — | 0.3306 | 8.62 |

### 多语言语音克隆

我们在公开多语言 TTS 套件和内部语音克隆压力测试集上评估 MOSS-TTS-Local-Transformer-v1.5，涵盖多语言合成、说话人相似度以及高难度说话人稳定性案例。

WER（↓）和 SIM（↑）为宏平均，以百分点报告。`N/A` 表示该基准测试只测说话人相似度，不报告 WER。

| 基准测试 | WER ↓ | SIM ↑ |
|---|---:|---:|
| `Seed-TTS-Eval` (excluding hard-zh) | 2.0350 | 68.9850 |
| `CV3-Eval` | 7.4800 | 61.5871 |
| `MiniMax Multilingual` | 6.3692 | 75.3121 |
| `X Voice` | 20.4787 | 63.0023 |

这些结果使用音频采样参数 `temperature=1.7`、`top_p=0.8` 和 `top_k=25` 测得。在 MOSI.AI 的测试中，`temperature=0.6`、`top_p=0.95`、`top_k=25` 和 `audio_repetition_penalty=1.2` 可能产生更好的质量。

## 已知限制

- **语音克隆取决于参考。** 非克隆语音请省略参考；克隆时请提供转写文本（`text` / `ref_text`）以获得最佳说话人相似度。
- **少见的失控生成。** 少数话语可能循环并一直生成到 `max_new_tokens`；设置 `token_count`（或降低 `max_new_tokens`）可以限制输出长度。
- **时长只是提示。** `${token:N}` / `token_count` 可以引导长度，但不是精确的剪辑时长。
- **可复现性受硬件约束。** 固定的 `seed` 只在相同的服务器配置和 GPU 上可复现；参见[种子可复现性](seed-reproducibility-local)。
