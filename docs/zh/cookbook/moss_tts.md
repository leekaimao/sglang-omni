# MOSS-TTS

[MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) 是由 MOSI.AI 和 OpenMOSS 团队开发的 delay-pattern 文本转语音模型。它重建 **24 kHz** 语音，并支持基于参考音频的零样本语音克隆、无参考合成、长篇语音生成、流式、token 级时长控制、拼音/IPA 发音控制、多语言合成以及语码转换。该模型支持 **31 种语言**，接受语言标签以引导多语言生成，并支持内联停顿标记（如 `[pause 3.2s]`）来实现显式韵律控制。

![MOSS-TTS delay-pattern 架构](../_static/image/moss-tts-arch-delay.png)

在架构上，MOSS-TTS-v1.5 是 [MOSS-TTS-Local-Transformer-v1.5](moss_tts_local.md) 的 `delay-pattern` 对应版本。Qwen3-8B 骨干网络预测一条文本流以及 32 个采用 delay-pattern 调度的残差矢量量化（RVQ）音频 codebook；生成的码经过去延迟处理后，由声码器（vocoder）重建为波形。在 SGLang-Omni 中，它作为 `preprocessing → tts_engine → vocoder` 流水线运行，通过兼容 OpenAI 的 `/v1/audio/speech` 端点提供服务。

| 组件 | 规格 |
|---|---|
| 架构 | `MossTTSDelayModel`（`moss_tts_delay`） |
| 骨干网络 | Qwen3-8B 自回归解码器（36 层，hidden=4096，GQA 32/8） |
| 音频 token | 采用 delay-pattern 调度的 32-codebook RVQ 深度 |
| 输出音频 | 24 kHz |
| 语言 | 31 种语言，支持可选语言标签 |
| 控制项 | 声音参考、目标时长 token、拼音/IPA、停顿标记、风格指令 |

## 前置条件

按照[安装](../get_started/installation.md)说明安装 `sglang-omni`，然后下载模型（公开模型，无需 token）：

```bash
hf download OpenMOSS-Team/MOSS-TTS-v1.5
```

处理器随 checkpoint 一起提供，因此不需要额外的 TTS 包。解码 base64（data-URI）参考音频还需要 `soundfile`（`uv pip install soundfile`）。

## 服务器配置

流水线为 `preprocessing → tts_engine → vocoder`。默认情况下，声码器在独立进程中运行（三个阶段的 GPU 显存占比为 0.10 / 0.72 / 0.18）：其 Python 解码循环不再与 AR 调度器共享解释器，这在 H200 上使单副本吞吐量在每个并发上限下都提升约 70%。`config_cls: MossTTSSingleProcessPipelineConfig` 可恢复单进程布局；受限的 24 GB 和 32 GB 配置保留该布局。

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config examples/configs/moss_tts.yaml \
  --port 8000
```

默认的模型专属布局将仓库本地的编码器和声码器组件放置在配置的流水线 GPU 上。编码器和解码器权重根据 `compute_dtype` 以 BF16 实例化；量化器和显式 FP32 归一化层保持 FP32。每个阶段只构建它所使用的编解码器（codec）组件，而不是加载完整的编解码器副本。

受限的 24 GB 和 32 GB 配置显式地将预处理移至 CPU。除非覆盖 `compute_dtype`，它们保持 BF16 计算。

语音输入准入遵循文本骨干网络的上下文元数据，而非通用的 4,096 字符预检。超出有效模型上下文的请求会被以兼容 OpenAI 的 HTTP 400 错误拒绝。

如需受限的 32 GB 资格认证布局，请使用：

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config examples/configs/moss_tts_32gb.yaml \
  --port 8000
```

该配置将请求并发和 CUDA Graph 捕获限制为批大小 1。它在一张显存为 32,607 MiB 的 RTX 5090 上完成了启动和无参考非流式合成，峰值占用 26,939 MiB。请将其视为一个实测的资格认证点，而不是对所有 32 GB 显卡的普遍结论。

如需直接实测的 24 GB 布局，请使用：

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config examples/configs/moss_tts_24gb.yaml \
  --port 8000
```

该配置在一张显存为 24,564 MiB 的 RTX 4090 上完成了启动、CUDA Graph 捕获、有参考和无参考合成、流式、取消和恢复。采样的显存峰值为 23,251 MiB，最少剩余 960 MiB 空闲。SGLang 分析得到的有效 KV 容量为 6,708 token，低于请求的 `max_total_tokens: 8192`。请将其视为一个并发为 1、CUDA Graph 上限为 1 的资格认证点，而不是更广泛的 24 GB 容量结论。

在对其他布局进行分析时，可以通过预处理/声码器阶段的 `factory.*` 条目显式更改这两个策略。显式 `device` 值优先于按阶段放置选择的 GPU。

## 合成语音

### 基础语音

MOSS-TTS 可以在没有参考剪辑的情况下合成语音：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
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
    "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
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
        "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
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

```json
{"ref_audio": "data:audio/wav;base64,UklGR.....", "ref_text": "Transcript of the clip."}
```

### 流式

设置 `"stream": true` 和 `"response_format": "pcm"` 以实时接收原始 PCM 音频块。

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

### 时长控制

MOSS-TTS 以目标**时长 token 数**（codec 帧；数量越大音频越长）为条件。可以在 `input` 上使用内联 `${token:N}` 前缀设置（合成前会被剥离），或使用 `token_count`（别名 `duration_tokens` / `tokens`）参数。该计数必须为正整数。

```json
{
  "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
  "voice": "default",
  "input": "${token:150}A sentence with an explicit duration target.",
  "ref_audio": "..."
}
```

如果省略，模型会自行选择时长；SeedTTS 基准测试通过 `--token-count auto` 为每个样本估计一个时长。

### 文本标记、风格与语言

模型可理解的内联文本标记（例如 `[pause Xs]`、拼音和 IPA）会原样传递。可选的 `instructions`（别名 `instruct`）字段携带自由文本风格指令，可选的 `language` 提示会偏置目标语言（省略它可让模型从文本推断）：

```json
{
  "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
  "voice": "default",
  "input": "今天天气不错 [pause 0.5s] 就该出去晒晒太阳。",
  "ref_audio": "...", "ref_text": "...",
  "language": "Chinese",
}
```

## 生成参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | 服务的模型 | 所服务模型的标识符 |
| `input` | （必填） | 要合成的文本。可以携带 `${token:N}` 时长前缀和内联标记 |
| `voice` | `default` | 声音标识符 |
| `references` | `null` | 用于克隆的参考剪辑。每项包含 `audio_path` 和 `text` |
| `ref_audio` / `ref_text` | `null` | `references[0].audio_path` / `references[0].text` 的简写 |
| `stream` | `false` | 流式输出原始 PCM 音频块 |
| `language` | `null` | 可选的目标语言提示。省略以让模型推断 |
| `instructions` / `instruct` | `null` | 可选的自由文本风格指令 |
| `token_count` / `duration_tokens` / `tokens` | `null` | 以 codec 帧数表示的目标时长。必须 `> 0` |
| `max_new_tokens` | `4096` | 最大生成帧数。显式取值必须 `> 0` |
| `temperature` | `1.5` 文本 / `1.7` 音频 | 采样温度。单一 `temperature` 会同时覆盖两个通道 |
| `top_p` | `1.0` 文本 / `0.8` 音频 | Top-p 采样。单一 `top_p` 会同时覆盖两个通道 |
| `top_k` | `50` 文本 / `25` 音频 | Top-k 采样。单一 `top_k` 会同时覆盖两个通道 |
| `repetition_penalty` | `1.0` | 音频重复惩罚 |
| `seed` | `null` | 非负整数。参见[种子可复现性](seed-reproducibility) |

也接受按通道的字段（`text_temperature`、`audio_temperature`、`text_top_p`、`audio_top_p`、`text_top_k`、`audio_top_k`、`audio_repetition_penalty`），它们优先于单值别名。

(seed-reproducibility)=
## 种子可复现性

MOSS-TTS 使用 `multinomial_with_seed` 对每一行、每个位置和每个 codebook 进行采样，从公开的 `seed` 派生每个请求的种子，并将其与（步骤，通道）级别的位置组合。因此，采样出的 token 只取决于它自己的种子和位置——绝不取决于其批次邻居——所以固定的 `seed` 在任意并发下都可复现，而不仅限于批大小为 1。限制：

- 可复现性仅在**固定的服务器配置和硬件**下成立。骨干网络的浮点非确定性（不同的批次形状、GPU 型号或 kernel）仍可能在不同部署间改变 logits，从而改变采样到的 token。
- `seed` 必须是非负整数；非整数或负值会被拒绝。
- **没有** `seed` 的请求会为每个请求抽取新的随机种子，因此它们在不同运行之间不可复现（但仍与批次邻居相互独立）。

## 基准测试

MOSS-TTS 从每条提示克隆（`--ref-format references`），并使用 `--token-count auto` 估计每个样本的时长。请在 `--max-concurrency 8` 下运行；更高的并发会使 WER 恶化。

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model OpenMOSS-Team/MOSS-TTS-v1.5 --port 8000 \
  --ref-format references --token-count auto \
  --output-dir results/moss_tts_en --lang en --max-concurrency 8
```

中文切分请使用 `--lang zh`。完整工作流请参阅 `benchmarks/README.md`。

## 基准测试结果

Seed-TTS-Eval 完整集（EN = 1088，ZH = 2020），1× H200，并发 8，`--token-count auto`。WER 使用 HF Whisper-large-v3（EN）/ FunASR paraformer-zh（ZH）评分。这些是记录在 `benchmarks/eval/benchmark_tts_seedtts.py` 中的参考数据（来源：PR #609）——可复现的参考值，而非 CI 阈值。

| 语言 | WER（语料库） | WER（排除 >50%） | 延迟均值 / p95（秒） | RTF 均值 | 吞吐量（qps） |
|---|---|---|---|---|---|
| EN | 1.68% | 1.32% (4 outliers) | 3.449 / 4.141 | 0.811 | 2.312 |
| ZH | 1.36% | 1.27% (2 outliers) | 3.608 / 4.153 | 0.635 | 2.213 |

少数话语会失控进入重复循环（WER > 50%）并主导原始微平均；排除它们之后，两种语言的语料库 WER 均约为 1.3%，每样本 WER 中位数为 0.00%。

## 已知限制

- **语音克隆取决于参考。** 非克隆语音请省略参考；克隆时请提供转写文本（`text` / `ref_text`）以获得最佳说话人相似度。
- **并发与 WER 的权衡。** 质量在 `--max-concurrency 8` 附近最佳；更高的并发会使 WER 恶化。
- **少见的失控生成。** 少数话语可能循环并一直生成到 `max_new_tokens`；设置 `token_count`（或降低 `max_new_tokens`）可以限制输出长度。
- **时长只是提示。** `${token:N}` / `token_count` 可以引导长度，但不是精确的剪辑时长。
