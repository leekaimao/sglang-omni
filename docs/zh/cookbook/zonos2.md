# ZONOS2

[ZONOS2](https://huggingface.co/Zyphra) 是 Zyphra 的混合专家（MoE）文本转语音模型。一个 MoE 自回归解码器按 **delay pattern** 调度预测 **9 个 DAC 音频 codebook**；随后这些 code 由 DAC 声码器（vocoder）解码回 **44.1 kHz** 语音。它能从一小段参考音频克隆音色，并支持可选的按语言文本归一化。在 SGLang-Omni 中，它以 `preprocessing → speaker_encode → tts_engine → vocoder` 流水线运行，并通过 OpenAI 兼容的 `/v1/audio/speech` 端点提供服务。

## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`。

ZONOS2 使用 Descript DAC 编解码器，它并不包含在基础的 `sglang-omni` 包中。请在 SGLang-Omni 仓库根目录下安装其模型专属依赖：

```bash
uv pip install \
  "descript-audiotools==0.7.2" \
  "descript-audio-codec==1.0.0"
```

然后下载模型：

```bash
hf download Zyphra/zonos2
```

处理器随 checkpoint 一起发布。语音克隆会使用 **ffmpeg** 对参考音频（文件、URL 或 base64 data-URI）转码，因此服务器的 `PATH` 中必须有 `ffmpeg`（例如 `apt-get install ffmpeg`）。

## 服务器配置

流水线为 `preprocessing → speaker_encode → tts_engine → vocoder`。

ZONOS2 自带 `params.json`，其 `model_type`（`zonos2`）会自动选择
`Zonos2ForCausalLM` 架构，因此 `serve` 只需要 `--model-path`——无需
`--config`（与 Higgs 一致）。

```bash
sgl-omni serve \
  --model-path Zyphra/zonos2 \
  --port 8000
```

## 语音合成

### 语音克隆

ZONOS2 从参考音频片段克隆音色。`references` 字段接受 `audio_path`
（本地路径、HTTP URL 或 base64 data URI）和 `text`（该片段的转写文本）。
共享 speech API 接受该转写文本，但 ZONOS2 目前只以参考音频为条件。

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

也接受 `ref_audio` 与 `ref_text` 作为 `references[0].audio_path` 与
`references[0].text` 的简写。`ref_text` 目前未被 ZONOS2 使用。

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
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

`audio_path` / `ref_audio` 可以是服务器可读的本地文件系统路径、HTTP(S)
URL，或 base64 **data URI**（`data:audio/wav;base64,<...>`，经 `ffmpeg` 转码）：

```json
{"ref_audio": "data:audio/wav;base64,UklGR.....", "ref_text": "Transcript of the clip."}
```

### 流式输出

设置 `"stream": true` 并配合 `"response_format": "pcm"` 可接收原始的有符号 16 位小端 PCM 字节。响应不是 SSE 也不是 base64。ZONOS2 返回 44.1 kHz 单声道音频，`Content-Type` 为 `audio/pcm`，并带有 `X-Sample-Rate`、`X-Channels` 与 `X-Bit-Depth` 响应头。

```bash
curl -sS -D output.headers -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "response_format": "pcm",
    "stream": true
  }' \
  --output output.pcm

ffmpeg -f s16le -ar 44100 -ac 1 -i output.pcm output.wav
```

### 语言

`language` 在提示词分词之前选择 NeMo 的"书面转口语"归一化器。支持归一化的语言有
`English`、`Chinese`、`Japanese`、`Korean`、`German`、`French`、
`Portuguese`、`Spanish` 与 `Italian`。省略该字段或使用 `Auto` 可保持输入不变；
没有 NeMo 包的已接受语言（如 `Russian`）同样原样通过。模型会从处理后的文本推断口语语言。

```json
{
  "input": "今天天气不错，就该出去晒晒太阳。",
  "ref_audio": "...", "ref_text": "...",
  "language": "Chinese"
}
```

## 生成参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `input` | （必填） | 要合成的文本 |
| `references` | `null` | 用于克隆的参考音频；`text` 被接受但当前未使用 |
| `ref_audio` / `ref_text` | `null` | 简写字段；`ref_text` 当前未使用 |
| `response_format` | `wav` | `stream=true` 时请使用 `pcm` |
| `stream` | `false` | 流式返回原始 44.1 kHz 单声道 PCM16 字节；要求 `response_format=pcm` |
| `language` | `null` | 可选的 NeMo 文本归一化语言；`Auto` 保留原始文本 |
| `max_new_tokens` | （模型默认值） | 生成帧数上限；显式取值必须 `> 0` |
| `temperature` | （模型默认值） | 采样温度 |
| `top_p` | （模型默认值） | Top-p 采样 |
| `top_k` | （模型默认值） | Top-k 采样 |
| `min_p` | （模型默认值） | Min-p 采样 |
| `repetition_penalty` | （模型默认值） | 音频重复惩罚 |

## 基准测试

ZONOS2 从每条提示词克隆音色（`--ref-format references`）。对运行中的服务器执行 seed-tts-eval 语音克隆基准测试：

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Zyphra/zonos2 --port 8000 \
  --ref-format references \
  --output-dir results/zonos2_en --lang en --max-concurrency 16
```

中文子集使用 `--lang zh`。完整流程参见 `benchmarks/README.md`。

## 基准测试结果

Seed-TTS-Eval **完整集**（EN 1088 / ZH 2020）、1× H100、并发 16、冷启动全程计时、
`--ref-format references`、`fp8 + frame_graph + async_decode + compile_sampler`。使用
**Qwen3-ASR-1.7B** 评分（CI 评分器；EN 词级 WER，ZH 字级 CER）。**所有配置的 WER/CER 中位数均为 0%**
——下面的语料级数字被少数未设 seed 采样导致的灾难性崩溃样本抬高（EN ≤3/1088，ZH 9–16/2020），因此配置间的语料级差异属于运行噪声而非配置差异（剔除这些样本后，语料级约为 EN ~1.3% / ZH ~1.4–1.7%）。所有 EN 配置都通过 <2% 门槛。

| 配置 | WER（语料级） | RTF 均值 | 吞吐（qps） |
|---|---|---|---|
| 非流式 | 1.7% | 0.69 | 6.2 |
| 流式，`stream_emit_chunk_frames=1` | 1.4% | 0.92 | 4.6 |
| 流式，自适应 `24→32`（历史基准） | 1.3% | **0.77** | **5.6** |
| 流式，双 GPU（`multi_gpu`） | 1.5% | 0.69 | 6.2 |

### 流式吞吐（`stream_emit_chunk_frames`）

流式模式下，AR 引擎通过进程内队列把采样出的帧推送给声码器。在稳态下，引擎**把 `stream_emit_chunk_frames=32` 帧合并为一条消息**，而不是每帧一次 `put()`；逐帧的 put 运行在 resolve 宿主循环上，会与下一次解码 launch 串行化，因此把它们批处理化可带来 **−17% 的流式 RTF 和 +21% 的吞吐，且不影响 WER**（与逐帧路径相比；块间延迟也从 0.37 s 降至 0.28 s）。这些测量使用的是上表所示的历史 `24→32` 生产者节奏。

当前对连续性安全的默认值在首条生产者消息中发送 58 行延迟 code：
声码器 40 帧的初始块加上其共享的 18 行去剪切/EOS 前瞻。后续生产者消息回到 32 行。显式的请求级 `initial_codec_chunk_frames` 覆盖会重新计算首个生产者边界；请求值 `0` 选择声码器稳定的 40 帧块。独立的 `stream_emit_first_chunk_frames=0` 流水线设置会禁用自适应生产者边界，并从一开始就使用稳定的 32 行消息大小。逐帧流式请设置 `stream_emit_chunk_frames=1`。双 GPU `multi_gpu` 流水线（codec + speaker encoder 在 `cuda:1`）可叠加使用，实测最佳流式 RTF（约 0.69）。

> ZH（`--lang zh`，完整 2020 条）——Qwen3-ASR 语料级 CER 1.8–3.0%（中位数 0%；语料数字反映的是 9–16/2020 的灾难性样本尾部，而非配置差异——剔除后约 ~1.4–1.7%）。
> 非流式 RTF 约 0.62–0.64，流式约 0.65–0.72；在 `on_stream_done` 尾部修复之后，合并发送对音频无影响。

## 已知限制

- **不使用参考转写文本。** 共享 API 接受 `text` / `ref_text`，但 ZONOS2 目前只以参考音频为条件。
- **语言只控制文本归一化。** 它不会添加独立的模型侧语言条件。
- **不支持按请求设置 seed。** 在采样具备隔离的请求 RNG 之前，带 `seed` 的请求会被拒绝。
- **偶发生成失控。** 少数语句可能循环生成直到 `max_new_tokens`；调低 `max_new_tokens` 可以限制输出。
