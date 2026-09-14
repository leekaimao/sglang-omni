# 音频翻译

本指南以 [Whisper ASR](../cookbook/whisper_asr.md) 为例，介绍 OpenAI 兼容的 `POST /v1/audio/translations` 端点，该端点将源语音翻译为英语。该端点仅由声明了音频翻译能力的流水线提供；其他 ASR 模型会返回明确的 HTTP 400，而不是静默地转写。

## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`，然后下载一个多语言 Whisper checkpoint：

```bash
hf download openai/whisper-large-v3
```

翻译需要多语言、非 turbo 的 checkpoint：`*.en` 系列 checkpoint 没有翻译任务，`whisper-large-v3-turbo` 在蒸馏时未包含该任务，因此服务这些 checkpoint 时无论调用哪个端点都只会返回转写结果。

## 启动服务器

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

## 翻译音频

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F language=fr \
  -F response_format=json
```

`language` 是可选的源语言提示，属于 SGLang-Omni 的扩展；OpenAI 官方的音频翻译 schema 并未定义该参数，且两个 API 的翻译目标语言都是英语。

## 模型支持情况

| 模型 | `/v1/audio/translations` |
|---|---|
| [Whisper ASR](../cookbook/whisper_asr.md) | 支持 |
| [Qwen3-ASR](../cookbook/qwen3_asr.md) | HTTP 400 |
| [Fun-ASR](../cookbook/fun_asr.md) | HTTP 400 |
| [ARK-ASR](../cookbook/arkasr.md) | HTTP 400 |
| [MOSS-Transcribe-Diarize](../cookbook/moss_transcribe_diarize.md) | HTTP 400 |

各模型的转写工作流请参见其对应的 cookbook 文档。

## 响应格式

| `response_format` | 行为 |
|---|---|
| `json` | 包含 `text` 的 JSON 对象 |
| `verbose_json` | 包含 `task="translate"`、文本、时长与分段的 JSON |
| `text` | 以 `text/plain` 内容类型返回的原始翻译文本 |
| `srt`, `vtt` | 带模型分段时间戳的 Whisper 字幕 |

`verbose_json` 返回覆盖整个音频时长的单一分段，与 `/v1/audio/transcriptions` 保持一致。字幕格式要求模型适配器提供真实的分段时间戳；不支持的模型会在推理之前返回 HTTP 400。

设置 `stream=true` 时，`json` 与 `text` 格式支持流式输出，其 SSE 生命周期与转写一致：先是一系列 `transcript.text.delta` 事件，然后是一个 `transcript.text.done` 事件，最后是 `data: [DONE]`。
