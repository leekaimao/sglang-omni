# TTS 模型使用方法

本指南以 [Fish Speech S2-Pro](https://huggingface.co/fishaudio/s2-pro) 为例。同一个 `/v1/audio/speech` 端点也可服务于 Higgs TTS、Voxtral TTS、Qwen3-TTS、Ming-Omni-TTS、MOSS-TTS、MOSS-TTS Local、dots.tts 和 ZONOS2。

## 前置条件

按照[安装](../get_started/installation.md)说明安装 `sglang-omni`，然后下载模型：

```bash
hf download fishaudio/s2-pro
```

Fish Audio 需要其模型特定的 DAC 依赖。在启动服务器之前，请先完成
[Fish Audio S2-Pro 前置条件](prerequisites)。

Qwen3-TTS 使用上游的 `qwen-tts` 包。请在安装时不带依赖，
以确保 SGLang-Omni 的 Transformers 5.12 / SGLang 0.5.19 技术栈保持不变：

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

两行都必须使用 `--no-deps`。`qwen-tts` 0.1.1 将 Transformers 固定在 4.57.3，
如果让它安装该固定版本，会替换掉 SGLang-Omni 其余部分所依赖的技术栈；正常解析
`sox` 会把 `numpy` 拉升到超过 `numba==0.65.1` 设定的上限，这会破坏
`librosa`，进而导致 `import qwen_tts` 失败。SGLang-Omni 在
`sglang_omni/models/qwen3_tts/compat.py` 中对两个 Transformers 版本之间的
API 差异做了适配（shim），因此被固定的 5.12 技术栈才是受支持的配置——详情参见
[Qwen3-TTS cookbook](../cookbook/qwen3_tts.md)。

## 支持的 TTS 模型

| 模型系列 | 示例配置 | 请求说明 |
|---|---|---|
| [Fish Speech S2-Pro](../cookbook/fishaudio_s2_pro.md) | `examples/configs/s2pro_tts.yaml` | 支持普通 TTS 以及通过 `references` 进行语音克隆 |
| [Voxtral TTS](../cookbook/voxtral_tts.md) | `examples/configs/voxtral_tts.yaml` | 使用 `input`、`voice`、`response_format` 和 `max_new_tokens`。进行 SeedTTS 基准测试时使用 `--no-ref-audio` |
| [Qwen3-TTS Base](../cookbook/qwen3_tts.md) | `examples/configs/qwen3_tts_0_6b.yaml`, `examples/configs/qwen3_tts_1_7b.yaml` | 需要通过 `ref_audio` 或 `references[0].audio_path` 提供参考音频。`language` 默认为 `auto` |
| [Qwen3-TTS CustomVoice](customvoice-checkpoints) | `examples/configs/qwen3_tts_0_6b_customvoice.yaml`, `examples/configs/qwen3_tts_1_7b_customvoice.yaml` | 使用内置说话人进行纯文本合成；省略 `voice` 即为 Vivian。两种规模均支持流式；指令控制请使用 1.7B |
| [Qwen3-TTS VoiceDesign](../cookbook/qwen3_tts.md) | `examples/configs/qwen3_tts_1_7b_voicedesign.yaml` | 需要 `task_type="VoiceDesign"` 和非空的 `instructions`。无需参考音频 |
| [Ming-Omni-TTS](../cookbook/ming_tts.md) | `examples/configs/ming_omni_tts.yaml` | 纯文本合成，或一条本地参考音频剪辑及其转录文本；支持流式；所提供的配置使用 TP1 |
| [Fun-CosyVoice3](../cookbook/fun_cosyvoice3.md) | 仅 `--model-path` | 需要通过 `ref_audio` 或 `references` 提供一条参考音频剪辑。支持零样本克隆、跨语言、instruct 模式、因果流式以及带缓冲的语速控制 |
| [MOSS-TTS](../cookbook/moss_tts.md) | `examples/configs/moss_tts.yaml` | 通过 `ref_audio` 或 `references[0].audio_path`（+ `text`）进行语音克隆。通过 `${token:N}` 或 `token_count` 控制时长。基准测试请使用 `--max-concurrency 8` |
| [MOSS-TTS Local](../cookbook/moss_tts_local.md) | `examples/configs/moss_tts_local.yaml` | 48 kHz 立体声 local-transformer MOSS-TTS；语音克隆 / 无参考；流式 |
| [Higgs TTS](../cookbook/higgs_tts.md) | 仅 `--model-path` | 语音克隆、流式；无需示例 YAML |
| [dots.tts](../cookbook/dots_tts.md) | `examples/configs/dots_tts.yaml`（MeanFlow）、`examples/configs/dots_tts_soar.yaml`（SOAR） | 48 kHz 连续潜变量（continuous-latent）TTS，需参考音频。MeanFlow（`dots.tts-mf`）使用连续批处理（continuous batching）（默认 `max_running_requests=16`），配合引擎级 `num_steps=4` 和 Euler。SOAR（`dots.tts-soar`）和 base（`dots.tts-base`）属于 flow matching，在 `max_running_requests=1` 下运行带 CFG 的单请求求解器；两者都使用 SOAR 配置。所有变体都要求 `ref_audio` + `ref_text`。仅支持 TP1 |
| [ZONOS2](../cookbook/zonos2.md) | `--model-path Zyphra/zonos2` | MoE TTS，9 个 DAC codebook，语音克隆；需要 Descript DAC 附加依赖（见 cookbook） |
| [AuK](../cookbook/auk.md) | 仅 `--model-path` | `tencent/AuK` 和 `tencent/AuK-Flash`。24 kHz 的指令驱动生成与编辑。参考音频可选；语音需要 `stage_params.auk_engine.gen_seconds`。会下载独立的 Qwen2.5-Omni-3B 编码器。串行、非流式引擎 |

## 启动服务器

模型默认值、各阶段 `process` 覆盖以及同 GPU 显存要求，请参见
[TTS 进程拓扑](tts_process_topology.md)。

下面的参考音频示例会从 Hugging Face 获取音频剪辑，因此这些命令包含了
Hugging Face 主机及其当前的下载重定向主机。当你的请求只使用文本、已上传音色、
本地/文件参考或 data URL 时，请省略这些标志。

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --port 8000
```

批量语音请求默认最多接受 32 个条目。使用
`--tts-batch-max-items` 可更改服务器端的请求信封上限：

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --tts-batch-max-items 32 \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --port 8000
```

对于 Voxtral：

```bash
sgl-omni serve \
  --model-path mistralai/Voxtral-4B-TTS-2603 \
  --config examples/configs/voxtral_tts.yaml \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --port 8000
```

对于 Qwen3-TTS Base：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --port 8000
```

对于 Qwen3-TTS CustomVoice：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --port 8000
```

对于 0.6B CustomVoice，请使用 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` 并搭配 `examples/configs/qwen3_tts_0_6b_customvoice.yaml`。

对于 Qwen3-TTS VoiceDesign：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --config examples/configs/qwen3_tts_1_7b_voicedesign.yaml \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --port 8000
```

对于 MOSS-TTS：

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config examples/configs/moss_tts.yaml \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --port 8000
```

对于 dots.tts MeanFlow：

```bash
sgl-omni serve \
  --model-path dots-studio/dots.tts-mf \
  --config examples/configs/dots_tts.yaml \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --allowed-media-domain us.aws.cdn.hf.co \
  --port 8000
```

对于 dots.tts SOAR：

```bash
sgl-omni serve \
  --model-path dots-studio/dots.tts-soar \
  --config examples/configs/dots_tts_soar.yaml \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --allowed-media-domain us.aws.cdn.hf.co \
  --port 8000
```

`dots.tts-base` 使用相同的配置；传入 `--model-path dots-studio/dots.tts-base` 即可。

SOAR 和 base 是 flow-matching checkpoint，因此它们以 `max_running_requests=1`
运行带无分类器引导（classifier-free guidance）的单请求求解器。目前连续批处理
（continuous batching）仅适用于 MeanFlow。`rednote-hilab/dots.tts-*` 是旧的组织名，
会重定向到 `dots-studio/dots.tts-*`；两者都可作为 `--model-path` 使用。

对于 Ming-Omni-TTS：

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-omni-tts-16.8B-A3B \
  --config examples/configs/ming_omni_tts.yaml \
  --port 8000
```

对于 Fun-CosyVoice3：

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --allowed-media-domain huggingface.co \
  --allowed-media-domain cas-bridge.xethub.hf.co \
  --allowed-media-domain us.aws.cdn.hf.co \
  --port 8000
```

## 使用 Curl

不带任何参考音频、直接从文本生成语音。这适用于
Qwen3-TTS CustomVoice、Voxtral 和 S2-Pro，不适用于 Qwen3-TTS Base。

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "fishaudio/s2-pro",
      "voice": "default",
      "input": "Hello, how are you?"
    }' \
    --output output.wav
```

Qwen3-TTS Base 需要参考音频：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output.wav
```

Qwen3-TTS CustomVoice 使用内置说话人，无需参考音频：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello from Qwen CustomVoice.",
    "voice": "Ryan",
    "language": "English",
    "instructions": "Speak clearly and calmly."
  }' \
  --output custom-voice.wav
```

省略克隆相关字段（`ref_audio`、`ref_text`、`references` 和 `x_vector_only_mode`），并省略 `task_type` 或将其设为 `CustomVoice`。对于 0.6B，请省略 `instructions`：出于兼容性它仍被接受，但不支持可靠的指令控制。关于说话人发现、流式以及 Eric/Dylan 的语言行为，参见 [CustomVoice checkpoint](customvoice-checkpoints)。

Qwen3-TTS VoiceDesign 使用文本加语音指令：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
      "voice": "default",
      "input": "Hello, how are you?",
      "task_type": "VoiceDesign",
      "instructions": "A warm, natural young adult voice."
    }' \
    --output output.wav
```

dots.tts 接受相同的参考字段。MeanFlow checkpoint 针对四个流步
（flow steps）进行了调优：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-studio/dots.tts-mf",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "stage_params": {"latent_engine": {"num_steps": 4}}
  }' \
  --output dots-output.wav
```

要想让 Fish Speech S2-Pro 的结果听起来自然，请使用带参考音频剪辑的语音克隆。

### Fish Speech 语音克隆

下面的示例使用来自 [`seed-tts-eval-mini`](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini) 的一段样例音频剪辑。`references` 字段接受 `audio_path`（本地路径、file URL、data URL 或 HTTP URL）和 `text`（该音频的转录文本）。

1. 非流式请求

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

2. 流式

启用流式以实时接收原始 PCM 音频分块。HTTP 流式需要同时设置
`"stream": true` 和 `"response_format": "pcm"`：

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

流式返回 16 位单声道 PCM 字节（`audio/pcm`），采样率元数据位于响应头中。
它不包含带内 JSON 事件、最终用量信息或终止哨兵（sentinel）。当客户端未设置
`initial_codec_chunk_frames` 时，模型会选择一个能保证连续性的首个声码器
（vocoder）分块。显式设置该字段可覆盖这一默认行为，或将其设为 `0` 以从头
使用模型的稳定分块大小。Ming-Omni-TTS 是唯一拒绝该字段的模型：其初始与
稳定节奏是 audio_decode 阶段的 `factory` 设置，因此设置该字段的请求会失败。

### 批量语音

当一个请求需要合成多条独立的语音时，请使用 `/v1/audio/speech/batch`。
批量默认值会与每个条目合并。条目字段会覆盖默认值，并且每个条目都会走
正常的 `/v1/audio/speech` 路径。

```bash
curl -X POST http://localhost:8000/v1/audio/speech/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "response_format": "wav",
    "items": [
      {"input": "First sentence."},
      {"input": "Second sentence.", "speed": 1.1},
      {
        "input": "Use a reference clip for this item.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his."
      }
    ]
  }'
```

响应会保持条目顺序。成功的条目包含 base64 编码的音频字节和所选的媒体类型。
失败的条目在条目级别包含 OpenAI 风格的错误对象。无效的批量信封（例如条目
过多）会导致整个 HTTP 请求失败。

### WebSocket 语音流式

在持久 websocket 上进行有状态的文本输入时，请使用 `/v1/audio/speech/stream`。
第一条消息必须是 `session.config`，然后发送 `input.text` 消息。发送
`input.commit` 可在保持 WebSocket 打开的同时刷新当前文本段，或使用 `input.done`
结束会话。服务器会以 `session.configured` 确认初始配置。

`stream_audio` 默认为 `false`。在默认设置下，每个完成的文本段会在
`audio.start` 和 `audio.done` 之间返回一个二进制音频帧。当
`stream_audio=true` 时，`response_format` 必须为 `pcm`，服务器会在
`audio.start` 和 `audio.done` 之间发送增量的二进制 PCM 帧。

```python
import asyncio
import json

import websockets


async def main():
    async with websockets.connect(
        "ws://localhost:8000/v1/audio/speech/stream"
    ) as ws:
        await ws.send(json.dumps({
            "type": "session.config",
            "session": {
                "model": "fishaudio/s2-pro",
                "voice": "default",
                "response_format": "pcm",
                "stream_audio": True,
                "split_granularity": "sentence",
            },
        }))
        print(await ws.recv())

        pcm_chunks = []
        await ws.send(json.dumps({
            "type": "input.text",
            "text": "Hello from the speech WebSocket. This is the second sentence.",
        }))
        await ws.send(json.dumps({"type": "input.commit"}))

        # input.committed is emitted after all audio for the segment. More
        # input.text/input.commit pairs can follow on the same WebSocket.
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                pcm_chunks.append(message)
                continue
            event = json.loads(message)
            print(event)
            if event["type"] == "input.committed":
                break

        await ws.send(json.dumps({"type": "input.done"}))

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                pcm_chunks.append(message)
                continue
            event = json.loads(message)
            print(event)
            if event["type"] == "session.done":
                break

        with open("websocket_output.pcm", "wb") as f:
            f.write(b"".join(pcm_chunks))


asyncio.run(main())
```

每个 `input.commit` 都会强制刷新所有剩余的缓冲文本（包括未在配置的句子
或子句边界处结束的文本）。该段的所有音频完成后，服务器会发出
`input.committed`，携带 `segment_index`、`segment_sentences` 以及累计的
`total_sentences`，然后继续在同一连接上接受更多输入。空提交是合法的：
它不刷新任何内容并报告 `segment_sentences: 0`。
`input.done` 会执行同样的最终缓冲区刷新，发出 `session.done`，然后关闭
WebSocket。

`split_granularity` 可以是 `sentence` 或 `clause`。未知消息类型和格式错误的
JSON 会返回 WebSocket `error` 事件。初始配置缺失或无效时会返回错误并关闭
会话。

### 上传音色

使用 `/v1/audio/voices` 一次性注册参考音频剪辑，之后在 `/v1/audio/speech`
请求中按名称复用。上传的样本以 `.safetensors` 文件形式存储在
`SPEAKER_SAMPLES_DIR` 下，并在服务器重启时恢复。如果未设置
`SPEAKER_SAMPLES_DIR`，服务器会使用 `~/.cache/sglang-omni/speakers`。
`SPEAKER_MAX_UPLOADED` 限制可存储的音色数量，默认为 `1000`。

上传一个音色样本：

```bash
curl -X POST http://localhost:8000/v1/audio/voices \
  -F "name=narrator" \
  -F "consent=consent-recording-id" \
  -F "ref_text=Transcript of the uploaded reference clip." \
  -F "speaker_description=Clear narration voice" \
  -F "audio_sample=@reference.wav;type=audio/wav"
```

列出预置和已上传的音色：

```bash
curl http://localhost:8000/v1/audio/voices
```

对于 CustomVoice，响应中的 `voices` 列表包含 `default` 以及所部署 checkpoint 的内置说话人。上传的参考音色不会用于 CustomVoice 合成。

按名称使用已上传的音色：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "input": "The uploaded voice can now be reused without resending audio.",
    "voice": "narrator",
    "response_format": "wav"
  }' \
  --output narrator.wav
```

删除已上传的音色：

```bash
curl -X DELETE http://localhost:8000/v1/audio/voices/narrator
```

可接受的上传格式包括 WAV、MP3、FLAC、OGG、AAC、WebM 和 MP4。每个文件
最大不得超过 10 MiB，并包含 1-30 秒的非静音参考音频。以相同 `name` 上传
会覆盖之前的样本。删除音色会移除持久化的样本。列表响应中包含 API 进程的
`cache_stats`，用于已上传音色引用查询的可观测性。

## 使用 Python

### 基础 TTS

这个无参考请求适用于 Fish Speech S2-Pro 和 Voxtral TTS。

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "fishaudio/s2-pro",
        "voice": "default",
        "input": "Hello, how are you?",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### OpenAI Python SDK

当客户端指向 SGLang-Omni 服务器时，该端点与 OpenAI Python SDK 兼容：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.audio.speech.create(
    model="fishaudio/s2-pro",
    voice="default",
    input="Hello, how are you?",
    response_format="wav",
)
response.stream_to_file("output.wav")
```

### 语音克隆

```python
REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
SPEECH_INPUT = "Get the trust fund to the bank early."
```

1. 非流式请求

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "fishaudio/s2-pro",
        "voice": "default",
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

2. 流式请求

```python
import wave

import requests

payload = {
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": SPEECH_INPUT,
    "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    "stream": True,
    "response_format": "pcm",
}

chunks = []
with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json=payload,
    stream=True,
    timeout=600,
) as stream:
    stream.raise_for_status()
    sample_rate = int(stream.headers.get("x-sample-rate", 24000))
    for chunk in stream.iter_content(chunk_size=None):
        if chunk:
            chunks.append(chunk)

with wave.open("output_stream.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate or 24000)
    w.writeframes(b"".join(chunks))
```

## 请求参数

下表列出了 `/v1/audio/speech` 端点接受的所有参数。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 所部署的模型 | 所部署模型的标识符 |
| `input` | string | （必填） | 要合成的文本 |
| `voice` | string | `"default"` | 预置或已上传音色的标识符 |
| `response_format` | string | `"wav"` | 输出音频格式：`wav`、`mp3`、`flac`、`pcm`、`aac` 或 `opus` |
| `speed` | float | `1.0` | 播放速度倍率，范围为 `0.25` 到 `4.0` |
| `stream` | bool | `false` | 启用原始 PCM 流式。为 true 时，`response_format` 必须为 `pcm` |
| `initial_codec_chunk_frames` | int | `null` | 可选的首个编解码器（codec）分块大小，用于流式 TTFA / 播放连续性调优。省略时，各模型采用自己的默认值：Qwen3-TTS 以 `1 -> 2 -> 4` 逐步过渡到稳定步长，Higgs TTS 使用 `20`，MOSS-TTS Local 使用 `5`，ZONOS2 使用 `40`。显式设为 `0` 则从头使用模型的稳定分块大小。Ming-Omni-TTS 完全拒绝该字段 |
| `stream_codec_output` | bool | `true` | 仅适用于 Qwen3-TTS。将 codec 帧在生成的同时转发给声码器（vocoder）。设为 `false` 可为 CustomVoice / VoiceDesign 恢复整句解码 |
| `suppress_bootstrap_silence` | bool | `true` | 仅适用于 Qwen3-TTS。在已验证的音色/语言组合上，从流式 CustomVoice 输出中抑制静音的引导（bootstrap）codec 帧音频；可听见的首帧始终原样发出。设为 `false` 可保留开头的静音 |
| `references` | list | `null` | 用于语音克隆的参考音频。每个条目包含 `audio_path`（本地路径 / file URL / data URL / 远程 URL）和 `text` |
| `ref_audio` | string | `null` | 参考音频路径 / URL / base64 字符串。等价于 `references[0].audio_path` |
| `ref_text` | string | `null` | `ref_audio` 的转录文本。等价于 `references[0].text` |
| `language` | string | `null` | 语言提示：`Auto`、`Chinese`、`English`、`Japanese`、`Korean`、`German`、`French`、`Russian`、`Portuguese`、`Spanish` 或 `Italian` |
| `task_type` | string | `null` | Qwen3-TTS 任务类型：`Base`、`CustomVoice` 或 `VoiceDesign`。存在参考音频/文本时推断为 `Base`，否则为 `CustomVoice` |
| `instructions` | string | `null` | Qwen3-TTS 风格指令或 VoiceDesign 指令 |
| `max_new_tokens` | int | `null` | 生成 token 的最大数量 |
| `token_count` | int | `null` | 模型特定的时长 token 目标值 |
| `duration_tokens` | int | `null` | 面向提供时长控制的模型的别名式时长 token 目标值 |
| `x_vector_only_mode` | bool | `null` | Qwen3-TTS Base 说话人嵌入模式 |
| `temperature` | float | `null` | 采样温度 |
| `top_p` | float | `null` | Top-p 采样 |
| `top_k` | int | `null` | Top-k 采样 |
| `repetition_penalty` | float | `null` | 重复惩罚 |
| `seed` | int | `null` | 模型特定。Qwen3-TTS Base 接受请求级 seed，Voxtral TTS 目前拒绝 seed |

无效的语音请求会返回 OpenAI 风格的错误信封：

```json
{
  "error": {
    "message": "stream=true requires response_format='pcm'",
    "type": "BadRequestError",
    "param": "response_format",
    "code": 400
  }
}
```

## H200 SeedTTS 基准测试命令

先下载完整的 SeedTTS 数据集：

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
```

在 8000 端口启动目标服务器后，运行 EN 和 ZH。在完整的 H200 运行完成之前，
不要将基准测试结果添加到文档中。

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --port 8000 \
  --output-dir results/qwen3_tts_0_6b_en \
  --lang en \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --port 8000 \
  --output-dir results/qwen3_tts_0_6b_zh \
  --lang zh \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --port 8000 \
  --output-dir results/qwen3_tts_1_7b_en \
  --lang en \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --port 8000 \
  --output-dir results/qwen3_tts_1_7b_zh \
  --lang zh \
  --max-concurrency 16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model mistralai/Voxtral-4B-TTS-2603 \
  --port 8000 \
  --output-dir results/voxtral_en \
  --lang en \
  --max-new-tokens 4096 \
  --max-concurrency 16 \
  --no-ref-audio \
  --voice cheerful_female

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model mistralai/Voxtral-4B-TTS-2603 \
  --port 8000 \
  --output-dir results/voxtral_zh \
  --lang zh \
  --max-new-tokens 4096 \
  --max-concurrency 16 \
  --no-ref-audio \
  --voice cheerful_female
```

## 交互式 Playground

SGLang-Omni 自带一个基于 Gradio 的 playground，用于交互式 TTS 实验：

```bash
./playground/s2pro/start.sh
```

该 playground 现在针对同一个 S2 Pro 后端提供两种演示模式：

- `Non-Streaming` 启动一个标准请求，并在生成结束后展示最终的 WAV。
- `Streaming` 消费 `/v1/audio/speech` 的原始 PCM 流，转换增量分块用于播放，同时还会写出一个最终合并的 WAV 产物以供检查。

启动脚本会先启动后端，等待 `/health`，然后使用以下命令启动 Gradio UI：

```bash
python -m playground.s2pro.app --api-base http://localhost:8000
```

演示操作视频见[这里](https://x.com/lmsysorg/status/2031412267213008984/video/1)。我们强烈推荐使用 playground，因为音频数据很难通过 CLI 交互。
