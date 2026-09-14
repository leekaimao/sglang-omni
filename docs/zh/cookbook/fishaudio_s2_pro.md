# Fish Audio S2-Pro

[Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) 是通过 `/v1/audio/speech` 提供服务的文本转语音模型。它支持普通 TTS、基于参考音频的语音克隆以及流式音频块输出。

(prerequisites)=
## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`。

Fish Audio 使用 Descript DAC 编解码器，它并不包含在基础的
`sglang-omni` 包中。请在 SGLang-Omni 仓库根目录下安装其模型专属依赖：

```bash
uv pip install \
  "descript-audiotools==0.7.2" \
  "descript-audio-codec==1.0.0"
```

然后下载模型：

```bash
hf download fishaudio/s2-pro
```

## 服务器配置

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

## 语音合成

普通 TTS：

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

语音克隆：

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

流式输出：

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

## 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 已服务的模型 | 已服务的模型标识符 |
| `input` | string | 必填 | 要合成的文本 |
| `voice` | string | `default` | 非参考请求使用的音色标识符 |
| `response_format` | string | `wav` | 输出音频格式 |
| `speed` | float | `1.0` | 播放速度倍率 |
| `stream` | bool | `false` | 流式返回原始 PCM 音频块 |
| `references` | list | `null` | 用于语音克隆的参考音频，每项包含 `audio_path` 与 `text` |
| `ref_audio` / `ref_text` | string | `null` | `references[0].audio_path` 与 `references[0].text` 的简写 |
| `max_new_tokens` | int | `2048` | 生成的语义 token 最大数量 |
| `temperature` | float | `0.8` | 采样温度 |
| `top_p` | float | `0.8` | Top-p 采样 |
| `top_k` | int | `30` | Top-k 采样。必须为 `-1` 或介于 `1` 与 `30` 之间 |
| `repetition_penalty` | float | `1.1` | 重复惩罚 |

## 已知限制

- `top_k` 被限制为 `-1` 或 `1..30`；请将请求保持在该范围内，因为非法取值目前会使 S2-Pro 流水线直接失败，而不是返回干净的参数错误。
- 参考音频的质量会显著影响克隆音色的质量。
- 交互式播放请使用流式输出；在命令行中检查原始音频响应并不方便。
