# Voxtral TTS

[Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) 是 Mistral AI 基于 Ministral-3B 骨干网络构建的开源权重文本转语音模型。它可以生成自然的 24 kHz 语音，韵律自然，支持 9 种语言，并附带一组预置的具名音色。在 SGLang-Omni 中，Voxtral 以 `preprocessing → tts_generation → vocoder` 流水线运行，并通过 OpenAI 兼容的 `/v1/audio/speech` 端点提供服务。


## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`，然后安装 Voxtral 专用的 tokenizer 并下载模型：

```bash
# Voxtral 预处理使用 mistral-common 提供的 Mistral Tekken tokenizer。
uv pip install 'mistral_common[audio]>=1.11.0'

hf download mistralai/Voxtral-4B-TTS-2603
```

模型仓库是公开的，因此不需要 Hugging Face token。

## 服务器配置

流水线为 `preprocessing → tts_generation → vocoder`。
首次启动可能需要几分钟，因为 `tts_generation` 阶段需要捕获 CUDA Graph。

```bash
sgl-omni serve \
  --model-path mistralai/Voxtral-4B-TTS-2603 \
  --config examples/configs/voxtral_tts.yaml \
  --port 8000
```

## 语音合成

### 预置音色

Voxtral 在 checkpoint 中附带预置音色。使用 `cheerful_female` 作为默认预置音色。

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "voice": "cheerful_female",
    "input": "SGLang-Omni is a great project!"
  }' \
  --output output.wav
```

### 具名音色

Voxtral 使用**预置的具名音色**发声（不支持从参考音频片段克隆）。通过 `voice` 字段选择：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "voice": "casual_male",
    "input": "Get the trust fund to the bank early.",
    "max_new_tokens": 4096
  }' \
  --output output.wav
```

可用音色以 `voice_embedding/*.pt` 文件的形式内置于 checkpoint 中。可以从下载好的快照中列出它们：

```bash
MODEL_PATH="$(hf download mistralai/Voxtral-4B-TTS-2603 --quiet)"
ls "${MODEL_PATH}/voice_embedding"
```

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "voice": "casual_male",
        "input": "Get the trust fund to the bank early.",
        "max_new_tokens": 4096,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### 流式输出

设置 `"stream": true` 与 `"response_format": "pcm"` 即可实时接收原始 PCM 音频块：

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "voice": "casual_male",
    "input": "Get the trust fund to the bank early.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

流式返回 `audio/pcm` 的 16 位单声道 PCM 字节，采样率元数据放在响应头中。完整的 Python 原始 PCM 消费端示例参见 [Higgs TTS cookbook](streaming)。

## 请求参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | 已服务的模型 | 已服务的模型标识符 |
| `input` | （必填） | 要合成的文本 |
| `voice` | `default` | checkpoint `voice_embedding/` 目录中的预置音色名 |
| `max_new_tokens` | `4096` | 生成的声学 token 最大数量 |
| `response_format` | `wav` | 输出容器格式（`wav`、`mp3`、`flac`、`opus`、`aac`、`pcm`） |
| `stream` | `false` | 流式返回原始 PCM 音频块 |

> Voxtral 的生成是**确定性的**：引擎将 `temperature` 固定为 `0.0`，因此 `top_p`、`top_k`、`temperature` 等采样参数不会被使用。Voxtral **不**支持基于参考音频片段的语音克隆（`references`）——请改用预置 `voice`。

## 基准测试结果

Seed-TTS EN（完整集，1088 条语句）、bf16、`max_new_tokens=4096`、
`--no-ref-audio --voice cheerful_female`、并发 16，WER 使用 HF
Whisper-large-v3 评分。硬件：1× H200 SXM。

| 指标 | 数值 |
|---|---|
| WER（语料级 micro 平均） | 1.20% |
| WER（样本均值 / 中位数） | 1.22% / 0.00% |
| WER（样本 p95 / 最大值） | 9.09% / 42.86% |
| WER > 50% 的样本数 | 0 / 1088 |
| 延迟均值 / 中位数（秒） | 2.94 / 2.86 |
| 延迟 p95 / p99（秒） | 4.56 / 5.37 |
| RTF 均值 / 中位数 | 0.519 / 0.541 |
| 输出吞吐（tok/s） | 383.7 |
| 吞吐（req/s） | 5.40 |
| 完成 / 失败请求数 | 1088 / 0 |

可使用 `benchmarks/README.md` 中记录的 SeedTTS 命令复现。Voxtral 模型卡还给出了并发 1 下约 70 ms 的首音频延迟；上表是并发 16 的吞吐导向运行，因此其 RTF 反映的是批量负载，而非延迟优化的单流数据。输出为 24 kHz。

## 已知限制

- **仅支持预置音色。** Voxtral 从 checkpoint 内置的具名音色中选择；在该引擎中不支持从参考音频片段克隆任意说话人。
- **确定性解码。** `temperature` 固定为 `0.0`；无法通过采样参数以确定性换取多样性。
- **语言覆盖。** 质量针对 9 种支持语言（英语、法语、西班牙语、德语、意大利语、葡萄牙语、荷兰语、阿拉伯语、印地语）调优。
- **非商业许可。** 权重采用 CC BY-NC 4.0 许可；不允许商业使用。
