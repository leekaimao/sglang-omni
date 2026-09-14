# Higgs TTS

[Higgs Audio v3 TTS](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)
是 Boson AI 推出的文本转语音模型。它生成 **24 kHz 语音**，支持 [**100+ 种语言**](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b#supported-languages)、从参考片段克隆音色，以及对情绪、风格、音效与韵律的细粒度**内联控制（inline control）**。

![Higgs Audio v3 Generation Architecture](../_static/image/higgs-architecture.png)

Higgs 的自回归解码器消费交错排列的文本与音频 token。音频由 **Higgs Tokenizer** 编码为 25 fps 的 8 个 codebook，通过 **delay pattern** 交错排布，再经**多 codebook 融合 embedding** 映射到骨干网络的隐藏状态。输出的 code 通过**多 codebook 融合头**，去延迟（de-delay）后解码回波形。多轮生成交错 `<|text|>…<|audio|>…` 块，使每个新块都以参考音频 + 先前块为依据。

| 组件 | 规格 |
|---|---|
| 骨干网络 | ~4B 自回归解码器（36 层，hidden=2560，GQA 32/8） |
| 多 codebook embedding / 头 | 融合为单张量，与文本 embedding 绑定 |
| 上下文长度 | 8,192 token（训练序列长度） |
| 音频 token | 8 codebook × 1026 词表，delay pattern |
| 采样率 | 24 kHz |
| 帧率 | 25 fps（每帧 40 ms） |

## 评测基准

### 多语言语音克隆

我们在公开的多语言 TTS 套件以及内部的 111 语言 Higgs-Multilingual 集上评测 Higgs Audio v3 TTS，覆盖常见语言与低资源语言。

WER / CER（↓，%），在各个基准的语言集上做宏平均。Higgs Audio v3 TTS 的结果可使用原始指标与归一化方式复现：

| 基准 | 语言数 | WER/CER ↓ |
|---|---:|---:|
| Seed-TTS | 2 | 1.11 |
| CV3 | 9 | 4.41 |
| MiniMax-Multilingual | 23 | 2.74 |
| Higgs-Multilingual | 111 | 3.61 |

### Emergent TTS

Emergent TTS 基准上各类别的胜率（↑）——评审相对于固定基线的偏好。基准文本按原样运行（不使用内联控制标签）。

| 类别 | 胜率 ↑ |
|---|---:|
| 总体 | 53.65% |
| 情绪 | 53.75% |
| 外来词 | 48.75% |
| 副语言 | 68.57% |
| 复杂发音 | 25.10% |
| 疑问句 | 61.43% |
| 句法复杂度 | 60.71% |

## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`，然后下载并启动模型服务：

```bash
hf download bosonai/higgs-audio-v3-tts-4b

sgl-omni serve \
  --model-path bosonai/higgs-audio-v3-tts-4b \
  --allowed-local-media-path docs/_static/audio \
  --port 8000
```

下面的语音克隆示例使用来自
`docs/_static/audio` 的本地参考片段。

## 语音合成

### 零样本（Zero-shot）

1. 使用 curl

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "Hello, how are you?"
  }' \
  --output output.wav
```

2. 使用 Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "bosonai/higgs-audio-v3-tts-4b",
        "voice": "default",
        "input": "Hello, how are you?",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

参考输出：

<audio controls>
  <source src="../_static/audio/higgs-1.wav" type="audio/wav">
</audio>

### 语音克隆

提供参考片段的转写文本（`text`）能显著提升克隆质量。

1. 使用 curl

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "Have a nice day and enjoy south california sunshine.",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

2. 使用 Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "bosonai/higgs-audio-v3-tts-4b",
        "voice": "default",
        "input": "Have a nice day and enjoy south california sunshine.",
        "references": [{
            "audio_path": "docs/_static/audio/male-voice.wav",
            "text": "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time.",
        }],
        "temperature": 0.8,
        "top_k": 50,
        "max_new_tokens": 1024,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

参考输入：

<audio controls>
  <source src="../_static/audio/male-voice.wav" type="audio/wav">
</audio>

参考输出：

<audio controls>
  <source src="../_static/audio/higgs-2.wav" type="audio/wav">
</audio>

(streaming)=
### 流式输出

标准请求需要等完整音频生成完毕才能收到任何内容，而流式（streaming）让你在**生成仍在进行时**就开始接收并播放音频。这能显著降低首音频延迟，对实时或交互式场景至关重要。

Higgs TTS 以原始 PCM 字节实现流式。你的客户端可以边到达边播放或缓冲每个
chunk，而不必等待完整响应。

在请求体中设置 `"stream": true` 与 `"response_format": "pcm"`
即可启用流式。生成期间，声码器（vocoder）会输出增量的音频 chunk。
HTTP 响应返回 `audio/pcm` 字节，并在响应头中提供采样率元数据。

1. 使用 curl

在请求体中设置 `"stream": true` 与 `"response_format": "pcm"`：

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

`-N` 标志禁用 curl 的输出缓冲，使每个 chunk 一到达就被写入。

流式返回 `audio/pcm` 的 16 位单声道 PCM 字节。它没有带内的 JSON
事件、最终的 usage 事件或终止哨兵。响应头会报告实际的流采样率、声道数与位深。Higgs 默认把
`initial_codec_chunk_frames` 设为 `20`；在 8 个 codebook 下，AR 生产者
在第 27 行冲刷对应的延迟 code 行。客户端仍可设置其他值，包括 `0`（从一开始就使用稳定的
chunk 大小）。生产者与声码器共享该生效值，且它只控制第一个
chunk。后续 chunk 回到 Higgs 正常的流式窗口。

2. 使用 Python

本例把流式 PCM 字节写入一个 WAV 文件。在真实应用中，你会把
chunk 直接送入音频播放器（例如通过 `pyaudio` 或
`sounddevice`）。

```python
import wave

import requests

REFERENCE_AUDIO = "docs/_static/audio/male-voice.wav"
REFERENCE_TEXT = "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time."
SPEECH_INPUT = "Get the trust fund to the bank early."

with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "bosonai/higgs-audio-v3-tts-4b",
        "voice": "default",
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
        "stream": True,
        "response_format": "pcm",
    },
    stream=True,
) as resp:
    resp.raise_for_status()
    chunks = []
    sample_rate = int(resp.headers.get("x-sample-rate", 24000))
    for chunk in resp.iter_content(chunk_size=None):
        if chunk:
            chunks.append(chunk)
            # In a real app: feed `chunk` to your audio player here

with wave.open("output_streaming.wav", "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    f.writeframes(b"".join(chunks))
```

参考输出：

<audio controls>
  <source src="../_static/audio/higgs-4.wav" type="audio/wav">
</audio>

### 内联控制 token

所有标签都遵循 `<|category:value|>` 语法，可以在话语中间插入。

- **情绪（Emotion）** — `elation`、`amusement`、`enthusiasm`、`determination`、`pride`、`contentment`、`affection`、`relief`、`contemplation`、`confusion`、`surprise`、`awe`、`longing`、`arousal`、`anger`、`fear`、`disgust`、`bitterness`、`sadness`、`shame`、`helplessness`
- **风格（Style）** — `singing`、`shouting`、`whispering`
- **音效（Sound effects）** — `cough`、`laughter`、`crying`、`screaming`、`burping`、`humming`、`sigh`、`sniff`、`sneeze`
- **韵律（Prosody）**
  - 语速 — `speed_very_slow`（约 0.65×）、`speed_slow`（约 0.85×）、`speed_fast`（约 1.2×）、`speed_very_fast`（约 1.4×）
  - 停顿 — `pause`（约 400–700 ms）、`long_pause`（约 700–1500 ms）
  - 音高 — `pitch_low`（约 −3 半音）、`pitch_high`（约 +2.5 半音）
  - 表现力 — `expressive_high`、`expressive_low`

把控制 token 直接嵌入 `input` 字段。不同类别的
token 可以组合使用。每个请求是一个单独的**回合（turn）**，两条规则可以让控制 token 稳定生效：

1. **把 delivery 类 token 放在回合开头。** 情绪（`<|emotion:…|>`）、风格
   （`<|style:…|>`）以及韵律中的*语速*（`<|prosody:speed_…|>`）、*音高*
   （`<|prosody:pitch_…|>`）与*表现力*（`<|prosody:expressive_…|>`）token
   决定整个回合的演绎方式，因此要把它们放在 `input` 的最开头、任何文本之前。位置类 token 是例外：
   `<|prosody:pause|>` / `<|prosody:long_pause|>` 要内联放在停顿应当出现的位置，而每个
   `<|sfx:…|>` 要紧跟在它触发的声音之前。

2. **每个音效都配上对应的拟声词。** `<|sfx:…|>` token 只有在紧随其后的文本写出匹配的声音时效果最好
   （例如 `<|sfx:laughter|>Haha`、`<|sfx:sigh|>Uh`、`<|sfx:sneeze|>Achoo`）——
   拟声词为模型提供了实现该音效的声学线索。

**演示**

1. 情绪：amusement + laughter

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "<|emotion:amusement|><|prosody:expressive_high|>Wait, wait, that was kind of hilarious. <|sfx:laughter|>Hehe, no, seriously, I was not ready for that.",
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

参考输出：

<audio controls>
  <source src="../_static/audio/control-tokens-test1.wav" type="audio/wav">
</audio>

2. 情绪：anger + shouting

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "<|emotion:anger|><|style:shouting|>No, that is not okay! We cannot ship something that sounds broken, delayed, and unnatural.",
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

参考输出：

<audio controls>
  <source src="../_static/audio/control-tokens-test2.wav" type="audio/wav">
</audio>

3. 情绪：sadness + sniff

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "<|emotion:sadness|><|sfx:crying|>I... I’m sorry. <|sfx:sniff|>Sff, We really tried. after all those late nights, I thought the whole thing had failed.",
    "references": [{
      "audio_path": "docs/_static/audio/ref_voice.wav",
      "text": "It was the night before my birthday. Hooray! It’s almost here! It may not be a holiday, but it’s the best day of the year."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

参考输出：

<audio controls>
  <source src="../_static/audio/control-tokens-test3.wav" type="audio/wav">
</audio>

4. 情绪：confusion + humming + sigh

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "<|emotion:confusion|><|sfx:humming|>Hmm... wait. <|sfx:sigh|>Uh, I’m not sure I understand. Do you mean the voice should speak faster, or the system should respond earlier?",
    "references": [{
      "audio_path": "docs/_static/audio/ref_voice.wav",
      "text": "It was the night before my birthday. Hooray! It’s almost here! It may not be a holiday, but it’s the best day of the year."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

参考输出：

<audio controls>
  <source src="../_static/audio/control-tokens-test4.wav" type="audio/wav">
</audio>

5. 情绪：surprise + screaming

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "<|emotion:surprise|><|prosody:pitch_high|><|sfx:screaming|>Ah! Wait, I almost forgot! Higgs Audio v3 also supports over one hundred languages.",
    "references": [{
      "audio_path": "docs/_static/audio/ref_voice.wav",
      "text": "It was the night before my birthday. Hooray! It’s almost here! It may not be a holiday, but it’s the best day of the year."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

参考输出：

<audio controls>
  <source src="../_static/audio/control-tokens-test5.wav" type="audio/wav">
</audio>

6. 组合使用：

下面是把情绪、音效与韵律 token 组合起来的示例——两位说话人之间一段高考风格的英语听力对话：

<details>
<summary>命令</summary>

第 1 部分——她询问缺的课：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "<|emotion:contemplation|>Hi David, I missed the biology class today because I caught a cold. <|sfx:cough|>Ahem! Sorry, Could you tell me what the teacher covered?",
    "references": [{
      "audio_path": "docs/_static/audio/female-voice.wav",
      "text": "By repeating what students say, teachers can demonstrate that they are listening. By extending what students say."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output part1.wav
```

第 2 部分——他讲解课程内容：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "<|emotion:enthusiasm|>Sure, no problem! We learned how plants make food through photosynthesis, and <|prosody:long_pause|> there will be a quiz this Friday.",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output part2.wav
```

第 3 部分——她说出结果：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "voice": "default",
    "input": "<|emotion:relief|>Oh, that is really helpful. Thank you!",
    "references": [{
      "audio_path": "docs/_static/audio/female-voice.wav",
      "text": "By repeating what students say, teachers can demonstrate that they are listening. By extending what students say."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output part3.wav
```

拼接（各句之间约 0.6 秒间隔）：

```bash
ffmpeg -y \
  -i part1.wav -f lavfi -t 0.6 -i anullsrc=r=24000:cl=mono \
  -i part2.wav -f lavfi -t 0.6 -i anullsrc=r=24000:cl=mono \
  -i part3.wav \
  -filter_complex "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1" \
  gaokao_listening.wav
```

</details>

参考输出：

<audio controls>
  <source src="../_static/audio/gaokao-listening.wav" type="audio/wav">
</audio>

#### 情绪

| Token | 说明 |
|---|---|
| `<\|emotion:elation\|>` | 兴奋 / 喜悦 |
| `<\|emotion:amusement\|>` | 愉悦 / 顽皮的笑 |
| `<\|emotion:enthusiasm\|>` | 热情 / 激动 |
| `<\|emotion:determination\|>` | 坚定 / 果断 |
| `<\|emotion:pride\|>` | 自豪 / 自信 |
| `<\|emotion:contentment\|>` | 平静的满足 |
| `<\|emotion:affection\|>` | 温暖 / 喜爱 |
| `<\|emotion:relief\|>` | 如释重负 |
| `<\|emotion:contemplation\|>` | 沉思 / 内省 |
| `<\|emotion:confusion\|>` | 困惑 |
| `<\|emotion:surprise\|>` | 惊讶 |
| `<\|emotion:awe\|>` | 敬畏 / 惊奇 |
| `<\|emotion:longing\|>` | 渴望 / 向往 |
| `<\|emotion:arousal\|>` | 高涨的欲望 |
| `<\|emotion:anger\|>` | 愤怒 |
| `<\|emotion:fear\|>` | 恐惧 |
| `<\|emotion:disgust\|>` | 厌恶 |
| `<\|emotion:bitterness\|>` | 苦涩 |
| `<\|emotion:sadness\|>` | 悲伤 |
| `<\|emotion:shame\|>` | 羞愧 |
| `<\|emotion:helplessness\|>` | 无助 |

#### 风格

| Token | 说明 |
|---|---|
| `<\|style:singing\|>` | 唱歌 |
| `<\|style:shouting\|>` | 喊叫 / 投射的声音 |
| `<\|style:whispering\|>` | 耳语 |

#### 音效

每个 token 之后要紧跟匹配的拟声词。

| Token | 说明 | 建议拟声词 |
|---|---|---|
| `<\|sfx:cough\|>` | 咳嗽 | Ahem |
| `<\|sfx:laughter\|>` | 笑声 | Haha / Hehe |
| `<\|sfx:crying\|>` | 哭泣 | Boohoo / Sob |
| `<\|sfx:screaming\|>` | 尖叫 | Ahh / Aaah |
| `<\|sfx:burping\|>` | 打嗝 | Burp |
| `<\|sfx:humming\|>` | 哼唱 | Hmm / Mmm |
| `<\|sfx:sigh\|>` | 叹气 | Uh / Ahh |
| `<\|sfx:sniff\|>` | 吸鼻子 | Sff |
| `<\|sfx:sneeze\|>` | 打喷嚏 | Achoo |

#### 韵律

| Token | 效果 |
|---|---|
| `<\|prosody:speed_very_slow\|>` | 约 0.65× 语速 |
| `<\|prosody:speed_slow\|>` | 约 0.85× 语速 |
| `<\|prosody:speed_fast\|>` | 约 1.2× 语速 |
| `<\|prosody:speed_very_fast\|>` | 约 1.4× 语速 |
| `<\|prosody:pitch_low\|>` | 约 −3 半音 |
| `<\|prosody:pitch_high\|>` | 约 +2.5 半音 |
| `<\|prosody:pause\|>` | 约 400–700 ms 停顿 |
| `<\|prosody:long_pause\|>` | 约 700–1500 ms 停顿 |
| `<\|prosody:expressive_high\|>` | 更有表现力的演绎 |
| `<\|prosody:expressive_low\|>` | 更平淡的演绎 |

### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 已服务的模型 | 已服务的 Higgs TTS 模型标识符 |
| `input` | string | （必填） | 要合成的文本 |
| `voice` | string | `"default"` | 音色标识符 |
| `response_format` | string | `"wav"` | 输出音频格式（`wav`、`mp3`、`flac`、`opus`、`aac`、`pcm`） |
| `stream` | bool | `false` | 启用原始 PCM 流式 |
| `references` | list | `null` | 用于语音克隆的参考音频。每项包含 `audio_path`（本地路径、文件 URL、data URL 或 HTTP URL）与 `text`（转写文本） |
| `ref_audio` / `ref_text` | string | `null` | `references[0].audio_path` / `references[0].text` 的简写 |
| `reference_codes` | list[list[int]] | `null` | 预编码的离散 code，形状 `[T, 8]`——`references[0].audio_path` 的替代 |
| `reference_text` | string | `null` | 提供 `reference_codes` 时参考音频的转写文本 |
| `max_new_tokens` | int | `2048` | 生成的多 codebook 步数上限 |
| `temperature` | float | `1.0` | 采样温度 |
| `top_p` | float | `null` | Top-p 采样 |
| `top_k` | int | `null` | Top-k 采样 |
| `seed` | int | `null` | 用于复现的随机种子 |


### 性能

Seed-TTS EN（完整集，每次运行 **N=1088**）上的吞吐量。对 Higgs 服务器（`max_running_requests=16`、bf16、开启 CUDA Graph）做客户端 `--max-concurrency` 扫描。每行是 **3 次运行的均值**。硬件：**1× H100**。

| 并发 | 吞吐量（req/s） | 平均延迟 | RTF（单请求） | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 1.62 | 617 ms | 0.147 | 6.89 |
| 2 | 2.70 | 742 ms | 0.180 | 11.37 |
| 4 | 5.45 | 733 ms | 0.177 | 22.84 |
| 8 | 8.91 | 898 ms | 0.217 | 37.38 |
| 16 | 14.74 | 1079 ms | 0.262 | 61.84 |


- **并发** — 客户端在途请求的最大数量（`--max-concurrency`）。
- **吞吐量（req/s）** — 完成的请求数除以基准测试总墙钟时间。
- **平均延迟** — 每个请求端到端的平均时间（从发送到收到完整响应）。
- **RTF（单请求）** — 每个请求处理时间与生成音频时长的平均比值。`<1` 表示快于实时。
- **audio_s/s** — 产生的音频总秒数除以基准测试总墙钟时间。

要复现这些结果，请按照[此脚本](https://github.com/sgl-project/sglang-omni/blob/main/benchmarks/eval/benchmark_tts_seedtts.py)中的说明操作。
