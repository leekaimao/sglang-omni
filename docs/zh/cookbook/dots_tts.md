# dots.tts

[dots.tts](https://huggingface.co/dots-studio/dots.tts-mf) 是 rednote-hilab 推出的文本转语音模型。它输出 48 kHz 语音，并通过一小段参考片段及其转写文本克隆说话人。

dots.tts 是连续隐变量模型，而不是 codec 模型。骨干网络不输出音频 token。每个 AR 步给出一个隐藏状态；MeanFlow DiT 用它采样一个隐变量 patch（4 帧 × 128 维）；语义编码器把该 patch 转换回下一个骨干输入；AudioVAE 把隐变量解码为波形。没有 codebook，也没有 token 采样器。所以 `temperature` 与 `top_k` 在这里不起作用——请改用求解器旋钮（`num_steps`、`guidance_scale`）。

| 组件 | 规格 |
|---|---|
| 骨干网络 | Qwen2 1.5B decoder（28 层，hidden=1536，GQA 12/2） |
| 声学尾部 | MeanFlow DiT（18 层，hidden=1024，16 头）+ VAE 语义编码器 |
| 隐变量 patch | 4 帧 × 128 维（一个 patch ≈ 160 ms 音频） |
| 上下文长度 | 2,048 token |
| 采样率 | 48 kHz |
| 求解器 | MeanFlow + Euler，引擎级 `num_steps=4`（SOAR：flow matching + CFG，`num_steps=10`） |

## 支持的 checkpoint

| Checkpoint | 状态 |
|---|---|
| [`dots-studio/dots.tts-mf`](https://huggingface.co/dots-studio/dots.tts-mf) | MeanFlow。连续批处理（continuous batching），`num_steps=4`。`examples/configs/dots_tts.yaml` |
| [`dots-studio/dots.tts-soar`](https://huggingface.co/dots-studio/dots.tts-soar) | Flow matching。一次一个请求（`max_running_requests=1`），带 CFG，`num_steps=10`。`examples/configs/dots_tts_soar.yaml` |
| [`dots-studio/dots.tts-base`](https://huggingface.co/dots-studio/dots.tts-base) | Flow matching，与 SOAR 相同。用 `examples/configs/dots_tts_soar.yaml` 加 `--model-path dots-studio/dots.tts-base` 启动 |

(dots-prerequisites)=
## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`，然后下载并启动服务器：

```bash
hf download dots-studio/dots.tts-mf --revision c28105adc8228143392b4e346994ff613ee48a06

sgl-omni serve \
  --config examples/configs/dots_tts.yaml \
  --allowed-local-media-path docs/_static/audio \
  --port 8000
```

`examples/configs/dots_tts.yaml` 把 checkpoint 锁定到验证过的快照，
因此上面的启动命令从配置读取 `model_path`。`--model-path` 覆盖该锁定，
只有在启动其他 checkpoint 或 revision 时才传入。

若要改为启动 SOAR，同时更换 checkpoint 与配置：

```bash
hf download dots-studio/dots.tts-soar

sgl-omni serve \
  --model-path dots-studio/dots.tts-soar \
  --config examples/configs/dots_tts_soar.yaml \
  --allowed-local-media-path docs/_static/audio \
  --port 8000
```

SOAR 是 flow-matching checkpoint。它运行带无分类器引导的单请求求解器，
因此其配置锁定 `max_running_requests: 1` 与 `num_steps: 10`；连续批处理仅限
MeanFlow。下面的每个请求示例在两种 checkpoint 上都可用——只有 `model` 字段
不同。

`examples/configs/dots_tts.yaml` 是权威的 MeanFlow 部署方式。它已经调优：
编译后的声学尾部与声码器（`optimize: true`，默认开启）；`max_running_requests=16`
的连续批处理；以及骨干解码 CUDA graph。只传 `--model-path` 会保留编译尾部与
批处理，但骨干解码保持 eager，单请求更慢（见[性能](dots-performance)）。请使用配置文件。

如果启动时失败并报 `dots.tts acoustic-tail admission failed at startup`，
说明显存放不下 `max_running_requests × max_generate_length` 个全长声学池——
请自行调低这些旋钮。引擎绝不会静默缩小它们。

下面的示例从 `docs/_static/audio` 读取本地片段。若要改为通过 HTTP 获取参考
音频，请放行所需域名，例如 `--allowed-media-domain huggingface.co`。

## 显存与容量

连续批处理会为每个槽位按完整生成长度预先分配声学尾部状态：

```text
patch_capacity = max_generate_length + 1
dit_cache_tokens = patch_capacity × (hidden_patch_size + latent_patch_size)   # MF: ×5
```

池的字节数大致按 `max_running_requests × patch_capacity` 缩放，包含 DiT KV（按
NFE 复制）、语义编码器 KV、scratch K/V、mask、窗口与 AdaLN 修正。启动时会打印
估算明细与空闲 CUDA 显存，当空闲显存低于估算值加 15%（留给 graph 与 workspace
的余量）时拒绝分配。

`mem_fraction_static`（`examples/configs/dots_tts.yaml` 中默认 `0.20`）只为
**SGLang 骨干**的 KV cache 做预算。声学尾部池是独立的，**不**包含在该比例内。

| 旋钮 | 对池大小的影响 |
|---|---|
| `max_running_requests` | 全长槽位的数量 |
| `max_generate_length` | `patch_capacity`（进而决定 DiT / 编码器序列长度） |
| `num_steps` | DiT KV 按每个 NFE 步复制 |

在整块 GPU 上，默认的 `16 × 500` 布局是预期用法。在较紧张的卡上，请显式调低
`max_running_requests` 与/或 `max_generate_length`，而不要指望自动缩减。当一个
在途请求需要空槽而全部繁忙时，会以 `dots.tts acoustic tail admission failed: ran out of slots` 失败——请降低客户端并发，或在显存允许时调高
`max_running_requests`。

增量 / 分桶的尾部 KV 分配尚未实现；容量仍然是全长预分配池。

## 语音合成

dots.tts 需要参考片段及其转写文本。说话人完全来自参考（x-vector 加提示词隐
变量），因此没有零样本（zero-shot）`voice` 预设。在默认的连续批处理部署下，
不带 `references` 的请求会被拒绝。

### 语音克隆

参考转写文本很重要。它会被拼在你的输入文本之前，使模型能对齐提示音频与提示
文本。错误的转写会损害克隆质量。

1. 使用 curl：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-studio/dots.tts-mf",
    "input": "Have a nice day and enjoy south california sunshine.",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "seed": 42
  }' \
  --output output.wav
```

2. 使用 Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "dots-studio/dots.tts-mf",
        "input": "Have a nice day and enjoy south california sunshine.",
        "references": [{
            "audio_path": "docs/_static/audio/male-voice.wav",
            "text": "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time.",
        }],
        "seed": 42,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

也接受 `ref_audio` / `ref_text` 作为 `references[0].audio_path` /
`references[0].text` 的简写。

### 流式输出

流式让你在生成仍在进行时就能播放音频，从而缩短首音频时间。dots.tts 流式输出
原始 48 kHz PCM：AudioVAE 解码器每隔几个隐变量 patch 就输出一个波形 chunk，
而不是等整句完成。

设置 `"stream": true` 与 `"response_format": "pcm"`：

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-studio/dots.tts-mf",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "docs/_static/audio/female-voice.wav",
      "text": "By repeating what students say, teachers can demonstrate that they are listening. By extending what students say."
    }],
    "stream": true,
    "response_format": "pcm",
    "seed": 42
  }' \
  --output output.pcm
```

`-N` 标志禁用 curl 的输出缓冲，使每个 chunk 一到达就被写入。响应是 48 kHz 的
16 位单声道 PCM，没有带内 JSON 分帧；用以下命令转换：

```bash
ffmpeg -f s16le -ar 48000 -ac 1 -i output.pcm output.wav
```

### 请求参数

顶层字段：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 已服务的模型 | 已服务的 dots.tts 模型标识符 |
| `input` | string | （必填） | 要合成的文本 |
| `references` | list | （必填） | 用于克隆的参考音频。每项包含 `audio_path`（本地路径、文件 URL、data URL 或 HTTP URL）与 `text`（转写文本）。只接受恰好一个参考 |
| `ref_audio` / `ref_text` | string | `null` | `references[0].audio_path` / `references[0].text` 的简写 |
| `response_format` | string | `"wav"` | 输出音频格式（`wav`、`mp3`、`flac`、`opus`、`aac`、`pcm`） |
| `stream` | bool | `false` | 启用原始 PCM 流式 |
| `seed` | int | `null` | 流采样器的 seed。固定给定请求的输出 |
| `language` | string | `null` | 提示词文本的语言标签；`auto` 或 `auto_detect` 表示从输入检测 |
| `instructions` | string | `null` | 风格指令；会把提示词模板切换为 `instruction_tts` |

求解器旋钮放在 `stage_params.latent_engine` 之下。它们**不是**顶层字段：
`CreateSpeechRequest` 会丢弃未知的顶层键，因此把 `num_steps` 与 `input` 并列
发送不会有任何效果，也不会报错。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `speaker_scale` | float | `1.5` | 在条件化之前缩放说话人 x-vector。更高的值更强地推向参考音色 |
| `guidance_scale` | float | `1.2` | 流采样器的无分类器引导强度 |
| `eos_threshold` | float | `0.8` | EOS 头结束生成的概率阈值。更低的值更早截断话语 |
| `num_steps` | int | `4`（MF）、`10`（SOAR） | 流求解器步数。MeanFlow 在引擎级固定该值：传其他值会使请求失败。SOAR 与 base 一次运行一个请求并尊重该值 |
| `ode_method` | string | `"euler"` | 流求解器。MeanFlow 只接受 `euler` |
| `max_generate_length` | int | `500` | 生成的隐变量 patch 上限（每个 ≈ 160 ms），受引擎的 `max_generate_length` 约束 |
| `normalize_text` | bool | `false` | 在分词前运行上游文本归一化器（数字、符号） |

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-studio/dots.tts-mf",
    "input": "Have a nice day and enjoy south california sunshine.",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "seed": 42,
    "stage_params": {"latent_engine": {"speaker_scale": 2.0, "eos_threshold": 0.6}}
  }' \
  --output output.wav
```

被拒绝的求解器取值以 HTTP 500 返回并带引擎的消息，例如 `dots.tts num_steps is fixed for continuous batching`。这是一个校验失败，不是服务器故障。

`temperature`、`top_p` 与 `top_k` 不适用。骨干 token logits 未被使用；seed 固定
之后，声学尾部是一次确定性的流求解。

(dots-performance)=
### 性能

Seed-TTS EN 基准，main commit `2b45073c`，seed 42，10 个预热请求。吞吐与延迟在
**1× H100** 上测量。服务器使用 `examples/configs/dots_tts.yaml`，启用了优化过的
声学尾部、声码器与骨干 CUDA Graph。每行都使用全部 1,088 个样本。

| 并发 | 吞吐量（req/s） | 平均延迟 | RTF（单请求） | audio_s/s | WER |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.935 | 1.070 s | 0.275 | 3.726 | 1.241% |
| 2 | 1.556 | 1.286 s | 0.314 | 6.493 | 1.256% |
| 4 | 2.493 | 1.603 s | 0.390 | 10.407 | 1.264% |
| 8 | 3.875 | 2.062 s | 0.502 | 16.173 | 1.323% |
| 16 | 4.760 | 3.344 s | 0.812 | 19.859 | 1.348% |
| 32 | 4.988 | 6.344 s | 1.596 | 20.818 | 1.331% |

所有请求均成功完成。

- **并发** — 客户端在途请求的最大数量（`--max-concurrency`）。
- **吞吐量（req/s）** — 完成的请求数除以基准测试总墙钟时间。
- **平均延迟** — 每个请求端到端的平均时间（从发送到收到完整响应）。
- **RTF（单请求）** — 每个请求处理时间与生成音频时长的平均比值。`<1` 表示快于实时。
- **audio_s/s** — 产生的音频总秒数除以基准测试总墙钟时间。
- **WER** — 生成语音的语料级词错误率，用 `Qwen/Qwen3-ASR-1.7B` 转写评分。

要复现，请按[前置条件](dots-prerequisites)启动服务器，然后对其运行基准测试：

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model dots-studio/dots.tts-mf \
  --ref-format references \
  --base-url http://127.0.0.1:8000 --port 8000 \
  --lang en --max-concurrency 16 --max-samples 1088 --warmup 10 --seed 42 \
  --generate-only --use-existing-server \
  --output-dir results/dots-seedtts-en-c16

python -m benchmarks.eval.benchmark_tts_seedtts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --model dots-studio/dots.tts-mf \
  --ref-format references --lang en --seed 42 \
  --transcribe-only --port 8000 \
  --output-dir results/dots-seedtts-en-c16
```
