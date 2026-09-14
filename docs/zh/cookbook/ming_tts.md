# Ming-Omni-TTS

[Ming-Omni-TTS-16.8B-A3B](https://huggingface.co/inclusionAI/Ming-omni-tts-16.8B-A3B)
是来自 inclusionAI 的混合专家音频生成模型。当前的 SGLang-Omni 服务路径通过
OpenAI 兼容的 `/v1/audio/speech` 端点支持**文本转语音**和**零样本语音克隆**，并生成
**44.1 kHz** 音频。

![Ming-Omni-TTS 模型架构](https://github.com/inclusionAI/Ming-omni-tts/raw/main/figures/ming_omni_tts.png)

服务流水线将 SGLang 自回归主干与 Ming 声学反馈环保持在同一个生成阶段：

```text
preprocessing -> reference_encode -> tts_engine -> audio_decode
                                      |       ^
                                      +-------+
                                       latent feedback
```

`reference_encode` 对仅文本请求是空操作。对于语音克隆，它会在自回归循环开始之前提取说话人
嵌入和提示 latent。`tts_engine` 运行 SGLang 主干、FlowLoss/CFM 声学尾部、停止头和反馈投影。
`audio_decode` 使用 Ming AudioVAE 将生成的 latent 序列转换为最终波形。

## 前置条件

按照[安装](../get_started/installation.md)的说明安装 `sglang-omni`，然后下载
checkpoint：

```bash
hf download inclusionAI/Ming-omni-tts-16.8B-A3B
```

提供的配置在 GPU 0 上使用 TP1。

## 服务器配置

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-omni-tts-16.8B-A3B \
  --config examples/configs/ming_omni_tts.yaml \
  --port 8000
```

提供的配置启用了 AR 与声学尾部 CUDA graph，以及用于流式 AudioVAE 转移的固定宽度 CUDA
graph。非流式全序列 AudioVAE 解码保持紧凑的 eager 模式，除非设置了 `stream`，请求均为
非流式。

对于非流式请求，`audio_decode` 通过一次全序列 AudioVAE 解码处理完整的生成 latent
序列。流式请求使用单独的增量 AudioVAE 路径，带有请求本地缓存和重叠状态。旧配置需要三处
修改，且这三处都在配置加载期间强制执行：从 audio_decode 阶段的 `factory` 组中移除
`decode_mode`，因为不再支持非流式分块解码；将 `tts_engine.stream_to` 设置为
`[audio_decode]` 以声明 latent 流边；并将 `audio_decode.can_accept_stream_before_payload`
设置为 `true`，使消费者能够接受在生成仍在运行时到达的 latent。提供的 YAML 已经包含全部
三处。

跨请求的非流式 AudioVAE 批处理尚未实现。唯一支持的非流式批处理配置是
`max_batch_size: 1` 加 `max_batch_wait_ms: 0`，如提供的 YAML 所示；其他值会在服务器启动
之前被拒绝。

`stream_slots` 是 AudioVAE 解码器可以同时保持活跃的流式请求最大数量。每个活跃流占用一个
槽位，以在音频分块之间保留其解码进度。如果所有槽位都被占用，额外的流会等待直到有槽位
释放。提供的配置使用 `stream_slots: 8` 来匹配其并发为 8 的工作负载。增大它可以支持更多
同时进行的流，但会使用更多 GPU 显存和固定图计算；减小它会降低这些开销，但也会降低流式
并发。它不影响非流式批处理。

## 合成语音

### 仅文本

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "SGLang-Omni is a great project!",
    "response_format": "wav"
  }' \
  --output output.wav
```

### 语音克隆

Ming-Omni-TTS 目前接受一个本地参考音频片段，并要求提供其转录文本。启动服务器时需授权
访问包含该片段的目录：

```bash
sgl-omni serve \
  --model-path inclusionAI/Ming-omni-tts-16.8B-A3B \
  --config examples/configs/ming_omni_tts.yaml \
  --allowed-local-media-path /path/to/references \
  --port 8000
```

然后将参考片段作为 `file://` URL 提交：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "file:///path/to/references/prompt.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "response_format": "wav"
  }' \
  --output cloned.wav
```

`ref_audio` 和 `ref_text` 可作为单个 `references` 条目的简写形式。

### 流式

流式返回 44.1 kHz、无文件头的单声道有符号 16 位小端 PCM（`s16le`），`Content-Type` 为
`audio/pcm`。`X-Sample-Rate`、`X-Channels` 和 `X-Bit-Depth` 头分别报告采样率、声道数和
位深；HTTP EOF 结束流。

Ming AudioVAE 使用分开的初始节奏与稳态节奏设置。它的第一个非终止调用会缓冲
`stages.audio_decode.factory.initial_chunk_patches` 个 latent patch 并且不输出 PCM；稍后
的调用会提供输出该初始组所需的右侧上下文。后续调用每次消耗 `steady_chunk_patches` 个
patch。提供的配置使用两个初始 patch 和四个稳态 patch，终止步骤会冲刷所有剩余部分。将
响应通过管道传给 `ffplay`，即可在生成过程中播放：

```bash
curl -sS --fail --no-buffer -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ming-omni-tts",
    "input": "SGLang-Omni supports streaming speech generation.",
    "stream": true,
    "response_format": "pcm"
  }' \
  | ffplay -nodisp -autoexit -f s16le -ar 44100 -ac 1 -
```

如果要保存原始流，请改用 `--output output.pcm`。该文件没有 WAV 头；使用
`ffmpeg -f s16le -ar 44100 -ac 1 -i output.pcm output.wav` 进行转换。

## 生成参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `input` | （必填） | 要合成的非空文本 |
| `references` | `null` | 至多一个带有非空 `text` 的本地参考片段 |
| `ref_audio` / `ref_text` | `null` | 参考片段及其转录文本的简写 |
| `max_new_tokens` | 省略时为 `200` | 每请求声学生成步数的上限。提供的配置接受 `1` 到 `256` 之间的值；生成可能提前停止 |
| `temperature` | 省略时为 `0.0` | FlowLoss 采样器使用的非负 SDE 温度 |
| `response_format` | `wav` | 启用 `stream` 时使用 `pcm`；参考基准测试使用 `wav` |
| `stream` | `false` | 启用时流式传输原始 PCM 音频 |
| `voice` | `default` | 仅接受默认音色选择器 |
| `speed` | `1.0` | 不支持其他语速值 |

高级 FlowLoss 控制参数可以通过 `stage_params.tts_engine` 传入：

```json
{
  "stage_params": {
    "tts_engine": {
      "cfg": 2.0,
      "sigma": 0.25,
      "temperature": 0.0
    }
  }
}
```

`cfg` 必须至少为 `1e-5` 且不能等于 `1.0`；`sigma` 和 `temperature` 必须非负。

## 基准测试

基准测试使用 Seed-TTS-Eval，并发为 8。对下表每一行分别以 `en` 和 `zh` 运行，并替换
输出目录中的 `{lang}`：

| 响应模式 | 输入模式 | 场景标志 | 输出目录 |
|---|---|---|---|
| 非流式 | 参考 | _（无）_ | `results/ming_tts/nonstream/reference/{lang}` |
| 非流式 | 仅文本 | `--no-ref-audio` | `results/ming_tts/nonstream/text_only/{lang}` |
| 流式 | 参考 | `--stream` | `results/ming_tts/stream/reference/{lang}` |
| 流式 | 仅文本 | `--stream --no-ref-audio` | `results/ming_tts/stream/text_only/{lang}` |

在 Ming-TTS 服务器运行于端口 8000 的情况下，将语言、标志和输出目录代入以下命令来生成
各个场景：

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url http://127.0.0.1:8000 \
  --model ming-omni-tts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir <output-directory> \
  --lang <lang> --ref-format references \
  --max-new-tokens 256 --max-concurrency 8 --warmup 8 \
  <scenario-flags>
```

生成完成后，停止 TTS 服务器并在另一个终端启动 ASR 服务器：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --port 8100
```

然后使用与生成时相同的语言和场景标志对每个输出目录进行转写：

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only --use-existing-server \
  --host 127.0.0.1 --port 8100 \
  --model ming-omni-tts \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir <output-directory> \
  --lang <lang> --ref-format references \
  --max-new-tokens 256 --max-concurrency 8 --asr-concurrency 1 \
  <scenario-flags>
```

## 基准测试结果

### 推荐的单 H200 TP1

推荐的 TP1 配置在 **1× H200 141 GB** 上评估，并发为 8，八个预热请求，使用完整的
Seed-TTS-Eval EN 和 ZH 拆分。流式使用两个初始 patch 加四 patch 稳态组，并启用了 AR、
声学尾部和流式 AudioVAE CUDA graph。非流式请求继续使用紧凑的全序列 AudioVAE eager
解码。

流式：

| 切片 | 语言 | 样本数 | 失败数 | 语料 WER/CER | RTF 均值 | 平均延迟（s） | 平均首音频（s） | 吞吐量（qps） | 音频 s/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 0.92% | 0.2023 | 0.955 | 0.4036 | 8.354 | 39.471 |
| text-only | ZH | 2020 | 0 | 0.67% | 0.2002 | 1.001 | 0.4040 | 7.985 | 39.991 |
| reference | EN | 1088 | 0 | 1.13% | 0.2369 | 1.053 | 0.5297 | 7.576 | 34.168 |
| reference | ZH | 2020 | 0 | 0.65% | 0.1997 | 1.146 | 0.4823 | 6.968 | 40.089 |

非流式：

| 切片 | 语言 | 样本数 | 失败数 | 语料 WER/CER | RTF 均值 | 平均延迟（s） | 吞吐量（qps） | 音频 s/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1088 | 0 | 0.90% | 0.1824 | 0.859 | 9.284 | 44.000 |
| text-only | ZH | 2020 | 0 | 0.71% | 0.1737 | 0.864 | 9.244 | 46.169 |
| reference | EN | 1088 | 0 | 1.21% | 0.2045 | 0.907 | 8.802 | 39.709 |
| reference | ZH | 2020 | 0 | 0.64% | 0.1657 | 0.948 | 8.425 | 48.396 |

全部 12,432 个请求成功完成。流式在 0.40-0.53 秒内返回首个音频负载，而非流式保持更高的
完整响应吞吐量。最差的语料 WER 为 1.21%，最差的语料 CER 为 0.71%。

流式播放连续性：

| 切片 | 语言 | 已评分 | N/A | 欠载 P95（s） | 欠载 P99（s） | C50 | C100 | C200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| text-only | EN | 1087 | 1 | 0.0000 | 0.0000 | 99.91% | 99.91% | 100.00% |
| text-only | ZH | 2020 | 0 | 0.0000 | 0.0000 | 100.00% | 100.00% | 100.00% |
| reference | EN | 1072 | 16 | 0.0000 | 0.0000 | 99.81% | 99.91% | 100.00% |
| reference | ZH | 2020 | 0 | 0.0000 | 0.0000 | 100.00% | 100.00% | 100.00% |

`N/A` 表示某个请求只返回了一个 PCM 负载，因此没有可供评分的负载间接缝。每个被评分切片
的实测播放欠载 P95 与 P99 均为零。

## 已知限制

- **服务优化。** 不支持前缀/radix 缓存和 `torch.compile`，它们在提供的配置中保持禁用。
- **参考输入。** 当前的请求适配器接受一个带有非空转录文本的本地参考音频文件；不支持远程
  URL、data URL、预计算的提示 latent 和说话人嵌入。
- **生成控制。** 不支持请求级 `seed`、logits 采样字段（`top_p`、`top_k`、
  `repetition_penalty`）、命名音色、显式语言选择、指令和时长控制。
  `initial_codec_chunk_frames` 会被拒绝，因为 AudioVAE 节奏是流水线级设置。
- **Checkpoint 覆盖范围。** 当前的服务实现仅支持 16.8B-A3B MoE checkpoint；不支持稠密的
  0.5B checkpoint。
