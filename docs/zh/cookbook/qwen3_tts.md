# Qwen3 TTS

[Qwen3-TTS-12Hz-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) 是 Qwen 团队推出的离散多 codebook 文本转语音模型。它能从一小段参考片段快速克隆音色，支持 10 种语言，并以低延迟流式输出 24 kHz 语音。名字里的 `12Hz` 指的是编解码器（codec）的**帧率**（每秒 12 个声学帧），而不是回放采样率。SGLang-Omni 通过同一条 `preprocessing → tts_engine → vocoder` 流水线与 OpenAI 兼容的 `/v1/audio/speech` 端点提供两个 checkpoint——`0.6B` 与 `1.7B`。

## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`。

Qwen3-TTS Base 使用上游的 `qwen-tts` 包。请不带依赖安装它，以保持项目的 Transformers 5.12 / SGLang 0.5.19 技术栈不变：

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

**两行**都需要 `--no-deps`，原因各不相同。

`qwen-tts` 锁定了 Transformers 4.57.3，正常安装会替换掉项目的 5.12.1。
而正常解析 `sox` 会把 `numpy` 拉过 `numba==0.65.1` 设置的上限（numba 要求
`numpy<=2.4`）；升级后的 `numpy` 随即破坏 `librosa`，于是服务器还没启动，
`import qwen_tts` 就以 `Numba needs NumPy 2.4 or less` 失败。

也不要在这一行里加入 `onnxruntime` —— 它已经是 SGLang-Omni 的依赖，
解析它同样会把 `numpy` 拉上去。

> 在这里**不要**带依赖安装 `qwen-tts`。它声明的依赖集可能拉入与
> SGLang-Omni 运行时不同的 Transformers/Torch 技术栈。

具体而言，`qwen-tts` 0.1.1 锁定 Transformers 4.57.3，其模型代码调用的
API 有些已被 Transformers 5.12 重命名或移除——最明显的是掩码工厂
（`create_causal_mask` 及其同类），它们现在把 `input_embeds` 拼写为
`inputs_embeds` 且不再接受 `cache_position`。SGLang-Omni 在
`sglang_omni/models/qwen3_tts/compat.py` 中修补了这些差异，每个 Qwen3-TTS
入口点都会在导入 `qwen_tts` 之前应用它。因此，锁定的 Transformers 5.12 /
SGLang 0.5.19 技术栈是受支持的配置，而不是一种权宜之计。

如果你遇到从 `qwen_tts` 内部抛出的 `TypeError`，不要通过安装该包自己的
Transformers 锁定版本来解决——那会破坏运行时的其余部分。请改为上报，
以便兼容层覆盖它。

Python 的 `sox` 包在某些路径上会调用系统的 `sox` 二进制，所以两个都要安装。

下载 checkpoint（两个仓库都是公开的，无需 token）：

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

## 服务器配置

流水线为 `preprocessing → tts_engine → vocoder`。首次启动可能需要几分钟，
因为 `tts_engine` 要捕获 CUDA graph。

```bash
# 0.6B
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000
```

```bash
# 1.7B
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --port 8000
```

### 确定性推理

动态批处理可能改变 Qwen3-TTS 的 codec 与波形输出，即使提示词、参考音频与
seed 都没有变化。0.6B 与 1.7B Base checkpoint 都提供一个可选的确定性模式：

```yaml
enable_deterministic_inference: true
```

启用后，相同的提示词、参考音频与 seed 在不同运行时 batch 大小下产生逐字节一致的
PCM。该模式会降低吞吐量，因为它串行化参考预处理与声码器（vocoder）解码，
并禁用初始与后续的声码器 CUDA Graph，因此默认关闭。

### 过载 / 准入策略

两个 SGLang 生成阶段旋钮约束服务器在饱和之后的行为：

| 旋钮 | 含义 | Qwen3-TTS 默认值 |
|---|---|---|
| `--tts_engine.engine.max_running_requests` | 并发运行槽位 | `16` |
| `--tts_engine.engine.max_queued_requests` | 快速拒绝前的等待队列深度 | `16` |

每个请求都先进入等待队列，因此 `max_queued_requests` 必须 **≥ 1**。容量约为
`running + queued`。超额到达的请求在预处理之前（或稍后当 AR 等待队列或请求构建
积压满时）收到 HTTP **503**（`The request queue is full.`）。Qwen3-TTS 默认使用
4 个请求构建 worker，pending 深度为 16。

### 可回退的 prefill CUDA graph

非 Base checkpoint（CustomVoice、VoiceDesign）默认使用带 token 阶梯（上限 512）的
可回退（breakable）prefill CUDA-graph 后端：

| 旋钮 | 含义 | 默认值 |
|---|---|---|
| `--tts_engine.engine.cuda_graph_backend_prefill` | Prefill graph 后端（`breakable` 或 `disabled`） | CustomVoice 上为 `breakable`，其他地方未设置 |
| `--tts_engine.engine.cuda_graph_bs_prefill` | 要捕获的 prefill token 数阶梯 | 到 `512` 的共享阶梯，外加一个 `1` 桶 |
| `--tts_engine.engine.cuda_graph_max_bs_prefill` | 阶梯上限 | 阶梯顶端 |

默认值是共享阶梯再加一个桶。当某个桶超过真实 token 数的两倍时，replay 回退到
eager；共享阶梯从 4 开始，所以 1 token 的 prefill 落进 4 号桶并 miss。在 10 与
20 RPS 下对 3203 次 prefill 的实测中，1301 次（40.6%）恰好只有 1 个 token，
它们也是唯一回退的形状：2 与 3 已经能在 4 号桶内 replay。加上单独的 `1` 桶后，
回退率降为零。

只有 CustomVoice 采用该默认值，由 checkpoint 的 `tts_model_type` 选择。Base 的
prefill 也携带参考音频，形状分布不同，而 VoiceDesign 尚未测量；两者都保持
eager 路径。

用 `--tts_engine.engine.cuda_graph_backend_prefill disabled` 退出。该默认值在
启动时带来额外的 graph 捕获开销。单独调高 `cuda_graph_max_bs_prefill` 会把默认
阶梯重新生长到新上限；自己声明 `cuda_graph_bs_prefill` 则完全保留你的列表。

调高 `max_running_requests` **不会**自动调高等待上限。对一个上限为 32 的实验：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --tts_engine.engine.max_running_requests 32 \
  --tts_engine.engine.max_queued_requests 16 \
  --port 8000
```

阶梯式的 `--concurrencies` 是一个闭环客户端：它在途请求从不超过
N 个，因此超过上限的负载只是一个会排空的突发。要让供给负载持续超过
`max_running_requests + max_queued_requests` 一段时间，请使用开环的持续过冲：

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server --stream \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --port 8000 \
  --max-running-requests 32 \
  --max-queued-requests 16 \
  --sustained-overshoot \
  --overshoot-duration-s 10 \
  --max-samples 64
```

到达率默认为 `2 × capacity`（可用 `--request-rate` 覆盖）。统计只计成功请求；
产物落在 `<output-dir>/overshoot/`。

闭环的 `--concurrencies 16,32,48,64` 扫描仍可用于对比健康点与超上限点，
但它不会保持过冲。每个并发的可检查产物写在 `<output-dir>/c<N>/` 下。

### Prefill 准入合并

在并发负载下，`tts_engine` 阶段可以合并 prefill 准入：调度器不再把每个就绪请求
各自放入一个 prefill batch，而是可以短暂扣住准入，让多个就绪请求一起被
prefill。

一个 prefill 步骤的调度器开销基本固定，因此更满的 batch 能降低 prefill 开销。
端到端的收益取决于这部分节省是否能盖过额外的准入延迟以及解码占用率的下降。

合并**默认关闭**，通过 `tts_engine` 的 factory 配置开启：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --tts_engine.factory.prefill_coalesce_requests 2 \
  --tts_engine.factory.prefill_coalesce_wait_ms 30 \
  --port 8000
```

或在 YAML 中按阶段写：

```yaml
stages:
  tts_engine:
    factory:
      prefill_coalesce_requests: 2
      prefill_coalesce_wait_ms: 30.0
```

只有当 `prefill_coalesce_requests >= 2` 时该闸门才启用。启用后，只要满足以下
任一条件，准入即被释放：

- 解码空闲，就绪请求可以立即开始；
- 等待队列达到 `prefill_coalesce_requests`；
- 最早的等待请求已等待 `prefill_coalesce_wait_ms`。

因此 `prefill_coalesce_wait_ms` 是新增准入等待的上界。如果目标队列大小提前达到，
准入可能更早释放。

上面的取值只是 Qwen3-TTS 负载的一个示例，并不是普适默认值。请把
`prefill_coalesce_requests` 与 `prefill_coalesce_wait_ms` 都匹配到你实际服务的
负载。合并在天然 prefill batch 较小、短暂扣留能明显提升批处理而基本不降低解码
占用率时最有用。如果等待过长，解码占用率的下降可能抵消 prefill 的节省。

对延迟敏感的流量，或额外等待换不来足够批处理增益的负载，请保持合并关闭。

### 进程拓扑

默认情况下三个阶段共享一个进程。按请求的参考预处理（speech-tokenizer 编码、
说话人 embedding、提示词 embedding）于是与 AR 调度器、声码器争夺同一个解释器，
当并发超过约 32 时，这会封顶单副本吞吐量。把预处理阶段挪到独立进程可消除该争用：
该阶段加载一个仅前端的组件（embedding 表、文本投影、predictor codec embedding、
说话人编码器）外加 speech tokenizer，在 1.7B checkpoint 上额外进程约占
2.2 GB GPU 显存，并通过负载把准备好的提示词张量送进引擎。此时每个 GPU 阶段都
必须声明显存比例，且引擎的静态比例必须与它声明的一致：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --preprocessing.process tts_frontend \
  --preprocessing.gpu 0 \
  --preprocessing.gpu_memory_fraction 0.05 \
  --tts_engine.gpu_memory_fraction 0.75 \
  --tts_engine.engine.mem_fraction_static 0.75 \
  --vocoder.gpu_memory_fraction 0.12
```

`--vocoder.process vocoder` 可与之组合（把 `tts_engine` 降到 0.72，给声码器
0.15）。固定 seed 下六条 SeedTTS 样本在两种布局中产生了相同的 PCM；额外进程的
代价是它的 CUDA 上下文加上前端权重。

## 语音合成

### 纯文本请求

Qwen3-TTS Base checkpoint 必须提供参考片段。纯文本请求由 CustomVoice 与
VoiceDesign checkpoint 支持；对应的启动命令见 [TTS 模型使用](../basic_usage/tts.md)。

### 语音克隆

`references` 字段接受 `audio_path`（本地路径或 HTTP URL）与 `text`（该片段的
转写文本）。提供转写文本会启用上下文学习（ICL）模式，能显著提升克隆质量；省略
则回退到说话人 embedding（x-vector）模式。

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "default",
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

也接受 `ref_audio` 与 `ref_text` 作为 `references[0].audio_path` 与
`references[0].text` 的简写。

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "voice": "default",
        "input": "Get the trust fund to the bank early.",
        "references": [{
            "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
            "text": "We asked over twenty different people, and they all said it was his.",
        }],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

非流式响应在 codec EOS 之后带 `X-Finish-Reason: stop`；当生成到达
`max_new_tokens` 时带 `X-Finish-Reason: length`。`length` 响应仍包含可解码的
音频，但话语可能不完整。批量响应把相同的值暴露为每一项的 `finish_reason`。

### 语言提示

`language` 把模型偏置到目标语言。默认为 `auto`（让模型自动检测）。支持的语言有
中文、英语、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语与意大利语。

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "default",
    "input": "今天天气不错，就该出去晒晒太阳。",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "language": "Chinese"
  }' \
  --output output.wav
```

### 流式输出

设置 `"stream": true` 与 `"response_format": "pcm"` 即可实时接收原始 PCM 音频
chunk：

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
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

流式返回 `audio/pcm` 的 16 位单声道 PCM 字节，采样率元数据在响应头中。完整的
Python 原始 PCM 消费端见 [Higgs TTS 实战指南](streaming)。

全部三种任务类型（Base/参考克隆、CustomVoice 与 VoiceDesign）对 HTTP 端点与带
`stream_audio=true` 的 `/v1/audio/speech/stream` WebSocket 会话都使用真正的增量
codec 与声码器流式。在请求上传入 `"stream_codec_output": false`，或以
`--preprocessing.factory.stream_codec_output false` 启动，可恢复整句解码。

在经过验证的话人/语言组合上（当前为默认采样下的 Ryan/英语），流式 CustomVoice
输出会扣住模型静音的 bootstrap codec 帧，从第一个 chunk 中去掉约 80 ms 的前导
静音。该帧仍会喂给声码器，因此后续每个样本都不变；运行时静音检查会在首帧实际
并非静音时原样放出音频。按请求用 `"suppress_bootstrap_silence": false`，或按部署
用 `--vocoder.factory.suppress_bootstrap_silence false` 退出。

省略 `initial_codec_chunk_frames` 时，Qwen3-TTS 会让前几个 chunk 按
`1 -> 2 -> 4` codec 帧爬坡再进入稳定步长，因此首音频在单个 AR 步之后即可离开，
同时播放缓冲在四个 chunk 内重建。传入显式值可在连续性与首音频时间之间取舍。
生成 codec 帧数少于首个 chunk 的话语永远到不了第一个 chunk，因此它们的音频在
最后一次冲刷中完整送达。

#### codec 解码默认值

流式解码默认运行在有状态增量 codec 上：每个后续 chunk 只针对预分配 arena 中
保存的按流状态解码其新增帧，稳态批次 replay 解码步被 `torch.compile` 编译的
CUDA graph，后续 worker 收集 4 ms。启动时花约一分钟编译稳态形状。左上下文
解码器仍可作为回退使用：

```yaml
stages:
  vocoder:
    factory:
      enable_stateful_codec_decoder: false
```

`incremental_codec_cuda_graph`、`incremental_codec_compile` 与
`followup_batch_wait_ms` 是各自的独立开关。在一张 H100
80GB 上按每秒 20 个请求、三个客户端 seed（各约 1200 个请求）实测：默认路径的
流断流率为 0.6%–2.3%，而左上下文解码器为 20.9%；首个可播放音频为 55–58 ms，
后者为 82–89 ms。

(first-audio-chunk-ramp)=
#### 首音频 chunk 爬坡

对延迟敏感的部署，整条早期 chunk 调度可以在服务端通过声码器阶段的
`stream_chunk_ramp` 配置：第 `i` 项决定流式解码第 `i + 1` 个 chunk 的 codec
帧数，越过爬坡后由稳定步长接管，因此 `[2, 4, 8]` 给出
`2 -> 4 -> 8 -> 8 -> ...` 的调度。通过流水线配置文件设置：

```yaml
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-0.6B-Base
stages:
  vocoder:
    factory:
      stream_chunk_ramp: [2, 4, 8]
```

```bash
python -m sglang_omni.cli serve --config qwen3_tts_ramp.yaml
```

更小的早期 chunk 降低首音频时间，但开始播放时缓冲的音频更少，因此连续性代价随
并发增长。默认的 `[1, 2, 4]` 是最激进的调度，适合低并发；中等并发建议
`[2, 4, 8]`，饱和服务建议 `[4, 8]`，此时更宽的首 chunk 换回播放缓冲。该爬坡与
旧式的 `initial_chunk_frames` / `stream_initial_followup_stride` 选项互斥，
其首项不得超过稳定步长，且按请求的 `initial_codec_chunk_frames` 仍只覆盖第一个
chunk。

## 生成参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | 已服务的模型 | 已服务的模型标识符 |
| `input` | （必填） | 要合成的文本 |
| `voice` | `default` | 音色标识符。对 Base 参考克隆而言，说话人条件由参考片段提供 |
| `references` | `null` | 用于克隆的参考片段。每项包含 `audio_path` 与 `text` |
| `ref_audio` / `ref_text` | `null` | `references[0].audio_path` / `references[0].text` 的简写 |
| `language` | `auto` | 目标语言提示（见上文列表） |
| `temperature` | `0.9` | 采样温度 |
| `top_p` | `1.0` | Top-p 采样 |
| `top_k` | `50` | Top-k 采样 |
| `repetition_penalty` | `1.05` | 重复惩罚 |
| `max_new_tokens` | `2048` | 生成的 codec token 上限 |
| `seed` | `null` | 用于复现的随机种子 |
| `stream` | `false` | 流式返回原始 PCM 音频 chunk |
| `initial_codec_chunk_frames` | 省略时爬坡 `1 -> 2 -> 4` | 首个流式声码器 chunk 的 codec 帧数。显式值只替换爬坡的首个 chunk。更小的值降低 TTFA 但更容易断流；`0` 从一开始就使用稳定步长 |
| `stream_codec_output` | `true` | codec 帧生成即转发给声码器。设为 `false` 可恢复 CustomVoice/VoiceDesign 的整句解码 |
| `suppress_bootstrap_silence` | `true` | 从流式 CustomVoice 输出中扣住静音 bootstrap codec 帧的音频（仅限经过验证的话人/语言组合，并有运行时静音检查兜底）。设为 `false` 保留前导静音 |

## 模型变体

| Checkpoint | 参数量 | 配置 |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | `examples/configs/qwen3_tts_0_6b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | `examples/configs/qwen3_tts_1_7b.yaml` |

两者暴露完全相同的请求 API。1.7B 模型容量更高（通常质量更好），但显存与延迟
代价更大；0.6B 模型更轻、更快。

(customvoice-checkpoints)=
## CustomVoice 检查点

CustomVoice 通过同一条流水线以内置说话人生成语音。使用时不带参考音频；省略
`ref_audio`、`ref_text`、`references` 与 `x_vector_only_mode`。省略 `task_type`
或将其设为 `CustomVoice`。

| Checkpoint | 配置 | instructions 指引 |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `examples/configs/qwen3_tts_0_6b_customvoice.yaml` | 为兼容而接受，但不推荐 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `examples/configs/qwen3_tts_1_7b_customvoice.yaml` | 受支持 |

两个已发布的 checkpoint 都提供 `Serena`、`Vivian`、`Uncle_Fu`、`Ryan`、`Aiden`、
`Ono_Anna`、`Sohee`、`Eric` 与 `Dylan`。说话人匹配不区分大小写；省略或使用
`default` 音色会选择 `Vivian`。`GET /v1/audio/voices` 列出 `default` 与已服务
checkpoint 的说话人。未知说话人或提供克隆字段返回 HTTP 400；上传的参考音色不会
用于 CustomVoice 合成。

两种规模都支持缓冲语音、批量请求、增量 HTTP PCM 输出与 WebSocket 音频输出。HTTP
流式要求 `stream=true` 且 `response_format="pcm"`；WebSocket 会话使用
`stream_audio=true` 与 `response_format="pcm"`。

**0.6B 的 instructions 兼容性：** SGLang-Omni 仍会把可选的 `instructions` 传入
0.6B 的提示词，以保持既有行为。已发布的 0.6B 模型不提供可靠的指令控制；需要风格
控制时请省略该字段或使用 1.7B。

**Eric/Dylan 的语言行为：** 对两种规模，`language: Auto` 会选择 Eric 的四川话
token 或 Dylan 的北京话 token。显式语言优先：`language: Chinese` 保持中文语言
token。这保持了 SGLang-Omni 的既有行为，与 QwenLM/Qwen3-TTS 的 Python 封装
（`qwen-tts` 0.1.1）不同——后者对 `Chinese` 也会选择方言 token。这是一个条件化
选择，并不保证说话人的口音消失。

用配套配置启动 1.7B checkpoint：

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000
```

然后在请求中选择一个内置说话人。对 0.6B，使用表中的模型/配置对并省略
`instructions`。

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "SGLang-Omni serves Qwen CustomVoice.",
    "voice": "Ryan",
    "language": "English",
    "instructions": "Speak clearly and calmly."
  }' \
  --output custom-voice.wav
```

## 基准测试结果

### 0.6B Base

Qwen3-TTS-12Hz-0.6B-Base 在 Seed-TTS EN 上（1088 条话语，每条从其提示词做参考
音色克隆），并发 16，用 HF Whisper-large-v3 评分。硬件：1× H200 SXM。

| 指标 | 数值 |
|---|---|
| WER（语料级，剔除失控离群值） | 1.07% |
| WER（按样本中位数 / p95） | 0.00% / 9.09% |
| WER（语料级微平均，原始） | 18.29% |
| 失控样本（WER > 50%） | 2 / 1088（0.2%） |
| 延迟均值 / 中位数（s） | 6.61 / 6.24 |
| RTF 均值 / 中位数 | 1.51 / 1.48 |
| 输出吞吐（tok/s） | 115.4 |
| 完成 / 失败请求 | 1088 / 0 |

典型输出很干净（WER 中位数 0.00%，p95 9.09%）。两条话语（0.2%）失控进入重复
循环，直到 `max_new_tokens` 生成了约 164 秒的循环音频，仅此就把原始微平均抬到
18.29%；剔除后语料级 WER 为 1.07%。RTF > 1 反映的是并发 16 下 0.6B codec 流水
线的表现，而不是单流延迟。1.7B checkpoint 用延迟换质量。

### 1.7B CustomVoice

Qwen3-TTS-12Hz-1.7B-CustomVoice 在完整的 Seed-TTS-Eval EN 与 ZH 划分上，并发
16，每种语言/模式 16 个预热请求，`max_new_tokens=2048`。EN 使用 Ryan/英语，ZH
使用 Vivian/中文，不带参考音频与 instructions。WER/CER 用 Qwen3-ASR-1.7B 在并发
32 下评分。硬件：1× H200 141 GB，BF16，TP1。采样覆盖与 seed 均未设置。

服务器使用 `--tts_engine.engine.max_running_requests 64`、
`--tts_engine.engine.cuda_graph_max_bs 64`、
`--tts_engine.engine.torch_compile_max_bs 64`、`--vocoder.process vocoder`、
`--tts_engine.gpu_memory_fraction 0.85` 与
`--vocoder.gpu_memory_fraction 0.10`；`torch.compile` 保持禁用。流式使用默认的
`1 -> 2 -> 4` chunk 爬坡，无请求级覆盖。每种语言/模式在同一预热服务器上按非流式
EN/ZH、流式 EN/ZH 的顺序各测量一次。计时窗口内目标 GPU 上没有外部 GPU 进程；
宿主 CPU、内存与 I/O 与另一个 profiling 任务共享。

| 指标 | 非流式 EN | 非流式 ZH | 流式 EN | 流式 ZH |
|---|---:|---:|---:|---:|
| 样本数 | 1088 | 2020 | 1088 | 2020 |
| 语料级 WER/CER | 1.608% | 0.984% | 2.085% | 0.927% |
| 语料级 WER/CER（剔除 >50% 离群值） | 1.359% | 0.984% | 1.454% | 0.927% |
| WER/CER 超过 50% 的样本 | 3 | 0 | 4 | 0 |
| UTMOS | 4.1723 | 3.1824 | 4.1500 | 3.1789 |
| QPS | 14.788 | 13.307 | 10.098 | 8.333 |
| 延迟均值（s） | 1.075 | 1.198 | 1.573 | 1.915 |
| RTF 均值 | 0.2335 | 0.2079 | 0.3380 | 0.3332 |
| TTFA 均值（s） | N/A | N/A | 0.1213 | 0.1047 |

语料级 WER（EN）/ CER（ZH）是总编辑距离除以总参考词数 / 字数，并包含全部样本。
过滤行剔除自身 WER/CER 超过 50% 的样本后重新计算语料级比例。UTMOS 是平均的音频
质量预测分，不是听测分。这些独立采样的运行并不是流式与非流式质量的配对比较；
Base 结果也使用了不同的条件化方式与不同的 ASR 评分器。

TTFA 度量第一个 PCM 负载的到达，而不是第一段可听语音；其平均负载时长在两种语言
中均为 80 ms。全部 3,108 个流式请求都做了连续性评分：98.99% 的 EN 与 93.76% 的
ZH 没有超过 50 ms 的播放断流。最大断流 EN 为 396.1 ms，ZH 为 2072.0 ms。因此
默认爬坡在并发 16 下并不能保证不间断播放；缓冲权衡见
[首音频 chunk 爬坡](first-audio-chunk-ramp)。

## 已知限制

- **建议使用参考音频。** 作为克隆模型，Qwen3-TTS Base 在没有参考片段时会产生
  机器人式的语音。
- **转写文本提升克隆效果。** 在 `references` 中提供 `text`（ICL 模式）比仅用
  说话人 embedding（x-vector）模式获得更好的说话人相似度。
- **语言检测。** `language: auto` 对简短或混码输入可能误检；已知目标语言时请
  显式设置 `language`。
- **偶发生成失控。** 约 0.2% 的话语（在 0.6B checkpoint 上观察到）可能陷入重复
  循环，一直生成到 `max_new_tokens`。调高 `repetition_penalty`（默认 `1.05`）
  或调低 `max_new_tokens` 可以缓解；1.7B checkpoint 更不易发生。
