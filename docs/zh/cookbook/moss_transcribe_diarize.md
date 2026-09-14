# MOSS-Transcribe-Diarize

[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) 是 OpenMOSS 团队推出的多说话人 ASR 与说话人分离（diarization）模型。

MOSS-Transcribe-Diarize 不支持 `/v1/audio/translations`；该端点返回 HTTP 400。请使用 `/v1/audio/transcriptions`。

![Model Architecture](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/resolve/main/Model_Architecture.png)

它在一次生成中同时完成转写、说话人归属与时间戳预测。凭借 128K 上下文，它支持最长约 90 分钟的音频，能处理会议、插话、长对话与重叠语音，并提供针对姓名、公司、产品术语与领域词汇的热词增强（hotword boosting）。MOSS-Transcribe-Diarize 通过 OpenAI 兼容的 `/v1/audio/transcriptions` 端点提供服务。
该模型的转写支持带真实分段时间戳的 `response_format=srt` 或 `response_format=vtt`。

| 组件 | 规格 |
|---|---|
| 架构 | `MossTranscribeDiarizeForConditionalGeneration` |
| 音频编码器 | Whisper encoder（24 层，d_model=1024） |
| 文本解码器 | Qwen3（28 层，hidden=1024，GQA 16/8） |
| 输出 | 带起止时间戳的说话人标注转写 |
| 端点 | `/v1/audio/transcriptions` |

## 架构与优化

MOSS-TD（MOSS-Transcribe-Diarize 的简称）通过 SGLang-Omni 服务有两个原因。第一，SGLang-Omni 的多阶段流水线是天然契合——ASR 与框架已为 TTS 编排的 encoder → prefill → decode 模式相同。第二，ASR 是多模态输入模型的一类，我们在 SGLang-Omni 中构建的许多优化（CUDA Graph 捕获、异步解码、连续批处理、KV cache 管理）可以直接迁移到 ASR 场景。

### 推理流水线

![ASR Pipeline](../_static/image/moss-td-asr-pipeline.svg)

MOSS-TD 遵循 Audio LLM 模式：Whisper 编码器产生连续 embedding，经投影进入一个 decoder-only LLM，自回归地生成转写。

1. **编码器。** 波形 → log-mel 频谱（80 bin）→ 24 层 Whisper Transformer → 4× 时间合并（每 4 帧拼接）→ VQAdaptor MLP（4096→1024）。输出是 LLM embedding 空间中的连续浮点向量序列。（"VQ"这个名字名不副实——并不涉及向量量化。）
2. **LLM Prefill。** 音频 embedding 替换提示词中的 `<|audio_pad|>` token。Qwen3 用一次并行前向处理完整提示词，构建 KV cache。
3. **AR 解码。** Qwen3 逐个生成文本 token（带说话人标签与时间戳的转写），直到 EOS。

对长音频（至多约 90 分钟），编码后的 token 序列可达数万 token。**分块 prefill（Chunked Prefill）**把它切成 4096 token 的块，每个调度步处理一块，并在块之间穿插其他请求的解码步。分块 prefill 期间会抑制流式输出，避免发出中间状态。

### ASR 与 TTS

ASR 与 TTS 在 sglang-omni 中共享大量服务基础设施——两者都用 `OmniScheduler` 调度、共享 CUDA Graph / KV Cache 管理 / 连续批处理，且构建在同一个 Qwen3 LLM 骨干上。关键差异在于编码什么、生成什么，以及流水线的结构：

| 维度 | ASR（MOSS-TD） | TTS（Higgs / MOSS-TTS） |
|---|---|---|
| 音频表示 | 连续特征（mel 频谱 → 编码器隐藏状态） | 离散 codec token（RVQ 多 codebook 编码） |
| 数据流 | 音频 → 文本 | 文本 → 音频 |
| 解码器 / 声码器 | 不需要——输出是纯文本 | 需要声码器从 codec token 重建波形 |
| 典型输入长度 | 可以很长（MOSS-TD 支持约 90 分钟） | 通常很短（参考音色：几秒） |
| 流水线阶段 | 单阶段（编码器 + LLM） | 多阶段（preprocessing → AR 引擎 → 声码器） |
| 流式 | 流式输出（增量文本）；流式输入可通过累积分块实现，但尚未优化——支持原生流式输入/输出的新编码器架构正在开发中 | 流式输出（增量音频）+ 流式声码器 |

这些差异决定了优化重点：ASR 优化聚焦 AR 解码循环（延迟的主要来源）与长序列内存管理，而 TTS 优化还针对声码器批处理/流式与多 codebook 生成策略。完整的 TTS 优化故事见 [Optimizing TTS Inference](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/omni/tts-optimization.md)。

### 时间花在哪里

低并发时 AR 解码占主导，但高并发下解码被批处理摊薄，编码器占比上升——短音频尤其明显。

在单张 H100 上的 profiling（CUDA Graph、bf16），展示三个阶段的占比分解：

| 音频长度 | 并发 | 编码器 | LLM Prefill | AR 解码 |
|---:|---:|---:|---:|---:|
| 5 s | 1 | 8.9% | 14.7% | 76.4% |
| 5 s | 4 | 20.0% | 22.3% | 57.7% |
| 5 s | 16 | 38.2% | 29.7% | 32.1% |
| 60 s | 1 | 4.0% | 2.1% | 94.0% |
| 60 s | 4 | 5.0% | 4.6% | 90.4% |
| 60 s | 16 | 13.7% | 9.5% | 76.8% |
| 20 min | 1 | 4.7% | 0.8% | 94.5% |
| 20 min | 4 | 9.2% | 1.9% | 88.9% |
| 20 min | 16 | 11.6% | 2.6% | 85.7% |

![Profiling Breakdown](../_static/image/moss-td-profiling.svg)

c=1 且音频较长时，AR 解码占总时间 94% 以上——杠杆几乎全在解码循环。c=16 且音频较短时，编码器 + prefill 合计占 68%，使编码器侧优化（CUDA Graph 捕获、Torch Compile、缓存）值得投入。

### 优化策略

![Optimization Overview](../_static/image/moss-td-optimization.svg)

优化栈与[我们为 TTS 构建的](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/omni/tts-optimization.md)一脉相承，共享同一套核心基础设施并做了 ASR 专属适配。

**CUDA Graph。** LLM 解码步把 batch 大小填充到预定义桶（1、2、4、8、……）并 replay 捕获的 CUDA graph，消除每个 token 的 kernel 启动开销。这是 AR 解码最大的一项优化。可回退的 prefill graph 则按 token 数分桶，阶梯从 1 和 2 开始：完全缓存的前缀仍要 re-prefill 其最后一个 token，而该 1 token 的 extend 否则会被填充到 4 token 下限、超出 SGLang 的 2 倍填充保护而回退到 eager。2 token 桶恰好落在该保护线上，因此无论如何都会 replay，只省下填充。Whisper 编码器得到同样的处理，按块数分桶（`encoder_chunk_buckets`，默认 `1..8` ≈ 4 分钟音频）。

**解码器 Torch Compile。** 默认流水线把 Qwen3 解码器形状编译到 batch 大小 4，超过该上限使用 eager 解码器。用 `--torch-compile-max-bs` 覆盖上限，或用 `--torch-compile off` 禁用解码器编译。编译在启动时对每个捕获的解码桶运行一次（`max-autotune-no-cudagraphs`），因此冷启动在服务器接收流量前要付出 autotuning 代价。该设置与下面的编码器编译选项相互独立。

**编码器 Torch Compile（可选）。** `encoder_torch_compile=True` 把编码器 CUDA graph 换成带 kernel 融合的 `torch.compile`（默认模式）。两者互斥。不得使用 reduce-overhead 模式：其 cudagraph 树会与该进程中始终运行的解码 CUDA graph 一起破坏内存（服务约 60 秒后出现非法内存访问）。代价是启动时每个桶一次性的编译；`dynamic=False` 意味着只有预热过的块数被加速，其余走 eager。

**异步解码。** 与 TTS 相同的一步前瞻：发起当前解码步的 GPU 工作，然后并行处理上一步的宿主侧工作（D2H 拷贝、完成检测、结果分发）。MOSS-TD 默认从 batch 大小 1 起启用前瞻。设置 `--async-lookahead-min-batch-size 2` 让 batch 大小 1 的解码保持同步，或用 `--decode-mode sync` 对该阶段禁用前瞻。两个交替使用的 pinned 宿主缓冲避免 GPU 异步 D2H 写与 CPU 读之间的竞争。完整机制与代码入口见 TTS 优化指南中的 [Asynchronous Decode + Lookahead](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/omni/tts-optimization.md#asynchronous-decode--lookahead)。

**LRU 编码器缓存。** Whisper 编码器前向对相同输入音频是确定性的——相同波形总是产生相同 embedding。我们利用这一点实现了一个 LRU 缓存（最多 64 条，4 GB 预算），把编码器输出存放在 CPU 上，以输入波形的内容哈希为键。缓存命中时，存储的张量被异步传回 GPU，完全跳过编码器。未命中时编码器正常运行，结果被搬到 CPU 存储。缓存同时按条目数与总字节数淘汰，总是先淘汰最久未用的条目。

与 TTS 中同一参考音色被许多提示词复用（高命中率）不同，ASR 输入在生产中通常是唯一的。该缓存对请求重试、用不同解码参数做 A/B 测试以及开发迭代最有用。

**流式输出。** 在 AR 解码期间通过 SSE 增量发出转写文本，让用户在生成过程中看到部分结果，而不必等待完整序列。三个机制控制何时发出：

1. **限流**（默认 50 ms）：token 累积在按请求的缓冲中，只有距上次发出足够时间后才冲刷。第一个 token 立即发出；EOS 无论计时如何总是触发冲刷。
2. **分块 prefill 抑制**：分块 prefill 期间（提示词块仍在处理），抑制所有发出，防止中间状态被误读为转写输出。
3. **不完整 UTF-8 处理**：累积的 token 一起解码。如果结果以 Unicode 替换字符结尾（表示跨 token 边界拆开的不完整多字节序列），则扣住不发，直到下一个 token 补全该序列。

## 模型使用

### 启动命令

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`，然后下载模型：

```bash
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize
```

启动模型服务：

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --port 8000 \
  --asr.engine.max_running_requests 16 \
  --asr.engine.cuda_graph_max_bs 16 \
  --mem-fraction-static 0.80
```

MOSS-TD 会短暂扣住新构建的 LM 请求以准入更大的 prefill。默认目标是 4 个请求、
最早请求截止 12 ms。还有更多请求构建未完成时，调度器等待两个限制之一；构建
工作排空后，只有解码空闲才立即释放。解码活跃期间，它会继续合并直到目标或
截止。用 `--prefill-coalesce-requests` 与 `--prefill-coalesce-wait-ms` 覆盖
两个限制，或把请求目标设为 `0` 禁用合并。

### 发送请求

需要解析的说话人分段时使用 `response_format=verbose_json`。`json` 只返回原始
转写文本。

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=verbose_json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "response_format": "verbose_json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
payload = resp.json()
print(payload["text"])
for segment in payload.get("segments", []):
    print(
        f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}"
    )
```

请求省略 `max_new_tokens` 时，服务器会根据音频时长双向定输出预算：默认配置
为 `max(512, 每音频秒 10 token)`，因此 60 分钟录音无需客户端改动即获得 36000
token 预算，而 6 秒片段被限制在 512 token，而不是继承旧的固定 5120 默认——
这避免了贪心解码在短的非语音音频上循环数千 token（#975），又不会截断密集的、
带时间戳的多说话人转写。零时长输入使用更紧的 128 token 回退，因为它没有值得
保留的转写。两个上限都不会超过运维者固定的更小默认值。表单还接受
`repetition_penalty`（0 < x <= 2，默认 1.0 = 关闭），在不触碰贪心默认的前提下
抑制嘈杂音频上的重复循环。运维者可以在阶段配置中固定 `max_new_tokens`，这会
对省略该字段的请求禁用时长缩放。请求中的显式 `max_new_tokens` 总是优先于两种
默认。调度器会把最终值钳制到音频提示词之后的剩余上下文，因此发送较大的显式
值是安全的。想要硬上限或比默认更大的预算时，请显式设置该字段，如下例使用
仓库中一个有两个说话人的片段：

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@docs/_static/audio/gaokao-listening.wav \
  -F response_format=verbose_json \
  -F max_new_tokens=65536
```

### 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `file` | file | 必填 | 以 multipart 表单数据上传的音频文件 |
| `model` | string | 服务器默认值 | 模型标识符 |
| `language` | string | 未设置 | 可选的语言提示 |
| `response_format` | string | `json` | `json`、`verbose_json` 或 `text` |
| `temperature` | float | 模型默认值（`0.0`） | 采样温度 |
| `max_new_tokens` | int | 按时长缩放 | 生成 token 上限。默认配置下，省略该字段的请求使用 `max(512, 10 × 音频秒数)`；空音频使用 128 token 回退。固定的阶段值会禁用时长缩放。显式值总是优先，并被钳制到剩余模型上下文 |
| `prompt` | string | 未设置 | 可选的指令覆盖；省略则使用内置的转写+分离提示词 |

`verbose_json` 把模型标记解析为 OpenAI 风格的 `segments`，含 `start`、`end`
与带说话人前缀的 `text`（例如 `[S01]...`）。`json` / `text` 返回完整转写字符
串，不做分段解析。

## 基准测试

感谢 Moss 团队提供的基准数据集，我们准备了 movies800times、aishell4_long 与
googletime 作为多说话人 ASR 基准数据集。movies800times 是含 800 段对话的短
序列数据集，aishell4_long 与 googletime 分别是长篇会议音频与英语播客的长序列
数据集。这些数据集目前为私有许可，可联系 Moss 团队获取访问权限。


```bash
# Short-sequence ASR / diarization
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset movies800times \
  --concurrency 16 \
  --asr.engine.max_running_requests 16 \
  --asr.engine.cuda_graph_max_bs 16 \
  --mem-fraction-static 0.80 \
  --output-dir results/moss_transcribe_diarize_movies800times

# note (Xinyu): Add one dedicated request-event profiling pass after the measured
# evaluation. This pass is excluded from reported accuracy and speed metrics.
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset movies800times \
  --concurrency 16 \
  --profile-events \
  --profile-event-dir /tmp/moss_td_bench_profile \
  --output-dir results/moss_transcribe_diarize_movies800times_profile

# Long-sequence ASR / diarization
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset aishell4_long \
  --concurrency 16 \
  --asr.engine.max_running_requests 16 \
  --asr.engine.cuda_graph_max_bs 16 \
  --mem-fraction-static 0.80 \
  --max-new-tokens 65536 \
  --request-timeout-s 1800 \
  --output-dir results/moss_transcribe_diarize_aishell4_long

# Long-sequence English podcast ASR / diarization
python -m benchmarks.eval.benchmark_asr_transcribe_diarize \
  --dataset googletime \
  --concurrency 16 \
  --asr.engine.max_running_requests 16 \
  --asr.engine.cuda_graph_max_bs 16 \
  --mem-fraction-static 0.80 \
  --max-new-tokens 65536 \
  --request-timeout-s 1800 \
  --output-dir results/moss_transcribe_diarize_googletime
```

`--profile-events` 通过 serve 的 profiler 端点启动请求级事件记录，多跑一轮，
并在 `transcribe_diarize_results.json` 与
`transcribe_diarize_speed_results.json` 的 `profile` 下加入其 `stage_breakdown`、
`hop_breakdown` 与速度指标。事件目录是服务器端路径，因此报告生成要求基准进程
看到同一个文件系统。对 router 或 DP 部署，请用逗号分隔的 `--profile-urls`
传入每个 worker 的 serve URL。

## 基准测试结果

这里给出 movies800times 与 aishell4_long 在单张 H100 80GB 上的基准结果。每行
是对 `max_running_requests=16`、`cuda_graph_max_bs=16`、
`mem_fraction_static=0.80` 服务器的 **3 次运行均值**。

### movies800times

| 并发 | 吞吐量（req/s） | 平均延迟（s） | RTF 均值 | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 2.57 | 0.388 | 0.0612 | 29.76 |
| 2 | 4.89 | 0.409 | 0.0659 | 56.55 |
| 4 | 6.62 | 0.513 | 0.0790 | 76.64 |
| 8 | 6.80 | 0.533 | 0.0810 | 78.70 |
| 16 | 7.08 | 0.659 | 0.0922 | 81.98 |

### aishell4_long

| 并发 | 吞吐量（req/s） | 平均延迟（s） | RTF 均值 | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 0.022 | 45.2 | 0.0197 | 50.64 |
| 2 | 0.032 | 60.7 | 0.0265 | 74.25 |
| 4 | 0.036 | 105.6 | 0.0461 | 81.64 |
| 8 | 0.040 | 172.6 | 0.0754 | 90.62 |
| 16 | 0.043 | 282.8 | 0.1237 | 98.83 |


- **并发** — 客户端在途请求的最大数量（`--concurrency`）。
- **吞吐量（req/s）** — 完成的请求数除以基准测试总墙钟时间。
- **平均延迟** — 每个请求端到端的平均时间（从发送到收到完整响应）。
- **RTF 均值** — 每个请求处理时间与输入音频时长的平均比值。`<1` 表示快于实时。
- **audio_s/s** — 处理的输入音频总秒数除以基准测试总墙钟时间。

要复现结果，请按上述命令或
[`benchmark_asr_transcribe_diarize.py`](https://github.com/sgl-project/sglang-omni/blob/main/benchmarks/eval/benchmark_asr_transcribe_diarize.py)
的入口操作。

## 致谢

感谢 OpenMOSS 团队与 SGLang Omni 团队的共同努力。

MOSS 团队：Donghua Yu, Zhengyuan Lin, Hanfu Chen, Yiyang Zhang, Yang Gao, Zhaoye Fei, Qinyuan Cheng, Shimin Li, Xipeng Qiu

SGLang Omni 团队：Yijiang Tian, Xinli Jing, Xiangrui Ke, Zhihao Guo, Ruoqi Zhang, Lifan Shen, Jintao Qu, Xuxiang Tian, Kaige Li, Ratish P, Haoguang Cai, Zijie Xia, Chenchen Hong, Xuesong Ye, Jingwen Gu,  Jiaxin Deng, Jiaxuan Luo, Xinyu Lu, Hao Jin, Chenyang Zhao, Yichi Zhang
