# Whisper ASR

Whisper ASR checkpoint 可以通过 OpenAI 兼容的 `/v1/audio/transcriptions` 端点启动。该路径在当前 SGLang-Omni 代码树中仍属实验性；生产部署前请验证 checkpoint 专属的准确率与运行行为。

## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`，然后下载 Whisper checkpoint：

```bash
hf download openai/whisper-large-v3
```

## 服务器配置

Whisper ASR 在单块 GPU 上运行一个 ASR 阶段。

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

## 编码器 CUDA Graph

编码器 CUDA Graph 默认启用。在 pre-LM 编码（默认）下，捕获桶跟随
`pre_lm_max_batch_size`（8），因此捕获 **1/2/4/8** 批次。`request_build_max_workers`
默认为 8，与 Qwen3-ASR 及 Fun-ASR 一致。当 `enable_pre_lm_encoder` 为 false
时，桶跟随原子 prefill 预算（`6144 // 1500 = 4`）。要使用 eager 编码器执行，
请覆盖流水线配置：

```yaml
config_cls: WhisperASRPipelineConfig
name: whisper
model_path: openai/whisper-large-v3-turbo

stages:
  asr:
    factory:
      enable_encoder_cuda_graph: false
```

该 graph 在 SGLang 的生成 graph 之后捕获。关闭 pre-LM 时，配置更大的 LM 侧
桶（12/16）之前请先调高 `max_prefill_tokens`。每个请求使用能容纳其 batch 的
最小已捕获桶。超过所有已捕获桶、特征形状不同或没有成功捕获的请求走 eager。
启动与首次 replay 日志会标明已捕获与实际执行的桶。

## 可回退的 prefill CUDA Graph

解码器主体默认使用 SGLang 的可回退（breakable）prefill CUDA Graph 后端。
Whisper 编码器状态与交叉注意力 K/V 在被捕获的解码器主体之外准备。默认捕获
阶梯停在原子准入所能组成的最大聚合解码器 token 数。该上限考虑了所有可能的
请求数，因为更多、更短提示词的 batch 可能比按最长请求大小组成的 batch 含有
更多解码器 token。在默认 6,144 预算、1,500 个编码器占位、每请求至多 232 个
解码器 token 下，上限为 696。启动日志报告捕获开销，并用
`prefill CUDA graphs attested` 确认生效的桶。

当前的基准结果使用 `cuda_graph_max_bs_prefill=256` 采集。要复现该捕获 profile
并限制启动时间与 GPU 显存，请显式设置 prefill graph 上限：

```yaml
stages:
  asr:
    engine:
      cuda_graph_max_bs_prefill: 256
```

Whisper 运行三个独立配置的 graph 平面，各自按自己的轴分桶；交叉注意力 K/V
在前两个平面之间每请求只写一次，之后只读：

| 平面 | 分桶轴 | 配置 |
|---|---|---|
| 编码器前向 | batch 大小 | `enable_encoder_cuda_graph`、编码器 graph 桶 |
| 解码器 prefill 主体 | 聚合 prefill token 数 | `cuda_graph_backend_prefill`、`cuda_graph_bs_prefill` |
| 解码器 decode | batch 大小 | `cuda_graph_bs`、`cuda_graph_max_bs` |

禁用一个平面不影响另外两个。

要只禁用 prefill graph 而保持 decode 与编码器 CUDA Graph 启用，覆盖 ASR 阶段：

```yaml
config_cls: WhisperASRPipelineConfig
name: whisper
model_path: openai/whisper-large-v3

stages:
  asr:
    engine:
      cuda_graph_backend_prefill: disabled
```

## Prefill 合并

Whisper 默认用 8 个 worker 线程构建请求，与其他 pre-LM ASR 流水线一致。合并
闸门以 2 个请求为目标，而默认的 6,144 token 原子预算允许 LM 调度器一次准入至多
4 个 1,504 token 的 Whisper 请求。不满的 batch 只在还有其他请求构建未完成时至多
等待 6 ms；单个请求以及没有剩余构建工作的不满 batch 会立即释放。

`request_build_max_pending` 约束的是已提交的请求构建 future，而不是请求积压。
当 `max_queued_requests` 未设置时，超过该构建 pending 上限的请求留在队列中等待
稍后构建。设置 `max_queued_requests` 可保留配置的有限队列拒绝行为。

用 `prefill_coalesce_requests` 与 `prefill_coalesce_wait_ms` 调整闸门。设置
`prefill_coalesce_requests: 0` 只禁用合并；再设置 `request_build_max_workers: 1`
可恢复优化前的请求构建路径：

```yaml
stages:
  asr:
    factory:
      request_build_max_workers: 1
      prefill_coalesce_requests: 0
```

## 异步解码

Whisper 在 batch 大小 ≥ 2 时启用共享的一步前瞻（one-step-lookahead）解码路径。
它把当前解码步的 GPU 工作与上一步的宿主侧结果处理重叠，而 batch 大小为 1 时
保持同步路径。默认运行请求上限为 64。要对比同步解码或排查请求生命周期问题，
可在阶段上禁用异步解码：

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --asr.factory.enable_async_decode false \
  --port 8000
```

## 转写音频

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "openai/whisper-large-v3",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## 翻译音频

Whisper 多语言 checkpoint 可以通过 `/v1/audio/translations` 把源语音翻译成
英语。请使用多语言、非 turbo 的 checkpoint：`*.en` checkpoint 没有翻译任务，
`whisper-large-v3-turbo` 蒸馏时也未包含它。

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F language=fr \
  -F response_format=json
```

对该端点，`language` 是可选的源语言提示，且是一个 **SGLang-Omni 扩展**。
OpenAI 官方的音频翻译请求 schema 不含 `language`；两种 API 的翻译目标都是
英语。响应格式及其他 ASR 模型见
[音频翻译支持矩阵](../basic_usage/audio_translations.md)。

## 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `file` | file | 必填 | 以 multipart 表单数据上传的音频文件 |
| `model` | string | 服务器默认值 | 模型标识符 |
| `language` | string | 未设置 | 可选的源语言提示；在 translations 上这是 SGLang-Omni 扩展 |
| `prompt` | string | 未设置 | 用作 Whisper 前文条件（prev-context）的可选文本 |
| `response_format` | string | `json` | `json`、`verbose_json`、原始 `text`、`srt` 或 `vtt` |
| `temperature` | float | `0.0` | 采样温度；默认为贪心解码 |

服务路由根据端点（`transcribe` 或 `translate`）选择内部 `task`；它不是公开的
表单字段。除非流水线另有配置，路由使用 ASR 阶段默认值。做冒烟测试时，请保持
请求最简并使用 `response_format=json`。

对转写与翻译，`srt` 和 `vtt` 请求 Whisper 模型推导的分段时间戳。支持非流式
字幕请求；带任一字幕格式的 `stream=true` 返回 HTTP 400。

## 长音频

Whisper 单个请求至多读取 30 秒音频：特征提取器在固定的 30 秒 mel 窗口上工作，
丢弃超出部分。在 SGLang-Omni 中，更长的上传会分块转写：我们在每个 30 秒边界
附近最安静的位置切分音频，把每块作为独立的引擎请求运行，再按顺序拼回转写。
默认情况下各块独立解码，调用方的 `prompt` 会发送到每一块。把
`condition_on_previous_text` 设为 `true` 可让各块顺序解码：调用方 prompt 条件化
第一块，之后每块使用紧邻前一块的解码文本作为其 prompt。如果某个启用的块出现
持续的重复字符循环，会不带前文重试一次；第一次结果被丢弃，重试结果条件化下一
块。检测器检查由 8–128 个归一化的字母、数字或组合字符组成、重复至少三次的
周期。它不依赖以空白分词的单词，因此也覆盖没有可靠词边界的语言。单个拉丁文字
单词重复三次的情况被排除，以保留字面重复。三份阈值捕捉到了观察到的病态循环，
且在提交的 TED-LIUM 评测上零重试；它仍是一个保守的恢复启发式，而非转写保证。

该可选行为与 OpenAI Whisper 的 `condition_on_previous_text=True` 不完全相同：
OpenAI Whisper 在内部窗口之间携带解码 token 历史，并可能在回退期间重置该历史，
而 SGLang-Omni 目前传递的是上一个服务器块的解码文本。当前的 TED-LIUM 评测
没有显示启用该实现带来准确率或吞吐优势，因此它保持默认关闭。更改默认值或实现
token 级一致性应在单独的 PR 中评估。

行为由两类取值决定。

调度策略可由你用点分参数或对应的 YAML 键调优：

| 名称 | 默认值 | 含义 |
|---|---|---|
| `--audio_chunking.max_audio_clip_s` | `30` | 单个请求发送给引擎的最长音频片段，即分块长度。与 Qwen3-ASR 不同，你只能调低它：30 秒是模型 mel 窗口的硬边界。 |
| `--audio_chunking.max_concurrent_chunks` | `8` | 各块独立时的按请求并发上限。启用前文条件化后，一个请求的 Whisper 块按顺序解码，而不同请求的块仍可合并成批。 |
| `--audio_chunking.max_total_audio_s` | `3600` | 整个上传的上限；超过会得到 HTTP 400。这是内存防护：块运行期间我们会把解码出的波形保存在内存中。 |

模型属性是 `WhisperASRPipelineConfig` 上的 ClassVar；没有任何配置路径可以触达
它们：

| 名称 | 取值 | 含义 |
|---|---|---|
| `allow_audio_chunking` | `true` | Whisper 能正确转写孤立的分块，因此分块开启。 |
| `max_native_clip_s` | `30` | mel 窗口边界。流式无法分块，因此 `stream=true` 至多接受 30 秒音频，超过返回 HTTP 400。 |
| `min_tail_s` | `1` | 值得转写的最短末尾块；若尾块更短，我们会把上一个切点提前以吸收它，避免 Whisper 在极短片段上产生幻觉。 |
| `condition_on_previous_text` | `false` | 是否让 Whisper 串行化各块并让每块以前一块的解码文本为条件。禁用时各块保持独立，可使用按请求的并发上限。 |

## 基准测试

使用共享的 SeedTTS 基准测端到端并发、WER、延迟与吞吐：

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-base \
  --max-samples 128 --concurrencies 1,2,4,8,16,32 \
  --repeats 5 --warmup --output whisper_concurrency.json
```

要复现下文的异步解码对比，解析锁定的 checkpoint 并在同一块 GPU 上分别启动
两种模式：

```bash
MODEL_REVISION=06f233fe06e710322aca913c1bc4249a0d71fce1
MODEL_PATH="$(
  hf download openai/whisper-large-v3 \
    --revision "$MODEL_REVISION" \
    --quiet
)"

CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --mem-fraction-static 0.30 \
  --port 8000

# Replace the command above with this one for the synchronous baseline.
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --mem-fraction-static 0.30 \
  --asr.factory.enable_async_decode false \
  --port 8000
```

每种模式运行一次相同的客户端命令，只改输出文件名：

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --model-path openai/whisper-large-v3 \
  --model-revision 06f233fe06e710322aca913c1bc4249a0d71fce1 \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --max-samples 128 \
  --concurrencies 1,2,4,8,16,32,64 \
  --repeats 3 \
  --warmup \
  --dtype float16 \
  --cuda-graph \
  --torch-compile \
  --max-running-requests 64 \
  --mem-fraction-static 0.30 \
  --fingerprint \
  --output whisper_async.json
```

## 基准测试结果

以下 W-PR1 结果使用 20 样本的 SeedTTS EN 子集，在单张 H200 上以 FP16 运行
`openai/whisper-base`。每种模式在每个并发下丢弃一次预热并做三次测量。

| 并发 | Eager req/s | CUDA Graph req/s | 吞吐增益 | Eager 平均延迟（s） | CUDA Graph 平均延迟（s） | 语料级 WER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.57 | 20.29 | 3.7% | 0.051 | 0.049 | 0.0415 |
| 2 | 28.41 | 30.87 | 8.7% | 0.070 | 0.065 | 0.0415 |
| 4 | 37.90 | 41.70 | 10.0% | 0.104 | 0.094 | 0.0415 |
| 8 | 42.10 | 49.00 | 16.4% | 0.185 | 0.158 | 0.0415 |

W-PR1 的全部 480 个测量请求均成功完成。语料级 WER 在每个并发下 eager 与
CUDA Graph 模式完全一致。

以下 W-PR2 结果在同一张 H200 与 20 样本子集上单独测量，每个并发五次测量加一次
丢弃的预热。基线使用 1 个请求构建 worker 且禁用合并；归因运行使用 2 个 worker
且禁用合并；优化运行使用 2 个 worker、批目标 2 以及感知构建 pending 的 6 ms
截止时间。

| 并发 | 基线 req/s | 两 worker req/s | 合并后 req/s | 总增益 | 闸门增益 | 基线延迟（s） | 合并后延迟（s） | 语料级 WER |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.04 | 22.51 | 22.46 | 6.8% | -0.3% | 0.047 | 0.044 | 0.0415 |
| 2 | 30.45 | 36.68 | 41.96 | 37.8% | 14.4% | 0.066 | 0.047 | 0.0415 |
| 4 | 40.24 | 55.62 | 62.83 | 56.2% | 13.0% | 0.097 | 0.063 | 0.0415 |
| 8 | 48.03 | 75.93 | 82.15 | 71.0% | 8.2% | 0.161 | 0.092 | 0.0415 |

全部 1,200 个测量请求均成功完成。语料级 WER 在三种模式与每个并发下都保持
0.0415。优化运行的日志显示 `Replaying Whisper encoder CUDA graph batch=2 request_batch=2`，以及含两条序列、3,008 个新 token 的 prefill 批次。

异步解码对比在同一张 H200 上使用 128 样本的 SeedTTS EN 子集、FP16 的
`openai/whisper-large-v3`，每个并发丢弃一次预热并做三次测量。基线禁用异步解码；
其余服务设置（包括 6,144 token 的 prefill 预算）完全相同。

| 并发 | 同步 req/s | 异步 req/s | 吞吐变化 | 同步 P95（s） | 异步 P95（s） | P95 变化 | 语料级 WER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.26 | 11.44 | +1.6% | 0.117 | 0.115 | -1.7% | 0.0084 |
| 2 | 18.45 | 19.53 | +5.8% | 0.140 | 0.133 | -5.4% | 0.0084 |
| 4 | 27.40 | 29.35 | +7.1% | 0.197 | 0.185 | -6.2% | 0.0084 |
| 8 | 38.77 | 40.88 | +5.4% | 0.285 | 0.268 | -6.2% | 0.0084 |
| 16 | 55.59 | 57.90 | +4.2% | 0.396 | 0.366 | -7.6% | 0.0084 |
| 32 | 66.47 | 69.91 | +5.2% | 0.691 | 0.639 | -7.6% | 0.0084 |

两种模式共 4,608 个测量请求全部成功完成，全部 2,304 对转写完全一致。batch
大小为 1 使用同步快速路径，因此其 1.6% 差异属于运行间噪声而非异步工作。并发
32 时，请求阶段 profiling 测得从 prefill 完成到请求完成的 P95：同步 614.3 ms，
异步 585.5 ms。另一个仅异步的 `openai/whisper-base` 预算对比说明了 6,144 为何
是默认值：相对 4,096，调度器队列 P95 从 92.2 ms 降至 52.2 ms，吞吐从 134.83
升至 166.69 req/s。

## 已知限制

- Whisper ASR 仍属实验性。生产部署前请验证 checkpoint 专属的准确率与运行行为。
- `verbose_json` 返回横跨音频时长的单个分段；`srt` 与 `vtt` 不支持，返回
  HTTP 400。
- 编码器 CUDA Graph 默认启用，且依赖 SGLang 生成 CUDA Graph。生产使用前请
  验证所选桶。
- 音频编码默认在 LM 准入之前运行（`pre_lm_max_batch_size=8`、
  `request_build_max_workers=8`）。在 `stages.asr.factory` 下设置
  `enable_pre_lm_encoder: false` 可回到 prefill 内运行编码器。
- pre-LM 编码器缓存（`pre_lm_cache_max_entries=1024`）把条目保存在页锁定
  （pinned）宿主内存中，使设备到宿主与宿主到设备的拷贝在 DMA 路径上异步执行，
  而不是通过可分页的中间缓冲阻塞 worker 线程。整个预算（large-v3 为
  `entries × 3.84 MB`，`≈3.9 GB`）在启动时锁定且不可换出；请相应设置容器内存
  上限，或设置 `pre_lm_cache_pin_host_memory: false` 回退到可分页内存。
- 原子准入（`chunked_prefill_size=0`）下，prefill 预算默认为 6,144 token
  （`⌊6144/1500⌋=4`）。这独立于 pre-LM 编码器批上限，约束 LM 侧 prefill 批处理。
- 分块 prefill 保持禁用，因为 Whisper 编码器前缀必须原子准入。超过当前 prefill
  预算的请求等待下一个批次，而不是拆分编码器前缀。
- 首次启动可能需要几分钟。
- 端点每个请求接受一个上传文件。
- 音频在转写前被重采样到 16 kHz。
- `prompt` 通过 Whisper 前文 token 条件化解码。只保留最后 223 个 prompt
  token（含 `<|startofprev|>` 共 224 个前文 token）——`max_new_tokens` 较大时
  更少，因为 prompt、任务前缀与输出共享 Whisper 的 448 token 解码器上下文。
  `max_new_tokens` 同样被钳制到该上下文。prompt 不得包含 Whisper 特殊 token。
