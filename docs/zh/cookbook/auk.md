# AuK

[AuK](https://huggingface.co/tencent/AuK) 与 [AuK-Flash](https://huggingface.co/tencent/AuK-Flash) 支持指令驱动的语音生成与编辑。二者共享一个四阶段流水线：预处理 → 条件编码（conditioning）→ DiT 采样 → VAE 解码。AuK-Flash 是经 DMD 蒸馏的四步配方。

发布的 checkpoint 使用：

| 组件 | 配置 |
|---|---|
| Conditioner | 冻结的 Qwen2.5-Omni-3B Thinker，仅文本与音频 |
| DiT | Flux 风格 MMDiT：10 个 double-stream block、20 个 single-stream block、dim=1536、24 个注意力头 |
| VAE | 共享的 reference encoder 与音频解码器；50 Hz、64 通道隐变量 |

| Checkpoint | 采样 |
|---|---|
| [`tencent/AuK`](https://huggingface.co/tencent/AuK) | Euler，NFE=32，CFG=2.0，sway=-1.0 |
| [`tencent/AuK-Flash`](https://huggingface.co/tencent/AuK-Flash) | 发布的四步时间网格，CFG=0。工厂参数 `nfe` / `cfg_strength` / `sway_sampling_coef` 会被忽略 |

## 前置条件

按照[安装指南](../get_started/installation.md)完成安装，然后在仓库根目录运行：

```bash
python -m sglang_omni.cli serve --model-path tencent/AuK --port 8000
```

```bash
python -m sglang_omni.cli serve --model-path tencent/AuK-Flash --port 8000
```

## 语音生成

`/v1/audio/speech` 通过 `input` 接收文本。在没有参考音频时，`instructions` 描述音色，默认值为 `A clear, natural voice.`。该模式下必须显式指定目标时长：

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Welcome home.",
    "instructions": "warm, relaxed female voice",
    "stage_params": {"auk_engine": {"gen_seconds": 3}},
    "seed": 1234,
    "response_format": "wav"
  }' --output speech.wav
```

语音克隆时，请提供 `ref_audio` 或一条结构化参考。模型会使用"同音色"指令并忽略音色描述。支持 HTTP(S) URL、音频 data URL 以及服务器本地路径。要使用 `file://` URL，需以 `--allowed-local-media-path /abs/reference-dir` 启动服务器，并引用该目录内的文件，例如 `file:///abs/reference-dir/reference.wav`。未启用该参数时，`file://` 引用返回 HTTP 400。

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Welcome home.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "seed": 1234,
    "response_format": "wav"
  }' --output speech.wav
```

省略 `gen_seconds` 时，语音克隆要求提供参考转写文本（`ref_text` 或 `references[0].text`）。目标时长按以下方式估算：

```text
target_seconds = reference_seconds × UTF8_bytes(input) / UTF8_bytes(ref_text)
```

显式的 `gen_seconds` 优先级更高且必须为正数。目标时长会向上取整到 20 ms 帧，默认上限为 30 秒。要修改上限，需在两个阶段上设置 `max_seconds`：`--preprocessing.factory.max_seconds` 和 `--auk_engine.factory.max_seconds`。

## 语音编辑

`/generate` 在 `prompt` 中接收原始 AuK 指令并返回 JSON。将 `output_modalities` 设为 `["audio"]`，`return_logprob` 设为 `false`（AuK 不产生 token 对数概率）。参考音频通过 `metadata.tts_params.ref_audio` 传入：

```bash
curl http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Remove the background noise.",
    "metadata": {"tts_params": {"ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"}},
    "output_modalities": ["audio"],
    "return_logprob": false
  }'
```

可通过 `stage_params.auk_engine.gen_seconds` 覆盖时长。否则编辑使用源音频的完整 20 ms 帧，并受时长上限约束。既无参考音频又无显式时长的原始请求默认为 5 秒。

## 采样

基础版 AuK 使用 Euler 积分，工厂默认值为 `nfe=32`、`cfg_strength=2.0`、`sway_sampling_coef=-1.0`。可通过 `--auk_engine.factory.*` 参数覆盖。Flash 版锁定为发布版的四步网格且禁用 CFG，因此这些参数对 `tencent/AuK-Flash` 无效。对这些设置及 `max_seconds` 的请求级覆盖会被拒绝。Qwen 使用 BF16 autocast；VAE 以 FP32 运行。

DiT 以 BF16 存储权重，默认不使用 autocast 运行（`--auk_engine.factory.weight_dtype bfloat16`），从而省去每步的 FP32→BF16 权重转换。设置 `--auk_engine.factory.weight_dtype float32` 可改为保留 FP32 权重并使用 BF16 autocast；这是对齐性测试所比对的上游精确配方，采样耗时约为默认模式的 1.3 倍。两种模式下 ODE 状态均以 FP32 积分。

`seed` 为目标噪声与参考 VAE 后验采样初始化相互独立的请求本地生成器，不改变进程 RNG。固定输入下采样可复现；不同的 batch 形状或计算后端仍可能产生数值差异。不支持多条结构化参考。

条件编码与 DiT 采样使用动态批处理，默认最大 batch size 分别为 8 和 16。VAE 解码将等长隐变量分组（最多 4 个请求）以保持边界行为。各阶段可在不同 CUDA stream 上重叠执行，并在同一进程/设备内共享 VAE 权重。条件编码阶段加载 Qwen 编码器、VAE 以及两个隐状态融合参数；只有采样阶段加载 DiT。可设置 `--conditioning.factory.max_batch_size`、`--auk_engine.factory.max_batch_size` 或 `--decode.factory.max_batch_size` 进行调优。音频在解码完成后一次性返回；未实现增量式音频流式输出。

## SeedTTS 评测

标准基准测试会识别 `tencent/AuK` 与 `tencent/AuK-Flash`，并从 `--model-path` 启动服务器。默认使用完整英文数据集、并发 1、1 次预热、seed 1234。它根据参考音频与转写文本估算时长，然后自动启动并停止 TTS 与 ASR 服务器：

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.benchmark_tts_seedtts \
  --model tencent/AuK --output-dir results/auk_en
```

添加 `--concurrency 16` 可以 16 个在途请求进行评测。使用 `--max-samples` 与 `--sample-offset` 可评测子集。`--generate-only` 与 `--transcribe-only` 分别运行单个阶段；为任一模式添加 `--use-existing-server` 可复用已在运行的服务器。显式 CLI 选项会覆盖 AuK 默认值。

`wer_results.json` 包含全样本平均 WER、`wer_below_50_per_sample_mean`（严格剔除 WER 高于 50% 的样本）以及 `n_above_50_pct_wer`。语料级 WER 单独报告，并按词数加权。

## 上游对齐

checkpoint 对齐测试会在有/无参考音频两种情况下，与上游比较参考隐变量、融合后的 Qwen 条件、生成的隐变量与波形。它将上游进程 RNG 对齐到请求 seed，以比较相同的随机输入。除推理依赖外，还需安装 `torchdiffeq`、`qwen-omni-utils` 与 `audioread`，并使用一块显存足以同时容纳两套实现的 GPU：

```bash
pip install torchdiffeq qwen-omni-utils audioread
git clone https://github.com/Tencent-Hunyuan/AuK.git /tmp/AuK
git -C /tmp/AuK checkout d9f30ffe4231dbc90b48cc83a35d310fece0b060
AUK_UPSTREAM_SOURCE=/tmp/AuK \
AUK_PARITY_CHECKPOINT=tencent/AuK \
python -m pytest tests/test_model/test_auk_parity.py -v
```

```bash
AUK_UPSTREAM_SOURCE=/tmp/AuK \
AUK_PARITY_CHECKPOINT=tencent/AuK-Flash \
python -m pytest tests/test_model/test_auk_parity.py -v
```

`AUK_QWEN_CHECKPOINT` 可选择本地编码器。只有同时设置 `AUK_UPSTREAM_SOURCE` 与 `AUK_PARITY_CHECKPOINT` 时测试才会运行，否则跳过。

## 致谢

本实现派生自上述修订版本的 [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK)。其 MIT 声明保留在 `sglang_omni/models/auk/LICENSE` 中。VAE 源码同样保留了 NVIDIA 与 alias-free-torch 的致谢。
