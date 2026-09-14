# Fun-CosyVoice3

[Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) 是阿里巴巴 FunAudioLLM 团队推出的轻量级文本转语音模型（0.5B 参数）。它采用 Qwen2.5-0.5B 骨干网络与 FSQ 语音 token（词表 = 6561 + 200 个特殊 token），以 CAMPPlus 说话人嵌入和通过 ONNX 语音 tokenizer 提取的提示语音 token 为条件。它支持零样本语音克隆、跨语言合成以及基于指令的风格控制。该模型通过 `preprocessing → tts_engine → vocoder` 流水线和兼容 OpenAI 的 `/v1/audio/speech` 端点，以 25 Hz 的 token 帧率生成 24 kHz 语音。

## 前置条件

按照[安装](../get_started/installation.md)说明从源码安装 `sglang-omni`。

Fun-CosyVoice3 需要 `sox` 和一些额外的 Python 包。在仓库根目录下，针对**当前检出的代码**安装该附加组件：

```bash
apt-get update && apt-get install -y sox
uv pip install -e ".[fun-cosyvoice3]"
```

克隆 CosyVoice 仓库及其 Matcha-TTS 子模块，并将两者加入 `PYTHONPATH`：

```bash
COSYVOICE_PATH=/path/to/CosyVoice
COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
MATCHA_TTS_COMMIT=dd9105b34bf2be2230f4aa1e4769fb586a3c824e

git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git ${COSYVOICE_PATH}
git -C ${COSYVOICE_PATH} checkout ${COSYVOICE_COMMIT}
git -C ${COSYVOICE_PATH} submodule update --init --recursive
git -C ${COSYVOICE_PATH}/third_party/Matcha-TTS checkout ${MATCHA_TTS_COMMIT}
export PYTHONPATH="${COSYVOICE_PATH}:${COSYVOICE_PATH}/third_party/Matcha-TTS:$PYTHONPATH"
```

**不要**在 CosyVoice 检出目录中运行 `pip install -r requirements.txt`。该文件锁定的 `torch`、`torchaudio`、`transformers` 和 `diffusers` 版本与 `sglang-omni` 核心锁定的版本冲突。只需要上面的 `fun-cosyvoice3` 附加组件和两条 `PYTHONPATH` 条目；CosyVoice 的 Flow 和 HiFT 模块在 `sglang-omni` 版本的这些共享包下可以正常导入。

checkpoint 中包含语音 tokenizer 和说话人编码器的 ONNX 模型，它们使用 `sglang-omni` 核心依赖中已锁定的 `onnxruntime`。

下载 checkpoint：

```bash
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

流水线为 `preprocessing → tts_engine → vocoder`。首次启动可能需要几分钟，因为 `tts_engine` 需要捕获 CUDA graph。

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000
```

## Apple Silicon

在 Apple Silicon 上，使用仓库安装脚本安装可选的 Fun-CosyVoice3 附加组件，然后将 Homebrew 的 keg-only FFmpeg 库暴露给 TorchCodec：

```bash
brew install sox
SGLANG_OMNI_EXTRAS=fun-cosyvoice3 ./install.sh
source .venv-apple/bin/activate
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

保持使用官方 checkpoint 作为 `--model-path`；它提供 ONNX 预处理资源。MLX 路径额外需要转换后的语音模型工件，其中包含 Qwen2、Flow 和 HiFT 权重。`mlx-audio` 不是运行时依赖。

### MLX

```bash
SGLANG_USE_MLX=1 sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --tts-engine.factory.mlx_model_path \
    mlx-community/Fun-CosyVoice3-0.5B-2512-4bit \
  --tts-engine.engine.quantization mlx_q4 \
  --port 8000
```

### Torch/MPS

不设置 `SGLANG_USE_MLX=1` 时，同一模型通过 PyTorch MPS 运行，且不需要转换后的 MLX 工件：

```bash
unset SGLANG_USE_MLX
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000
```


## 合成语音

### 零样本语音克隆

CosyVoice3 从一段简短的参考音频剪辑克隆声音。`ref_audio` 可以是本地路径、文件 URL、data URL 或 HTTP URL。`ref_text`（参考剪辑的文字转写）是可选的，但建议提供以获得更好的对齐效果。

1. 使用 CURL：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "SGLang-Omni makes text-to-speech fast and easy to deploy.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output.wav
```

2. 使用 Python：

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his.",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### 跨语言合成

CosyVoice3 支持跨语言语音克隆，即参考说话人所说的语言与合成文本的语言不同。省略 `ref_text` 即可进入跨语言模式。

1. 使用 CURL：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "今天天气真好，我们一起出去散步吧。",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
  }' \
  --output output.wav
```

2. 使用 Python：

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "input": "今天天气真好，我们一起出去散步吧。",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### 基于指令的风格控制

传入 `instructions` 以引导韵律、情感或说话风格：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Welcome to our annual developer conference.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "instructions": "Speak in a cheerful and energetic tone, as if addressing a large audience."
  }' \
  --output output.wav
```

### 语速控制

使用 `speed`（默认为 `1.0`）调整播放速度：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "This is spoken at one point three times normal speed.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "speed": 1.3
  }' \
  --output output.wav
```

### 流式

增量式 Flow + HiFT 解码支持流式。设置 `stream: true` 和 `response_format: "pcm"` 即可在 AR 生成完成之前输出音频。声码器（vocoder）使用 CosyVoice3 的因果分块循环：每跳 25 个语音 token（约 1 秒音频）。当 AR 产生 `28 + prompt_pad` 个 token 后即发出第一个 PCM 块（其中 `prompt_pad` 将 Flow 提示长度取整为 25 个 token 的倍数）。后续的跳在声码器中从 25 增长到 50 再到 100 个 token，以提高批处理效率，且不影响音频连续性或实时率——只有解码粒度发生变化，总合成时间不变。调度器每个步骤每个请求最多处理一跳，以防止积压的流独占 GPU。非流式请求则一次性解码整个话语。

可选的服务调节参数（声码器工厂）：

| 工厂参数 | 默认值 | 说明 |
|---|---|---|
| `token_hop_len` | `25` | 基础跳大小；必须与训练分块大小一致 |
| `token_max_hop_len` | `100` | `25 → 50 → 100` 增长的上限 |
| `disable_hop_growth` | `false` | 保持关闭；高并发下开启增长性能更好 |

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

## 生成参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | 服务的模型 | 所服务模型的标识符 |
| `input` | （必填） | 要合成的文本 |
| `ref_audio` | `null` | 用于语音克隆的参考音频（路径 / URL / data URL） |
| `ref_text` | `null` | 参考音频的文字转写。可提升克隆质量；跨语言模式下请省略 |
| `instructions` | `null` | 用于风格/韵律/情感引导的指令文本 |
| `speed` | `1.0` | 播放速度倍率 |
| `temperature` | `0.7` | 采样温度 |
| `top_p` | `0.8` | Top-p 采样 |
| `top_k` | `20` | Top-k 采样 |
| `repetition_penalty` | `1.1` | 重复惩罚 |
| `max_new_tokens` | `min(2048, 20x target text tokens)` | 生成的语音 token 最大数量。如果省略，则根据目标文本长度推导（上限 2048）；在生成长度至少达到该长度的 `2x` 之前，停止 token 也会被抑制 |
| `seed` | `null` | 用于可复现性的随机种子 |
| `stream` | `false` | 增量因果 Flow + HiFT；在 `28 + prompt_pad` 个语音 token 后输出第一个 PCM 块（`prompt_pad` 将提示长度取整为 25 的倍数） |

## 服务优化

### Flow 解码器批处理

对于完整的缓冲请求，调度器准入使用精确的 mel 帧数（`flow_batch_admission_frames`，默认 `8000`）。自适应 Flow 分组默认开启：它按总 mel 长度对已准入的请求排序，并在组内最大长度差距和全局新增填充预算保持在以下范围内时，让相邻请求共享一次 Flow 求解：

```text
flow_merge_max_gap_frames = 384
flow_merge_pad_budget_percent = 25
```

HiFT 分组相互独立，并对生成的 mel 应用其现有的 `hift_max_padding_waste` 策略。因果流式使用单独的 Flow + HiFT 路径。

仅在测量目标 GPU 之后再增加常规 Flow 批处理预算。

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 \
  --vocoder.factory.flow_batch_admission_frames 4000
```

另一方面，可以降低准入预算以减少延迟并降低 GPU 峰值显存。

### 声码器配置

声码器配置控制批处理、精度和加速。调度器接受 `max_batch_size`（16）和 `max_batch_wait_ms`（30）来调节批次组装。Flow 使用 `dtype`（bfloat16）进行 autocast，而 HiFT 使用 `hift_dtype`（float32），与 Flow 相互独立；bfloat16 在 H200 上没有加速效果且会降低保真度。缓冲 Flow CUDA Graph 默认开启。`enable_dit_torch_compile` 和 `enable_flow_estimator_trt` 保持按需开启且互斥。

TTS 引擎阶段接受 `onnx_intra_op_threads`（16），用于语音 tokenizer 和说话人编码器的 ONNX 会话。预处理阶段接受 `max_concurrency`（8），用于限制并发的参考条件化请求数。

### DiT 骨干网络的 torch.compile

`torch.compile` 默认关闭。当你希望 DiT 的 kernel 启动开销最低时启用它。首次启动（Inductor 缓存为空时）大约需要 100 秒，并为每种话语长度构建一个符号化（`dynamic=True`）图；后续启动会复用该缓存。请保留缓存，以免再次支付编译成本（`~/.cache/torch/inductor`，或 `TORCHINDUCTOR_CACHE_DIR`）。

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --vocoder.factory.enable_dit_torch_compile true \
  --port 8000
```

不要将其与 TensorRT 同时启用。

### DiT 骨干网络的 TensorRT

TensorRT 通过从随附的 ONNX 构建缓存的 `.plan` 引擎来加速 DiT。CFG batch 被固定为 2 且 mel 维度动态；更大的请求批次通过切分 cond/uncond 对来处理。TensorRT 与 torch.compile 互斥。

```bash
uv pip install tensorrt
```
更多细节请参阅 NVIDIA 的 [pip 安装指南](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html)。

启用该标志：

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --vocoder.factory.enable_flow_estimator_trt true \
  --port 8000
```

在单张 H200 上的 Flow 延迟和声码器 RTF 对比：

| 后端 | 批大小 | Flow 延迟 | 声码器 RTF |
|---|---|---|---|
| eager | 1 | 1131 ms | 0.359 |
| torch.compile | 1 | 1070 ms | 0.340 |
| TensorRT (CFG batch=2 engine) | 1 | 20 ms | 0.012 |
| eager | 4 | 267 ms | 0.026 |
| torch.compile | 4 | 220 ms | 0.022 |
| TensorRT (chunked 4× CFG pairs) | 4 | 77 ms | 0.011 |

在单张 H200 上，针对缓冲式 `/v1/audio/speech` 运行完整 SeedTTS EN 集（1088 个样本）：

| 后端 | 并发 | 延迟均值 | RTF 均值 | 吞吐量 |
|---|---|---|---|---|
| eager | 1 | 1.091 s | 0.241 | 0.916 req/s |
| torch.compile | 1 | 1.016 s | 0.221 | 0.984 req/s |
| TensorRT | 1 | 0.871 s | 0.189 | 1.147 req/s |
| eager | 16 | 5.690 s | 1.300 | 2.800 req/s |
| torch.compile | 16 | 3.151 s | 0.706 | 5.059 req/s |
| TensorRT | 16 | 2.549 s | 0.570 | 6.243 req/s |

该结果仅作演示之用，因为在 TensorRT 评估完成之后我们还有进一步的优化。

### 零样本语音克隆

CosyVoice3 从一段简短的参考音频剪辑克隆声音。`ref_audio` 可以是本地路径、文件 URL、data URL 或 HTTP URL。`ref_text`（参考剪辑的文字转写）是可选的，但建议提供以获得更好的对齐效果。

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "SGLang-Omni makes text-to-speech fast and easy to deploy.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his."
  }' \
  --output output.wav
```

#### Python

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his.",
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### 跨语言合成

CosyVoice3 支持跨语言语音克隆，即参考说话人所说的语言与合成文本的语言不同。省略 `ref_text` 即可进入跨语言模式。

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "今天天气真好，我们一起出去散步吧。",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
  }' \
  --output output.wav
```

### 基于指令的风格控制

传入 `instructions` 以引导韵律、情感或说话风格：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Welcome to our annual developer conference.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "instructions": "Speak in a cheerful and energetic tone, as if addressing a large audience."
  }' \
  --output output.wav
```

### 语速控制

使用 `speed`（默认为 `1.0`）调整播放速度：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "This is spoken at one point three times normal speed.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "speed": 1.3
  }' \
  --output output.wav
```

### 流式

增量式 Flow + HiFT 解码已启用。设置 `stream: true` 并使用 `response_format: "pcm"`，使服务器能够在语音 token 生成结束之前输出音频。

声码器遵循 CosyVoice3 的因果分块循环：每跳 25 个语音 token（`pre_lookahead_len=3`，25 Hz → 每块约 1 秒音频）。当 AR 阶段产生 `28 + prompt_pad` 个 token 后即发出第一个 PCM 块，其中 `prompt_pad`（0–24）将 Flow 提示 token 长度向上取整为 25 的倍数。后续的跳像上游 `CosyVoice3Model` 一样从 25 → 50 → 100 个 token 增长（默认行为；保持增长开启）。调度器的每个步骤对每个请求最多运行一跳，因此积压的流无法独占 GPU。非流式请求仍然对整个话语使用缓冲式 Flow + HiFT 路径。

可选的服务调节参数（声码器工厂；如果更改跳大小，请保持 `tts_engine.factory.token_hop_len` 同步）：

| 工厂参数 | 默认值 | 说明 |
|---|---|---|
| `token_hop_len` | `25` | 基础跳大小 / AR 后续刷新；必须与训练分块大小一致 |
| `token_max_hop_len` | `100` | `25 → 50 → 100` 增长的上限 |
| `disable_hop_growth` | `false` | 除非做 A/B 对比，否则保持关闭；在 c=16 时开启增长效果更好 |

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

## 生成参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | 服务的模型 | 所服务模型的标识符 |
| `input` | （必填） | 要合成的文本 |
| `ref_audio` | `null` | 用于语音克隆的参考音频（路径 / URL / data URL） |
| `ref_text` | `null` | 参考音频的文字转写。可提升克隆质量；跨语言模式下请省略 |
| `instructions` | `null` | 用于风格/韵律/情感引导的指令文本 |
| `speed` | `1.0` | 播放速度倍率 |
| `temperature` | `0.7` | 采样温度 |
| `top_p` | `0.8` | Top-p 采样 |
| `top_k` | `20` | Top-k 采样 |
| `repetition_penalty` | `1.21` | 重复惩罚 |
| `max_new_tokens` | `min(2048, 20x target text tokens)` | 生成的语音 token 最大数量。如果省略，则根据目标文本长度推导（上限 2048）；在生成长度至少达到该长度的 `2x` 之前，停止 token 也会被抑制 |
| `seed` | `null` | 用于可复现性的随机种子 |
| `stream` | `false` | 增量因果 Flow + HiFT；在 `28 + prompt_pad` 个语音 token 后输出第一个 PCM 块（`prompt_pad` 将提示长度取整为 25 的倍数） |

## 基准测试

针对运行中的服务器，使用 Seed-TTS-Eval 测量因果流式性能：

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 --lang en --max-concurrency 16 \
  --use-existing-server --generate-only --stream \
  --output-dir results/fun_cosyvoice3_en
```

中文跨语言切分请使用 `--lang zh --no-ref-text`。完整工作流请参阅 `benchmarks/README.md`。

## 模型架构

| 组件 | 详情 |
|---|---|
| LLM 骨干网络 | Qwen2.5-0.5B（24 层，hidden=896，14 个注意力头，2 个 KV 头 GQA） |
| 语音 Tokenizer | FSQ codebook（vocab=6561）+ 200 个特殊 token，25 Hz 帧率 |
| 说话人编码器 | CAMPPlus（192 维嵌入，ONNX） |
| Flow 模型 | CausalMaskedDiffWithDiT（DiT depth=22，dim=1024，heads=16） |
| 声码器 | CausalHiFTGenerator（24 kHz 输出） |
| 采样率 | 24000 Hz |

## 已知限制

- **需要参考音频。** CosyVoice3 语音克隆需要一段参考音频剪辑；它不支持没有说话人参考的纯文本合成。
- **30 秒限制。** 用于语音 token 提取的参考音频不得超过 30 秒。
- **说话人相似度。** 提供 `ref_text`（文字转写）比省略它（跨语言模式）能获得更好的声音相似度。
- **参考格式。** 该端点要么接受 `ref_audio` 加可选的 `ref_text`，要么接受 `references` 中的一项；此 checkpoint 会拒绝多个参考。
- **提示模式。** 为参考提示提供 `ref_text` 或 `instructions` 之一，而不是两者都提供。`instructions` 选择 CosyVoice3 的 `instruct2` 条件化。
- **参考条件化缓存。** 本地文件、data URL 和字节载荷按音频内容与编码器配置进行缓存。可变的 HTTP URL 会在每次请求时特意重新编码，而不是仅按 URL 缓存。
- **语速控制。** 由共享的 `/v1/audio/speech` 响应编码路径在解码后的波形上一次性应用。
- **语音转换。** 语音转换不在当前零样本 TTS 的范围内。
- **流式解码。** 因果 Flow + HiFT 在每跳之后输出 PCM（跳默认按 25 → 50 → 100 增长）。质量可能与缓冲式整句路径略有差异。可选开启的 TensorRT（`enable_flow_estimator_trt`）同样会加速流式跳；不要将其与 `enable_dit_torch_compile` 同时启用。TRT 会冻结 DiT 注意力，因此 streaming+TRT 与 PyTorch 流式不是位级一致的。流式时请保留 Module TRT 包装器：CosyVoice 的原生 TRT enqueue 与打包的跳批次 CFG 形状不兼容。
- **Flow 批处理范围。** Flow 批处理支持 CosyVoice PyTorch 估计器和可选开启的 TensorRT 估计器。缓冲式 HiFT 分组相互独立并使用 `hift_max_padding_waste`。流式会跨请求合并首个/后续跳（`_can_batch_stream_chunks`，短暂的同伴等待），从而在高负载下保持较低的 TTFP。
- **cosyvoice 依赖。** `cosyvoice` 包没有 PyPI 发布，必须从 GitHub 安装。Matcha-TTS 是必需的子模块，也必须可导入；声码器只使用 CosyVoice 的 Flow 和 HiFT 路径。
