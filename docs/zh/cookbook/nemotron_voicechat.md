# NemotronLabs VoiceChat

[NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) 是一个 11B 端到端全双工语音到语音模型。它以 12.5 Hz 帧锁定：每 80 ms 的呼叫方音频会推进一个 Fast Conformer 感知编码器、一个每帧输出一个文本 token 和一个功能 token 的 Nemotron-H thinker、一个将每个文本 token 转换为 31 个 RVQ 码的 EAR-TTS talker，以及一个将这些码渲染为 22.05 kHz 音频的 RVQ-VAE 编解码器（codec）。模型自行决定何时说话；没有外部 VAD。

SGLang-Omni 目前将其作为**离线**流水线提供服务：输入一段录音，输出智能体的回复文本和音频。通过 `/v1/realtime` 进行的实时双工会话在 [#1909](https://github.com/sgl-project/sglang-omni/issues/1909) 中跟踪，不属于本页内容。

## 前置条件

该 checkpoint 是单个 44 GB 的 fp32 `model.safetensors`，附带 NeMo 风格的 `config.json`。使用以下命令下载：

```bash
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B
```

该 checkpoint 既不附带 HF tokenizer，也不附带主干的 `config.json`；二者都来自其配置中指定的兄弟仓库（`model.stt.model.pretrained_llm`，当前为 `nvidia/NVIDIA-Nemotron-Nano-9B-v2`），约 17 MB 的文件且不含权重。在离线环境中，请同时下载这些文件：

```bash
hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2 config.json tokenizer.json tokenizer_config.json special_tokens_map.json
```

如果这些文件缺失且 Hub 无法访问，启动会失败并给出指明仓库名和此命令的错误。

四个阶段共享一块 GPU。在单块 H200（143.8 GB）上使用离线示例的实测中，运行峰值约为 129 GB：talker 引擎预留 `mem_fraction_static=0.35`（约 47 GB，其中大部分为 KV 池），thinker 使用剩余部分（约 73 GB，其中 17.7 GB 为 bf16 权重）。如果 GPU 较小或被共享，请降低 `--talker.engine.mem_fraction_static`；感知编码器和编解码器（codec）各只需几 GB。

## 运行离线示例

```bash
python examples/run_nemotron_voicechat.py \
  --model-path /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
  --audio /path/to/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav \
  --out reply.wav
```

该示例构建 `NemotronVoiceChatPipelineConfig`，使用 `MultiProcessPipelineRunner` 启动四个阶段，发送一个请求并将回复以 22.05 kHz 的 16 位 PCM 写出。使用 `CUDA_VISIBLE_DEVICES` 选择 GPU。一旦 checkpoint 进入页缓存，启动约需一分钟；41 秒的 `turn_taking.wav` 样本以接近实时的速度渲染。

两个输入约定很重要：

- 录音会被重采样到 16 kHz，并且**使用声道 0**（双方通话录音中智能体位于声道 1）；它会被零填充到 1280 个采样点（80 ms）的整数倍。
- 在呼叫方最后一句话之后留出尾部静音。模型只会在听到静音时应答，因此在问题结束后立即截断的片段只会得到较短或空的回复。

回复文本只是 thinker 的口语 token；模型正在聆听的帧携带一个标记 token，并从文本中丢弃。

## 请求参数

| 参数 | 效果 |
|---|---|
| `temperature`、`top_p`、`top_k` | **被忽略。** thinker 始终进行贪心解码，以确保它流式传给 talker 的 token 就是它确认提交的 token；非零的 `temperature` 会记录一条警告。 |
| `max_new_tokens` | 被忽略；帧数由输入长度固定（每 80 ms 一个 token）。 |

音频随机性由 checkpoint 自身的设置控制，从 `config.json` 读取（`inference_noise_scale`、`inference_top_p_or_k`）：talker 使用 Gumbel-max 分量选择和 MoG 头内部的高斯噪声对其码进行采样，因此即使文本相同，同一输入的两次运行也会产生不同的码序列和略有差异的音频。目前没有 `seed` 控制。

## 已知限制

- 离线，一次仅一个请求（两个引擎均为 `max_running_requests=1`）。没有打断（barge-in）或中断 API。
- 单一说话人（`Aria`，即 checkpoint 内置的提示 latent）。
- `function_head` 工具调用通道会被解码并一路传递，但没有组件对其执行操作。
- 分类器无关引导（classifier-free guidance）处于关闭状态（`guidance_scale=0`），尽管 checkpoint 配置以 0.2 启用它。
- 即使双方都使用确定性采样，talker 输出与 NeMo 的离线脚本也不逐位一致（在一个 12 秒的测试样本上，约 82% 的量化器单元和 150 帧中的 63 帧一致；转录文本和幅度包络匹配）。已知原因：主干和 KV 缓存以 bfloat16 运行，因为 SGLang 的注意力后端和 `sgl_kernel` 的 Gemma RMSNorm 没有 fp32 路径，而 NeMo 以 fp32 运行整个 talker；NeMo 的离线脚本默认启用分类器无关引导；并且 NeMo 的离线配方会在第 0 帧之前走过系统提示区域（此处为实时约定；参见 `prompt_region_steps`）。thinker 与 NeMo 逐帧一致。
- NeMo 将感知编码器的冲刷行解码为第 151 帧；本流水线会计算它但不解码它，因此回复短一帧（80 ms）。

## 测试

`tests/unit_test/nemotron_voicechat/` 在 CPU 上运行且无需权重：请求帧数、流式编解码器（codec）与整句解码的等价性，以及 checkpoint shim 隔离。
