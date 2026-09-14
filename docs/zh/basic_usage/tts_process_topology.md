# TTS 进程拓扑

`StageConfig.process` 是进程拓扑的唯一事实来源。它只是普通的按阶段配置：模型的配置类声明默认值，配置文件或点分形式的 CLI 标志可以像覆盖任何其他阶段字段一样覆盖它。省略该字段则保留所选模型或 YAML 配置声明的拓扑。

配置文件可以让声码器（vocoder）隔离持久化：

```yaml
stages:
  vocoder:
    process: vocoder
```

或者让声码器留在共享进程中：

```yaml
stages:
  vocoder:
    process: pipeline
```

MOSS-TTS delay 默认采用隔离布局，为 `preprocessing`、`tts_engine` 和 `vocoder` 声明的 GPU 内存占比是 0.10 / 0.72 / 0.18。`config_cls: MossTTSSingleProcessPipelineConfig` 选择单进程变体；受内存约束的 24 GB 和 32 GB 配置以及 MPS DP2 方案固定使用该变体，因为它们测得的预算描述的是那种布局。

## 在启动时改变放置

同一字段也可以在命令行中用点分写法设置，无需编辑源配置：

```bash
# put the vocoder in its own process
python -m sglang_omni.cli serve \
  --model-path MODEL \
  --vocoder.process vocoder
```

重复使用同一个进程名会把其中的阶段共置在一起。下面的覆盖所产生的拓扑，正是内置 Higgs-TTS 配置已经声明的拓扑：

```bash
python -m sglang_omni.cli serve \
  --model-path bosonai/higgs-tts-3-4b \
  --preprocessing.process tts_frontend \
  --audio_encoder.process tts_frontend
```

```text
tts_frontend : preprocessing, audio_encoder
pipeline     : tts_engine
vocoder      : vocoder
```

把阶段设置到它已在其中运行的进程是一个幂等的空操作。对同一个阶段的 `process` 用不同值写两次会被当作冲突拒绝，就像任何其他被写两次的路径一样。

## 放置如何被验证

进程拓扑在启动前由放置规划器和拓扑规划器验证：

- 每个非 TP 阶段都必须声明 `process`；TP 阶段为每个 rank 派生一个进程。
- 一个进程组可以跨越 CPU 阶段和至多一个 GPU 上的阶段。
- 当多个进程组共享一个 GPU 时，所涉及的每个 GPU 阶段都必须声明 `gpu_memory_fraction`，且单个 GPU 上的总和必须满足 `placement.max_total_gpu_memory_fraction_per_gpu`。验证时会指出缺少占比声明的阶段。
- TP rank 的进程名不得与其他进程组冲突。

并非每个交接都能容忍进程边界：有些阶段通过第二个进程无法读取的进程本地注册表交换状态（例如，MOSS-TTS 流水线通过进程本地队列把准备好的请求从 preprocessing 交给 AR 引擎）。模型通过 `PipelineConfig.process_local_edges` 声明这些边，而在拓扑编译期间、任何 worker 启动之前，拆分这样的边会被拒绝。

Qwen3-TTS 只在 `preprocessing` 和 `tts_engine` 共享一个进程时才把准备好的请求保存在进程本地的模块状态中；当 preprocessing 被放到自己的进程中时，该阶段会加载一个 prompt 前端，并通过 `tensor_cpu` payload 字段把准备好的 prompt 张量传送出去，因此这条边也可以跨越进程。`tensor_cpu` 编解码器（codec）保留每个张量自身的 dtype，并让中继在控制平面之外携带它，而 `typed_tensor` 做不到这一点。

Ming-Omni-TTS 在 `StagePayload.data` 中携带预处理字段，并用 `typed_tensor` 线上编解码器序列化参考编码器的 `spk_emb` 和 `prompt_latent` 张量。因此 `preprocessing -> reference_encode` 和 `reference_encode -> tts_engine` 这两条边都可以跨越进程边界。

## 资源与性能权衡

把一个阶段拆分出去会创建另一个 OS 进程，通常还会创建另一个 CUDA 上下文。它可以通过让声码器的调度和 GPU 工作与生成过程重叠来提高吞吐量，但它也会改变 IPC 和序列化路径，可能增加空闲显存，还可能复制进程本地的缓存或运行时状态。把共享缓存或本地交接的阶段分到一组可以降低这种开销。

当多个进程共享一个 GPU 时，所有受影响的 GPU 阶段都必须声明相互兼容的 `gpu_memory_fraction` 值，且它们的总和必须满足放置限制。这些占比是放置核算层面的声明，并不是存在分配器强制执行的运行时限制的证明：只有当工厂的签名接受该参数时，工厂才会收到 `total_gpu_memory_fraction`，而 `engine.mem_fraction_static` 覆盖值可能代表另一个运行时数值。请保持两者一致。

性能取决于模型、硬件、并发度、请求形态和流式模式。在把隔离固化到模型或 YAML 配置之前，请先对目标负载进行测量。
