# TTS 模型集成

关于为原生 `/v1/audio/speech` 服务添加新 TTS 模型家族的说明。请先阅读
[main.md](./main.md) 了解更宏观的阶段 / 调度器 / 协调器图景；本页只覆盖 TTS 特有的部分。

## 工作顺序

大致是添加一个新模型的步骤。每一步都在下面的章节中展开。

1. 选定 HF 架构字符串（`config.json::architectures[0]`）。
2. 搭建 `sglang_omni/models/<name>/` 骨架，包含 `__init__.py` + `config.py`。在
   `PipelineConfig` 子类上设置 `architecture` 并导出 `EntryClass`。注册表从这里找到模型。
3. 如果上游 HF config 不在原版 `transformers` 中，在导入时或 AR 工厂中调用
   `AutoConfig.register("<model_type>", <Config>)`。
4. 在 `sglang_model.py` 中编写 SGLang 模型类（或者像 Higgs 那样拆分），然后在
   `sglang_omni/model_runner/sglang_model_runner.py::_register_omni_model` 中添加一行，
   让 SGLang 能解析该架构。
5. 在 `stages.py` 中实现三个阶段工厂。AR 工厂通过
   `build_sglang_server_args` 构建服务器参数，把它们交给
   `create_sglang_infrastructure`，并返回一个 `OmniScheduler`。
6. 编写 `request_builders.py` 和 `payload_types.py`。把中止清理接入每个接触共享状态的调度器。
7. 添加 `examples/configs/<name>.yaml` 并在
   [docs/basic_usage/tts.md](../basic_usage/tts.md) 中列出该模型。
8. 添加底部列出的无需 GPU 的单元测试。

## 布局

新模型位于 `sglang_omni/models/<name>/` 之下。大多数 TTS 模型最终会有这些文件：

```text
__init__.py           # package marker; subpackages here are auto-discovered
config.py             # PipelineConfig subclass, stage list, EntryClass
stages.py             # factory functions referenced by StageConfig.factory
request_builders.py   # adapt the incoming request into scheduler input
payload_types.py      # typed state passed between stages
sglang_model.py       # SGLang-side model class registered under the HF arch
model_runner.py       # custom AR runner, only when the default does not fit
```

并非每个模型都需要每个文件（Higgs 把它的模型拆成 `model.py` +
`modeling.py`；Voxtral 把它的流水线子模块放在 `pipeline/` 下）。使用
任何合适的形态，但不要把模型代码放进框架层。

## 流水线形态

最小可用的流水线是三个阶段。Qwen3-TTS、Voxtral-TTS 和 S2-Pro
都保持这个形态；Qwen3-TTS 和 S2-Pro 把 AR 阶段叫作 `tts_engine`，而
Voxtral 使用类似的 `tts_generation` 名称。

1. **preprocessing** - 校验请求、获取并 tokenize 参考、构建 prompt
   状态。把繁重的 CPU/GPU 工作放在这里，这样 AR 循环不会被它拖住。
2. **tts_engine** - 音频或编解码器（codec）token 的自回归生成。只要
   你想要 SGLang 的 KV cache、批处理、中止处理和请求上限，就使用
   `OmniScheduler`。只有当 forward 确实是模型特定的时候才考虑自定义 model runner。
3. **vocoder** - code 转波形。一个带 `batch_compute_fn` 的
   `SimpleScheduler` 可以处理大多数批量声码器（vocoder）。当音频需要在生成完成之前离开服务器时，使用流式调度器。

代码树中有两个变体值得了解：

- 当需要每请求一次在 AR 设备上运行一个重型编码器（encoder）时，在 preprocessing 和 `tts_engine` 之间插入一个额外的 **audio_encoder** 阶段；
  Higgs TTS 为其多码本（multi-codebook）参考嵌入（embed）就是这样做的。
- 对于从 engine 到 vocoder 的按分块流式，在 engine 的
  `StageConfig` 上设置 `stream_to=["vocoder"]`，并在 vocoder 的
  `StageConfig` 上设置 `can_accept_stream_before_payload=True`。S2-Pro 是这方面的参考实现。

把以上全部内容在 `config.py` 中声明式地接线（阶段顺序、终端
标志、GPU 部署、扇出）。然后在模块作用域暴露 `EntryClass = YourPipelineConfig`，
并在类上设置 `architecture: ClassVar[str] = "<HFArch>"`。
`sglang_omni/models/registry.py` 会遍历
`sglang_omni/models/` 的每个子包，拾取每个 `EntryClass`，并把
`architecture` 属性与模型的 HF config 匹配；不需要在任何地方手工编辑列表。

代码侧跑通后，在
`examples/configs/<name>.yaml` 下放一个可运行的启动文件，并把该模型添加到
[docs/basic_usage/tts.md](../basic_usage/tts.md)，让用户有东西可以给
`sgl-omni serve --config` 用。

### SGLang 接线

`tts_engine` 工厂必须在其 GPU 上启动一个 SGLang worker。两个共享的
辅助函数承担了繁重工作：

- 来自
  `sglang_omni.scheduling.sglang_backend` 的
  `build_sglang_server_args(checkpoint_dir, ...)`
- 来自 `sglang_omni.scheduling.bootstrap` 的
  `create_sglang_infrastructure(server_args, gpu_id, *, model_arch_override=...)`，
  它返回 `OmniScheduler` 所期望的
  `(model_worker, tree_cache, req_to_token_pool, token_to_kv_pool_allocator,
  model_config)` 元组

你编写的每个 GPU 阶段工厂——engine 以及编码器和声码器（vocoder）——都要声明 `device: str | None = None, gpu_id: int | None = None`，
并用 `sglang_omni.utils.device.resolve_concrete_device` 解析它们；
绝不硬编码 `"cuda"` 或单独读取 `gpu_id`。该契约以及强制它的扫描测试在
[config.md](config.md) 中的 "Device and GPU placement
contract" 下描述。

还有两处粘合代码仍需手工添加：

- 把你的 SGLang 模型类插入
  `sglang_omni/model_runner/sglang_model_runner.py::_register_omni_model` 内的
  `ModelRegistry.models[...]`。当 HF 架构字符串与你注册的类名不匹配时，
  传入相同的键作为 `model_arch_override`。
- 如果上游权重无法干净地加载到你的 SGLang 模块中，添加一个
  `weight_loader.py`（形态参见 Higgs 的 `DiscreteWeightMapper`）。
  大多数模型不需要。

## 请求从哪里进来

`POST /v1/audio/speech` 通过
`sglang_omni/serve/speech_service.py::SpeechRequestValidator` 校验 OpenAI 载荷，然后把它
降级（lower）为一个 `GenerateRequest`。该请求进入流水线，你的模型的
请求构建器把它转换成 AR 调度器需要的任何形式。

在这个边界上有两件事容易出错：

- **端点默认值会静默覆盖模型默认值。** HTTP 层会填入一套采样默认值（目前是
  S2-Pro 的值）。对于任何其他模型，这些值看起来完全就像用户显式要求了
  它们。通过请求传递一个 `explicit_generation_params` 列表（或等价物），
  并让你的请求构建器区分“用户设置了这个”和“端点填入了它”。对端点有
  主见的任何字段都需要同样的技巧。
- **输入是异构的。** TTS 客户端以多个名称发送文本
  （`input`、`text`，有时是聊天式结构），参考也有多种形态
  （`ref_audio` + `ref_text`，或一个 `references[]` 列表）。
  在请求构建器内规范化你接受的内容，在那里校验必需的
  参考，并把该逻辑排除在 AR 阶段之外，这样坏请求会在任何东西接触 GPU
  之前失败。

构建器应该给调度器一个带类型的 dataclass（示例见各个
`payload_types.py` 文件），而不是自由形态的 dict；AR 阶段在每一步都读取这些字段。

## 调度器契约

每个调度器都暴露相同的四个方法加两个队列，阶段代码依赖于此：

```python
inbox: Queue[IncomingMessage]
outbox: Queue[OutgoingMessage]
start() -> None
stop() -> None
abort(request_id: str) -> None
```

按职责选择：

- `SimpleScheduler` - 单个可调用对象，可选地通过
  `batch_compute_fn` 批量化。适合 preprocessing 和大多数声码器（vocoder）。
- `OmniScheduler` - 封装 SGLang 的调度器方法、KV cache 和
  请求上限机制。AR 生成使用它。
- 自定义调度器 - 当以上两者都不合适时（有状态的流式
  声码器（vocoder）、自定义 detokenizer 等）。

容易坑人的部分是中止清理。任何你按 `request_id` 为键存储在调度器之外的
东西，比如从 preprocessing 到 AR 的预处理张量暂存、一个会话句柄，或一个参考缓存，都必须在三条路径上被释放：

- preprocessing 阶段在交接前中止
- AR 阶段在其请求构建器消费交接数据前中止
- preprocessing 在请求已被中止*之后*才完成，于是
  结果被直接丢弃

在每一个接触该共享状态的调度器上把同一个清理函数接为
`abort_callback`，并让它幂等；按设计，它会被同一个 id 调用不止
一次。

## 张量与设备

让张量留在生产它们的设备上，直到确实有东西需要 CPU 字节。反复出现的
错误是在请求构建器里“以防万一”地调用 `.cpu()`；它每请求付出一次
同步，而且 AR runner 还得把张量搬回去。

实际的界限：

- preprocessing 应该在 AR 设备/dtype 上创建 prompt 和参考张量，或者在
  解码交接前恰好规范化一次。
- 调度器请求数据应该以设备张量的形式携带规范化后的张量。不要在 AR
  请求数据中存储 CPU 副本，除非目标消费者明确是 CPU 侧的。
- 在不需要梯度时 detach 张量，并且只在清晰的所有权边界处做类型转换，例如
  preprocessing 输出、反馈缓冲区写入，或最终的声码器（vocoder）/HTTP 序列化。
- 元数据用的 CPU 物化，例如稳定的缓存键哈希，应该只产出元数据；它不应
  替换 prefill/decode 所使用的设备张量。

还有一个陷阱：如果你的前缀包含拼接进 token 流的连续嵌入（Higgs 和
Qwen3-TTS 都这么做），radix 缓存键必须从嵌入内容推导。否则两个恰好
共享相同占位符 token ID 的不同 prompt 会混叠到同一个 KV
前缀，你会看到一个用户的音频漏进另一个用户的。

## 评审前要测试什么

覆盖上述每条规则的无需 GPU 的单元测试：

- 请求边界 - 采样默认值保留和必需输入校验
  （见“请求从哪里进来”）
- 调度器请求数据 - 设备/dtype 不变量以及“调度器契约”下
  列出的中止清理竞态路径
- 阶段局部行为 - 你选定的声码器（vocoder）批处理或流式中的任何一种

对于端到端质量，运行共享的 TTS 基准测试：

```bash
python -m benchmarks.eval.benchmark_tts_seedtts --help
```

报告 WER/CER、样本数、吞吐量和 `rtf_mean`。如果模型在某个特定语言或
数据拆分上落后，写一句话说明可能的原因
（采样配置、codec/vocoder 版本、文本规范化、评测设置），
而不是只留下数字让它自己说话。
