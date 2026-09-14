# 参考编码服务

`ReferenceEncodeService` 拥有用于即时（ad-hoc）TTS 参考编码的可复用机制：

- 以缓存键查找；
- 字节上限的 LRU 存储；
- 编码进行中时同键单飞（single-flight）；
- 失败向等待者传播而不缓存失败；
- artifact 的存储/加载转换以及调用方持有的返回值；
- 基本的缓存统计。

该服务面向即时请求参考。已注册或已上传的音色
继续使用 `SpeakerArtifactCache`，因为它们有不同的生命周期、
键空间和失效路径。

## API 形态

实现位于 `sglang_omni/scheduling/reference_encoder.py`。

```python
@dataclass(frozen=True)
class ReferenceEncodeKey:
    model_id: str
    model_revision: str
    encoder_id: str
    encoder_config_hash: str
    artifact_kind: str
    input_key: str
    options_key: str = ""


class ReferenceEncodeHook(Generic[InputT, ArtifactT, StoredT]):
    def normalize_input(self, raw_input: Any) -> InputT: ...
    def cache_key(self, item: InputT) -> ReferenceEncodeKey | None: ...
    def encode_one(self, item: InputT) -> ArtifactT: ...
    def store_artifact(self, artifact: ArtifactT) -> StoredT: ...
    def load_artifact(self, stored: StoredT) -> ArtifactT: ...
    def revalidate(self, item: InputT, key: ReferenceEncodeKey) -> bool: ...


class KeyedReferenceEncodeHook(ReferenceEncodeHook[InputT, ArtifactT, StoredT]):
    model_id: str
    model_revision: str
    encoder_id: str
    encoder_config_hash: str
    artifact_kind: str

    def input_key(self, item: InputT) -> str | None: ...
    def options_key(self, item: InputT) -> str: ...


class TensorReferenceEncodeHook(
    KeyedReferenceEncodeHook[InputT, Tensor, Tensor]
):
    storage_dtype: torch.dtype | None
    output_dtype: torch.dtype | None


class ReferenceEncodeService(Generic[InputT, ArtifactT, StoredT]):
    def get_or_encode(self, raw_input: Any, *, desc: str | None = None) -> ArtifactT: ...
    def stats(self) -> dict[str, int]: ...
```

`ReferenceEncodeService` 是同步的、线程优先的。现有的 TTS
preprocessing 和编码器阶段已经在
`SimpleScheduler` 或 `ThreadedSimpleScheduler` 内部运行同步模型代码，所以增加异步表面只会
强迫嵌套事件循环管理而不改变底层工作。

## 职责划分

服务拥有机制：

- `_inflight` 单飞映射；
- 在服务自有锁下访问 `StageOutputCache`；
- 缓存插入、字节预算和 LRU 逐出；
- 跟随者等待、超时处理和异常扇出；
- 失败不投毒（no-poison-on-failure）行为；
- 命中、未命中、合并、失败、不可缓存输入、条目、字节数
  和逐出的统计。

hook 拥有模型语义：

- 请求特定的输入规范化；
- 可缓存性；
- 模型/checkpoint/config 的键组成部分；
- `encode_one`；
- artifact 的设备和 dtype 策略；
- 存储/加载转换；
- 针对可变本地文件的重校验。

拥有结构化身份的 hook 应继承 `KeyedReferenceEncodeHook`。它们
提供键元数据、`input_key` 和 `encode_one`，外加当服务尚未接收到
带类型条目时的输入规范化。默认实现构建
`ReferenceEncodeKey` 并在插入前重查输入和选项键。产出张量的
编码器应继承 `TensorReferenceEncodeHook`，它还会
存储一个缓存持有的 CPU 克隆，并以 `output_dtype` 返回一个调用方持有的
克隆。结构化 artifact（例如嵌套的 prompt 字典）在
`KeyedReferenceEncodeHook` 之上保留自己的存储/加载策略。

## 缓存键契约

`ReferenceEncodeKey` 必须包含所有能改变被编码
artifact 身份的输入：

- 模型家族或 checkpoint 身份；
- 模型或编码器版本；
- 编码器实现和配置哈希；
- artifact 类别；
- 规范化后的参考内容身份；
- 影响 artifact 的编码选项。

本地参考文件应使用
`reference_path_cache_key(path, trust_stat=False)` 并在缓存
插入前重校验。字节和数据 URI 载荷应按模型 hook 实际消费的
字节或原始载荷作为键。远程 URL 不应仅按 URL
字符串缓存，除非上游抓取层已经物化了不可变的
内容身份。

## artifact 策略

hook 应存储缓存持有的 artifact，通常是 detached 的 CPU 张量或一个
由 detached CPU 张量组成的小字典。`load_artifact` 必须返回一个
调用方持有的对象，通常通过克隆并移动到期望的 dtype 或
设备。`TensorReferenceEncodeHook` 通过
`storage_dtype` 和 `output_dtype` 提供该策略。服务在
存储表示上强制执行字节预算。

如果一个被存储的 artifact 大于 `max_bytes`，领导者请求和所有
同键跟随者仍会收到结果，但该 artifact 不会被插入
LRU。

## 失败与等待者

对于一个可缓存的键：

1. 缓存命中返回 `hook.load_artifact(stored)`。
2. 如果另一个请求已在编码同一个键，跟随者等待
   领导者 future。
3. 领导者编码一次，存储 artifact 表示，可选地
   把它插入 LRU，唤醒等待者，并移除 in-flight 条目。

领导者失败会传播给等待者且不被缓存。下一个请求
可以作为新的领导者重试。跟随者超时不会移除领导者的
in-flight 条目。

## M4a 与 M4b 边界

本文档只覆盖 **M4a**：即今天随代码发布的即时
参考缓存和同键单飞。

**M4b（不同键批量合并）未实现，且不是本文的目标**；
描述它只是为了标记范围边界。在剖析证明它值得额外的
调度表面之前，不要添加 M4b 运行时代码。

在构建 M4b 之前，对 FishAudio S2-Pro、Qwen3-TTS 和
MOSS-TTS Local，以并发 8 和 16 运行每请求使用不同参考音频的
冷缓存负载。跟踪 preprocessing/参考编码的 p50/p95、端到端 TTFA
和延迟、吞吐量、缓存命中/未命中/合并计数，以及 GPU/CPU 利用率。
只有当不同键的参考编码仍是首要瓶颈，且批量化相对 M4a 带来至少 15 % 的
p95 延迟降低或 20 % 的吞吐量提升时，才为该模型构建
M4b。

如果以后构建 M4b，它应该是
`ReferenceEncodeService` 的可选扩展，而不是默认路径：

- 添加显式的 hook 能力，例如默认为 `False` 的 `can_encode_batch()`，
  并只对选择加入的 hook 调用 `encode_batch(items)`；
- 添加服务旋钮，例如 `max_batch_size=1` 和 `max_batch_wait_ms=0`；
- 保持 M4a 顺序：规范化输入、计算缓存键、检查缓存、
  合并同键 in-flight 工作，然后才把不同的缓存未命中
  领导者入队以做不同键批量化；
- 使用一个最多排空到 `max_batch_size` 或
  `max_batch_wait_ms` 的内部队列，调用一次 hook 批量编码，然后对每个条目独立地
  存储、重校验、缓存插入和唤醒；
- 在批量失败时，按条目重试，这样一个坏参考不会让
  整个批次失败。

模型推广应保持证据驱动。MOSS-TTS Local 已有自己的
批量参考编码器，不应仅仅为了套用这个通用
表面而迁移。FishAudio S2-Pro 是第一个合理的候选者，但前提是剖析
显示真实收益；它的批量路径会解码/重采样每个参考、填充
波形、调用一次 codec、并拆分输出，同时保持与
`encode_one` 的一致性。Qwen3-TTS 应保持仅 M4a，除非上游封装
为 `create_voice_clone_prompt` 暴露了安全的批量原语。
