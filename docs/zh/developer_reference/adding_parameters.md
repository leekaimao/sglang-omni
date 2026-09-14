# 添加一个参数

一份实用指南，讲的是如何把一个新设置接入 SGLang-Omni 的配置面：它属于哪里、如何声明与校验，以及一个值如何从 YAML 或命令行到达读取它的代码。关于配置层本身的参考性描述，见 [Config](config.md)。

## 第一个问题：谁消费这个值？

stage 设置按消费者分组，而这个分组决定了其余一切——拼写方式、校验位置，以及接收该值的代码。消费者恰好有三种：

| 消费者 | 所在位置 | 示例 |
|---|---|---|
| 父进程（placement、进程规划） | stage 顶层 | `gpu`、`tp_size`、`process`、`gpu_memory_fraction` |
| stage 工厂（构造器 kwargs） | `factory.*` | `max_concurrency`、`dtype`、`prefill_coalesce_requests` |
| SGLang 引擎（ServerArgs） | `engine.*` | `mem_fraction_static`、`max_running_requests`、`disable_radix_cache` |

每个分组都对应同样的两种拼写——一种路径语言，写了两次：

```yaml
# YAML: the stages: mapping, keyed by stage name
stages:
  tts_engine:
    tp_size: 2
    factory:
      max_concurrency: 8
    engine:
      mem_fraction_static: 0.7
```

```bash
# CLI: dotted flags, the stages. prefix implied
sgl-omni serve --config omni.yaml \
    --tts_engine.tp_size 2 \
    --tts_engine.factory.max_concurrency 8 \
    --tts_engine.engine.mem_fraction_static 0.7
```

两种拼写都会变成按叶子（per-leaf）粒度的补丁，并经由同一套机制解析，因此优先级（CLI 高于文件、文件高于模型默认值）、重复检测，以及 `sgl-omni config explain` 的来源信息都是免费获得的。这些你都不用自己写。

## 情形 1：仅被一个模型使用的工厂 kwarg

大多数参数都属于这种情形，而且它**完全不需要改动 schema**。`factory` 分组接受未声明的键并不加改动地传递；工厂的签名就是校验。

1. 把该参数加进你的 stage 工厂签名，连同它的默认值：

```python
# sglang_omni/models/dots_tts/stages.py
def create_vocoder_executor(
    model_path: str,
    *,
    stream_slots: int = 16,
    ...
) -> DotsTTSStreamingVocoder:
```

2. 没有第二步。用户立刻就可以设置它：

```bash
sgl-omni serve ... --vocoder.factory.stream_slots 8
```

运行时按名字把 `factory.*` 的值叠加到作者的 kwargs 之上，并且**拒绝工厂不接受的键**——像 `--vocoder.factory.stream_slotz 8` 这样的拼写错误会在启动时失败，并指名 stage 和参数。这之所以可行，是因为工厂显式声明了每个参数：永远不要给 stage 工厂加 `**kwargs` 兜底，它会把拼写错误和被静默忽略的设置都变成空操作。

### 校验一个模型特有的参数

自由格式的键并没有被关在主流校验流水线之外。当参数有值得声明的规则——取值范围、枚举、急切校验——就为它定类型，与共享字段完全一样：继承该分组、以静态约束声明该字段，并用它为 stage 指定类型。

```python
class VocoderFactoryArgs(FactoryArgs):
    stream_slots: int | None = Field(default=None, ge=1, le=64)

class VocoderStageConfig(StageConfig):
    factory: VocoderFactoryArgs = Field(default_factory=VocoderFactoryArgs)

class MyPipelineConfig(PipelineConfig):
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "vocoder": VocoderStageConfig,
    }
```

`stages.vocoder.factory.stream_slots` 现在是一条带类型的路径，得到与直接声明在 `FactoryArgs` 上的字段完全相同的对待，且作用域限于一个 stage：静态范围在解析时强制执行，无损转换规则适用于 CLI 文本和 YAML 标量，`config explain` 会枚举它。这与 `EngineStageConfig` 使用的模式相同，也与共享分组的原则相同：约束是字段上的声明，而不是手写的检查。（带有真实字段的子类不是空壳。）

另外两个位置覆盖字段声明无法表达的内容——它们同样在解析时运行，因为解析器在每次合并时都会重建并重新校验流水线类：

- **跨字段与跨 stage 的规则**放在流水线类的 `model_post_init` 里——Ming-TTS 在那里校验它的音频解码节奏与批处理契约，Ming-Omni 校验它的 GPU 冲突规则。
- **需要消费者运行时状态的规则**留在消费者里——dots 声码器（vocoder）拒绝与 latent 引擎准入限制不一致的 `stream_slots`，这一关系只在启动时才知道。

一个除了"工厂接受它"之外没有别的规则的参数不需要以上任何东西；签名默认值和内建的未知 kwarg 拒绝机制就足够了。

## 情形 2：跨模型共享的工厂 kwarg

当一个旋钮普遍到若干条流水线都要调节它——批处理、并发、合并——就在 `sglang_omni/config/schema.py` 的 `FactoryArgs` 上声明它，使它被急切校验并出现在路径枚举中：

```python
class FactoryArgs(BaseModel):
    ...
    max_concurrency: int | None = Field(default=None, ge=1)
    prefill_coalesce_wait_ms: float | None = Field(default=None, gt=0)
```

已声明字段的规则：

- **默认值是 `None`，表示"未设置"。** 工厂自己的签名默认值继续负责；只有被设置的值才会传递。永远不要把工厂的默认值复制进 schema——那会成为第二事实来源。
- **静态声明取值范围**，用 `Field(ge=/gt=/lt=/le=)`，字符串枚举用 `Literal[...]`，非空字符串用 `Field(min_length=1)`。Pydantic 在构造和解析时强制执行约束，路径编译把同一约束带入 CLI/YAML 转换——一次声明，覆盖所有通道。不要把范围写成手写的 `model_post_init` 检查；那要留给声明无法表达的规则（见下文）。
- **类型形状已经处理好了。** 转换只做无损转换：int 可以放进 float 字段，`0`/`1` 可以放进 bool 字段；但数值字段拒绝 bool，int 字段拒绝 float，对 CLI 文本和原生 YAML 标量一视同仁。这些你是免费得到的。

分组上的 `model_post_init` 只用于声明说不出的事情：跨字段规则，或建议性警告（`prefill_coalesce_requests=1` 会警告该值是空操作）。

## 情形 3：引擎（ServerArgs）设置

`engine.*` 映射到该 stage 引擎的 SGLang ServerArgs，并且只存在于引擎 stage 上——一个 stage 通过使用 `EngineStageConfig`（经由流水线类中的 `stage_config_types`）把自己声明为引擎 stage。在任何其他 stage 上写 `engine.*` 都是路径错误，因此用户无法设置没有代码会读取的值。

- 任何 ServerArgs 键都已经可以作为自由格式透传使用：`--tts_engine.engine.disable_radix_cache true`。键是否存在由 SGLang 决定，判定发生在引擎启动时。
- 只有当一个键被普遍调节、值得急切校验时，才在 `EngineArgs` 上声明它，例如 `mem_fraction_static: Field(gt=0, lt=1)`。

消费者用 `stage.engine.overrides()` 读取被设置的键——一个恰好包含所写内容的 dict，其余一切由 SGLang 自己的默认值支配。

## 情形 4：stage 放置（placement）设置

由父进程在任何 stage 代码运行之前读取的字段——placement、进程分组、TP 形状——位于 `StageConfig` 的 stage 顶层。只有当启动器或规划器确实消费它时才添加。校验拆分与别处相同：范围用静态 `Field` 约束（`tp_size: Field(default=1, ge=1)`），跨字段形状用 `model_post_init`（TP stage 的 `gpu` 列表必须与 `tp_size` 匹配且唯一）。

**不要**在下游重新检查这些规则。规划器和拓扑遍历器信任已校验的配置；一个约束只有一个落点。

`device` 和 `gpu_id` 是这条规则的特例：placement 拥有 `gpu_id`，工厂只能接收设备*类型*，工厂通过一个共享辅助函数解析这一对。在改动二者中任何一个之前，先阅读 [config.md](config.md) 中的 "Device and GPU placement contract" 一节。

## 情形 5：流水线级别的设置

整条流水线的值（`model_path`、`name`、`placement.*`）是 `PipelineConfig` 的顶层字段，书写时不带 stage 前缀（`--model_path ...`，YAML 顶层）。模型特有的流水线类可以用同样的方式添加自己的字段（见 `MossTTSLocalPipelineConfig` 的 cache 和 cuda-graph 字段）。单个模型的跨 stage 不变量属于该流水线类的 `model_post_init`——例如 Ming-Omni 拒绝与 thinker 的 TP 范围冲突的 talker GPU。

## 用户不设置的值

对于派生而非配置的值，存在两种机制。最后才考虑使用它们。

**作者派生的工厂 kwargs** —— 流水线类上的 `stage_factory_kwargs(stage_name)` 返回某个 stage 的启动时构造器 kwargs。当流水线作者比静态默认值更了解情况时使用它（例如 qwen3-tts 固定确定性推理设置）。两条硬规则：按键而言配置通道获胜（显式的 `factory.*` 值覆盖钩子的值），并且钩子不得读取*其他* stage 的配置——跨 stage 共享是用户要在 YAML 里做的声明：

```yaml
# shared: writes one value into several stages, by selector
shared:
  - select: {engine: true}          # every SGLang engine stage
    engine:
      mem_fraction_static: 0.6
  - select: {stages: [talker, vocoder]}
    factory:
      dtype: bfloat16
```

显式的逐 stage 写入永远优先于 `shared:` 展开。

**合并后派生** —— 只在所有来源合并之后才存在的值（例如在 `serve` 中由已解析的 TP 大小和 GPU 拓扑派生的 `disable_custom_all_reduce`）。派生是一种兜底：它只能填充没有来源设置过的键，永不覆盖显式值，并且 `config resolve`/`explain` 必须运行同一派生，使预览与启动一致。如果你发现自己在想要第三种此类机制，停下来问问这个值是否可以干脆作为配置默认值。

## 校验放在哪里：单一落点规则

每条规则都恰好有一个归属，由规则需要看到什么来决定：

| 规则需要 | 落点 | 示例 |
|---|---|---|
| 只需要值本身 | 静态 `Field` 约束 / `Literal`——放在共享分组上，或为单个模型建立逐 stage 的分组子类 | `mem_fraction_static: Field(gt=0, lt=1)`；`VocoderFactoryArgs.stream_slots` |
| 正确的转换 | 无——无损强制转换是内建的 | int 字段拒绝 bool |
| 同一对象上的兄弟字段 | 该模型的 `model_post_init` | TP 的 `gpu` 列表与 `tp_size` 匹配 |
| 一条流水线的多个 stage | 流水线类的 `model_post_init` | Ming GPU 冲突检查；Ming-TTS 音频解码契约 |
| 消费者的运行时状态 | 消费者，在使用点 | 声码器（vocoder）的 `stream_slots` 与 latent 引擎 |
| 工厂的参数列表 | 无——签名检查是内建的 | 拒绝未知的 `factory.*` 键 |

反模式，每一种都至少从这个代码库中移除过一次——不要重新引入：

- **第二个校验落点。** 如果 serve 预检了一个 schema 也在检查的范围，那么在规则改变的那一天，两者之一就是错的。
- **能力白名单。** "支持 X 的工厂"清单会过时；工厂签名已经说明了它接受什么。
- **保留字段和投机枚举。** 每个字段和每个枚举成员今天就必须有消费者。
- **空子类壳。** `StageConfig` 子类必须承载真实差异（如 `EngineStageConfig.engine_stage`）；只会改名的壳是噪音。
- **为未发布的拼写做兼容。** 迁移提示覆盖的是已在 `main` 上发布过的拼写；只在你分支上存在过的抽象会被无声删除，不留痕迹。
- **用户配置中的拓扑。** 哪些 stage 存在、它们如何路由、请求从哪里进入——这些属于模型的 `config.py`，永远不属于 YAML 或 CLI。

## 新参数检查清单

1. 说出消费者是谁；这就确定了分组和拼写。
2. 先走自由格式：模型特有的工厂 kwarg 只需要一个签名参数。只有跨模型的旋钮才在 `FactoryArgs`/`EngineArgs` 上声明。
3. 范围和枚举是静态声明（`Field(...)`、`Literal`），不是代码。
4. 跨字段规则放进所属模型的 `model_post_init`；没有任何东西被检查两次。
5. 端到端验证整个面：

```bash
sgl-omni config resolve --model-path <model> --<stage>.factory.<name> <value>
sgl-omni config explain stages.<stage>.factory.<name> --model-path <model>
```

`resolve` 必须在消费者将要读取的地方显示该值，`explain` 必须把它归因于你的来源。两个命令构建的补丁集与一次启动构建的完全相同，因此如果预览是对的，启动就是对的。

6. 测试：在接受路径（值到达消费者）和拒绝路径（超范围，以及自由格式键的未知 kwarg 拒绝）上打固定测试，放在拥有该规则的层的测试套件里。
