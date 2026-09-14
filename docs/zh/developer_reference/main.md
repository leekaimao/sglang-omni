# 架构

SGLang-Omni 是面向 Omni 模型（接受文本、图像、音频、视频混合输入，并可输出文本、音频或其他模态的模型）的多阶段运行时。

## 系统总览

```text
HTTP API -> Client -> Coordinator -> Stage -> Scheduler -> ModelRunner -> model forward
```


| 层 | 职责 |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| [HTTP API](./apiserver_design.md) | OpenAI 兼容的请求/响应 schema、SSE 分帧、HTTP 错误 |
| [Client](./apiserver_design.md)   | `GenerateRequest` 到 `OmniRequest` 的转换、结果聚合、音频编码 |
| [Coordinator](./pipeline.md)      | 请求生命周期、入口阶段提交、终态结果收集、中止广播 |
| [Stage](./pipeline.md)            | 控制面 IO、relay IO、扇入、流路由、调度器收件箱/发件箱桥接 |
| [Scheduler](./pipeline.md)        | 各阶段的执行循环，以及向阶段发件箱的失败传播 |
| [ModelRunner](./pipeline.md)      | AR 前向准备、模型前向调度、输出抽取 |
| [Communication](./communication.md) | 阶段之间的控制面消息与 relay 数据传输 |
| [TTS Integration](./tts_model_integration.md) | 新增 TTS 模型家族的检查清单与生命周期规则 |

具体设计细节请参阅各层专属文档。

## 目录布局

```text
sglang_omni/
|-- pipeline/       # Inter-stage orchestration, stages, coordinator, processes
|-- scheduling/     # Scheduler loops and inbox/outbox message types
|-- model_runner/   # Shared model runner abstractions for AR stages
|-- models/         # Model-specific configs, stages, request builders, modules
|-- config/         # PipelineConfig, StageConfig, config manager, topology
|-- relay/          # Data transfer backends
|-- serve/          # HTTP server and OpenAI-compatible API adapter
|-- client/         # Internal client used by API adapters
`-- proto/          # Request, payload, stage, and control-plane message types
```

## 模型目录约定

模型专属代码应放在 `sglang_omni/models/<model>/` 之下。

推荐布局：

```text
models/<model>/
|-- config.py             # PipelineConfig subclass and StageConfig list
|-- stages.py             # stage factories
|-- routing.py            # optional data-driven routing helpers
|-- request_builders.py   # inter-stage payload transforms
|-- payload_types.py      # typed model-specific payload state
|-- callbacks.py          # feedback callbacks or strategy, when needed
`-- components/           # model modules, processors, vocoders, adapters
```

只有模型局部的行为应该放在这里。框架自有的层仍然是
`Stage`、`Coordinator`、调度器、model-runner 基类、relay、运行时准备，以及
runner。
