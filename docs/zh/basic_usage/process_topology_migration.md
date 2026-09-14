# 进程拓扑迁移

`StageConfig.process` 现在是进程归属的唯一来源。旧有的
`--isolate-stage`、`--stage-process` 和 `fused_stages` 条目已被移除，且不会自动迁移。

## 替代方案

| 已移除的条目 | 替代写法 |
| --- | --- |
| `--isolate-stage vocoder` | `--stages.vocoder.process vocoder` |
| `--stage-process preprocessing=frontend` | `--stages.preprocessing.process frontend` |
| `fused_stages: [[a, b]]` | 为阶段 `a` 和 `b` 设置相同的 `process` 值。 |

## 迁移 `fused_stages`

迁移前：

```yaml
fused_stages:
  - [preprocessing, audio_encoder]
```

迁移后：

```yaml
stages:
  - name: preprocessing
    process: frontend
  - name: audio_encoder
    process: frontend
```

更新完整配置时，请保留原有的 factory、routing、runtime 和 placement 字段。每个非 TP 阶段都必须声明 `process`；名称相同则共享同一个 OS 进程，名称不同则各阶段相互隔离。TP 阶段独占其进程。

顶层 `processes` 映射仅为阶段已声明的进程名称配置副本（replica）。校验与放置规则参见
[配置参考](../developer_reference/config.md)。
