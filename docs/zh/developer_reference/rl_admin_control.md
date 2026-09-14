# RL 管理控制

SGLang-Omni 为推理侧的 RL 工作流提供了一组小型管理 API。其契约遵循 SGLang 与 Miles 的控制面设计，同时保留 Omni 流水线边界：

```text
HTTP / router -> Client -> Coordinator -> Stage -> Scheduler -> ModelWorker
```

控制面只承载元数据和小的结果摘要。张量数据和大体积 checkpoint 数据必须通过磁盘、分布式组或其他数据面传输。

## 认证

出于向后兼容，管理端点默认不需要认证。当以下任一项被设置时，它们要求 `Authorization: Bearer <key>`：

- 传给 worker/router `create_app(...)` 的 `admin_api_key`
- 环境变量 `SGLANG_OMNI_ADMIN_KEY`

外部 router 还接受 `--admin-api-key`。router 会把 `Authorization` 头转发给 worker，因此部署可以在两层使用同一个密钥。

## Worker 端点

worker 服务器支持：

- `GET|POST /model_info`
- `POST /pause_generation`
- `POST /continue_generation`
- `POST /update_weights_from_disk`
- `POST /update_weights_from_tensor`
- `POST /init_weights_update_group`
- `POST /destroy_weights_update_group`
- `POST /update_weights_from_distributed`
- `GET|POST /weights_checker`

`/update_weights_from_disk` 是当前主要实现的更新路径。它会暂停目标调度器，可选地中止活跃请求，调用底层 SGLang model runner 的更新方法，可选地清空缓存，然后恢复运行（除非 `keep_pause=true`）。从磁盘更新在调度器线程上执行。如果存在活跃请求，除非请求设置 `abort_all_requests=true` 或生成已经以 `mode=retract` 暂停，否则更新会被拒绝。

`/init_weights_update_group` 与 `/destroy_weights_update_group` 管理 SGLang/Miles 的分布式更新进程组。随后 `/update_weights_from_distributed` 通过管理控制面发送元数据（`names`、`dtypes`、`shapes`、`group_name` 以及可选的 `load_format` / `weight_version`），而真实张量通过分布式组传输。分布式更新路径与磁盘更新使用相同的调度器线程生命周期：活跃请求必须被中止或安全回退（retract），默认清空缓存，并在 runner 更新成功后更新可见的 `weight_version`。如果分布式更新失败，调度器将保持暂停状态，因为 SGLang 可能已部分更新了模型权重；请先通过重新加载等方式修复 worker，再调用 `continue_generation`。

`/update_weights_from_tensor` 目前仍保留给未来的张量数据面集成，worker 与 router 的 HTTP API 均返回 HTTP 501。一旦启用，失败的张量更新会使调度器保持暂停，因为张量加载不是事务性的；请先修复 worker 再调用 `continue_generation`。

## 阶段与 TP 行为

Coordinator 向每个目标阶段发送一个管理操作并等待各阶段的结果。对于 TP 阶段，rank 0 会把操作扇出（fan out）到 follower rank，收集每个 rank 的结果，并返回带 `rank_results` 的阶段级聚合结果。

没有管理能力的调度器的阶段会返回一个"成功跳过"结果，因此混合流水线可以广播模型信息或暂停命令，而不会在预/后处理阶段上失败。

## Router 行为

外部 router 将管理请求广播到每个非 dead 状态的 worker。更新与暂停路由会在广播进行期间临时把目标 worker 从正常请求路由中禁用，广播结束后恢复每个 worker 之前的禁用状态。

router 使用一把管理更新锁来串行化暂停、分布式组生命周期和权重更新广播。如果另一个更新持有锁的时间过长，router 会返回 HTTP 503，而不是无限期阻塞后续管理调用方。如果分布式组初始化失败或超时，目标 worker 将保持禁用状态，直到操作人员在恢复后显式重新启用。

## 权重检查器

`/weights_checker` 支持 `snapshot`、`reset_tensors`、`compare` 和 `checksum`。Omni 检查器基于每个张量的名称、dtype、形状和原始字节计算严格的 SHA256 摘要，再由排序后的张量摘要推导每个 rank 的校验和。全模型 SHA256 检查会阻塞该 worker 上的推理，直到摘要计算完成。
