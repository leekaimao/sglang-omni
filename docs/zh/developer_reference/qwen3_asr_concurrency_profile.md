# Qwen3-ASR 高并发基准测试与瓶颈剖析

这是 issue #1324 的 Q-PR2 交付物：一份可复现的 current-main 上
Qwen3-ASR 服务在并发 1–64 的基准测试，与固定服务基线在相同的预分段
音频上对比，并带有足够的内部遥测来归因请求时间去了哪里。下面的数字是
Q-PR3（#1326，已合并）、Q-PR4 和 Q-PR5 的 before 基线。

## 方法

- 模型：`Qwen/Qwen3-ASR-1.7B`，bf16，每服务器一块 80 GB GPU（DP=1）。
- 数据：完整的 SeedTTS 参考集——EN 1088 条、ZH 2020 条——由
  `benchmarks.dataset.seedtts` 分段；两个系统通过同一个客户端接收相同的
  文件（`benchmarks/eval/benchmark_asr_seedtts.py`）。
- 扫描：并发 `1, 8, 16, 32, 64`，每档一次丢弃的预热加上三次
  测量重复；闭环客户端。
- current 是评审中 commit 上的 SGLang-Omni，用默认的
  Qwen3-ASR 流水线服务（`max_running_requests=32`、
  `request_build_max_workers=2`、`request_build_max_pending=16`，异步
  decode 开启）。基线是同一 checkpoint 的固定 OpenAI 兼容服务栈及其
  原厂配置，跑在同一主机的相同 GPU 上。双方都没有调优。
- 遥测：按请求的 profiler 事件（请求构建、准入队列、
  首次 forward、decode 尾部）、主机 CPU 和按 GPU 的利用率采样、原始
  按请求 JSONL，以及环境指纹，均由本 PR 添加的基准测试的
  `--profile-events --sample-util --fingerprint --save-raw-dir`
  标志捕获。

### 复现

```bash
# current
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --port 8511

python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8511 --concurrencies 1,8,16,32,64 --repeats 3 --warmup \
  --profile-events --profile-event-dir /tmp/asr_profile \
  --sample-util --util-gpu-ids <gpu> --fingerprint \
  --save-raw-dir raw-current --output asr_en_current.json

# baseline: point the same client at the baseline server's port
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port <baseline-port> --concurrencies 1,8,16,32,64 --repeats 3 --warmup \
  --sample-util --util-gpu-ids <gpu> --fingerprint \
  --output asr_en_baseline.json
```

## 共享主机方差

测量主机是共享的；共租户负载在多次运行之间漂移，并直接体现在这些
主机瓶颈型负载中（见 #907）。安静主机和
繁忙主机上的运行都有报告：绝对峰值会移动，但结构性结论——并发拐点、其成因以及
阶段分解——在每次运行中都相同。current 与基线的对比是
背靠背进行的，所以双方看到相似的外部负载。

## 结果 — current（安静主机）

SeedTTS EN，1088 条，三次重复的均值：

| conc | req/s | RTFx | lat mean | lat p95 | corpus WER | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.6 | 54.8 | 0.088 s | 0.129 s | 0.0122 | 3264/3264 |
| 8 | 27.5 | 130.4 | 0.289 s | 0.437 s | 0.0122 | 3264/3264 |
| 16 | 53.0 | 251.2 | 0.304 s | 0.458 s | 0.0122 | 3264/3264 |
| 32 | 108.0 | 511.4 | 0.295 s | 0.399 s | 0.0122 | 3264/3264 |
| 64 | 103.0 | 486.8 | 0.610 s | 0.741 s | 0.0125 | 3165/3264 |

SeedTTS ZH，2020 条，三次重复的均值：

| conc | req/s | RTFx | lat mean | lat p95 | corpus CER | completed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.3 | 66.9 | 0.070 s | 0.088 s | 0.0062 | 6060/6060 |
| 8 | 64.2 | 300.8 | 0.124 s | 0.175 s | 0.0064 | 6060/6060 |
| 16 | 96.5 | 451.9 | 0.165 s | 0.231 s | 0.0062 | 6060/6060 |
| 32 | 126.4 | 591.6 | 0.252 s | 0.353 s | 0.0062 | 6060/6060 |
| 64 | 129.2 | 604.3 | 0.489 s | 0.613 s | 0.0062 | 5993/6060 |

解读：

- 吞吐量在并发 32 处饱和，在两种语言上都不再扩展到 64，同时平均
  延迟大约翻倍。
- 在并发 64，两种语言都流失 1–4 % 的请求并返回 HTTP 500：单个
  worker 最多准入 `request_build_max_pending=16` 个请求构建和
  `max_running_requests=32` 个运行中请求，所以 64 深的闭环
  溢出了构建积压。
- WER/CER 在每一档都保持在区间内（并发 64 的小幅上升是被丢弃请求的
  分母效应，不是转写回归）。
- 相对于 #1326 之前的参考（在相同 EN 数据集和硬件档次上并发 32 为
  97.9 req/s），异步 decode 把饱和吞吐量提升了约 10 %。

## 结果 — current 对比基线（背靠背，同一主机窗口）

SeedTTS EN，1088 条，三次重复的均值；两次扫描在
相同的相邻 GPU 上背靠背运行，所以双方看到相同的共租户
负载。基线的一次并发 1 重复被共租户突发命中
（7.3 req/s，对比 22.6/21.4）；下面的均值包含它。

| conc | current req/s | baseline req/s | ratio | current lat mean/p95 | baseline lat mean/p95 | current completed | baseline completed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.0 | 17.1 | 1.42× | 0.083 / 0.102 s | 0.076 / 0.149 s | 3264/3264 | 3264/3264 |
| 8 | 41.9 | 70.3 | 1.68× | 0.191 / 0.256 s | 0.117 / 0.174 s | 3264/3264 | 3264/3264 |
| 16 | 62.6 | 120.6 | 1.93× | 0.255 / 0.346 s | 0.132 / 0.171 s | 3264/3264 | 3264/3264 |
| 32 | 79.5 | 183.8 | 2.31× | 0.401 / 0.549 s | 0.175 / 0.242 s | 3264/3264 | 3264/3264 |
| 64 | 80.9 | 264.6 | 3.27× | 0.777 / 0.947 s | 0.238 / 0.291 s | 3208/3264 | 3264/3264 |

Corpus WER 在每一档都保持在 0.0122–0.0125（current）和
0.0123–0.0124（基线）。解读：

- 差距是扩展性差距，不是单请求模型 forward 差距：基线
  越过 current 的并发 32 饱和点继续扩展，并在 64 处完成每个
  请求，而 current 在那里流失 1–2 %。
- 排除受竞争的重复后，基线即使在并发 1 也快约 1.8×
  （均值 0.044 s 对 0.083 s），所以差距的一部分是固定的
  每请求服务开销，其余部分随并发增长——与
  下方量化的准入上限和主机分发成本一致。
- 这在方向上复现了外部报告，只是幅度在这套
  硬件/时间窗上更大；路线图描述的并发拐点在
  current main 上得到确认。

SeedTTS ZH，2020 条，三次重复的均值，同样的背靠背配对
（current 的并发 1 重复在这个时间窗内部分受竞争）：

| conc | current req/s | baseline req/s | ratio | current lat mean/p95 | baseline lat mean/p95 | current completed | baseline completed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.1 | 25.2 | 2.78× | 0.116 / 0.178 s | 0.039 / 0.053 s | 6060/6060 | 6060/6060 |
| 8 | 47.8 | 131.0 | 2.74× | 0.167 / 0.238 s | 0.061 / 0.083 s | 6060/6060 | 6060/6060 |
| 16 | 68.4 | 186.5 | 2.73× | 0.233 / 0.324 s | 0.085 / 0.117 s | 6060/6060 | 6060/6060 |
| 32 | 86.6 | 245.5 | 2.83× | 0.368 / 0.522 s | 0.129 / 0.164 s | 6060/6060 | 6060/6060 |
| 64 | 82.1 | 265.7 | 3.24× | 0.772 / 1.017 s | 0.238 / 0.339 s | 5961/6060 | 6060/6060 |

Corpus CER 在每一档都保持在 0.0061–0.0063（current）和
0.0064–0.0066（基线）。

## 瓶颈分解（带剖析的采样，EN）

来自请求事件 profiler 的按请求阶段均值；在这条路径上编码器工作运行在
首次 LM forward 内部，所以"first forward"是编码器 +
prefill：

| conc | build | build→queued | queued→scheduled | first forward | decode tail | total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.6 ms | 0.7 ms | 1.1 ms | 28.0 ms | 55.3 ms | 91 ms |
| 8 | 6.0 ms | 16.5 ms | 2.0 ms | 29.5 ms | 131.8 ms | 196 ms |
| 16 | 6.1 ms | 20.0 ms | 1.9 ms | 30.1 ms | 185.6 ms | 259 ms |
| 32 | 6.5 ms | 23.1 ms | 5.9 ms | 46.2 ms | 357.7 ms | 472 ms |
| 64 | 6.3 ms | 30.1 ms | 412.8 ms | 33.3 ms | 392.6 ms | 910 ms |

GPU 利用率在每一档都保持在 33 % 到 54 %（均值）之间；GPU 从来
不是约束。

按顺序排列的主要成本：

1. **并发 64 处的准入排队。** queued→scheduled 等待从约 6 ms 跳到
   约 413 ms——整个拐点都在这里。超出
   `max_running_requests=32` 的请求等待；超出构建积压的请求
   被拒绝。这是 Q-PR5 的目标。
2. **decode 步的主机分发。** 随并发上升，decode 尾部从 55 ms 增长到
   每请求约 360 ms，而 GPU 利用率下降——decode 墙钟时间由运行
   批次共享的按步主机侧工作主导，与 #907 的因果剖析
   一致（GPU-idle-94 %、CPU-sensitivity-0.69）。Q-PR5 的旋钮扫描以及
   任何进一步的主机侧批处理改进都针对这一点。
3. **首次 forward 停顿。** 编码器 + prefill 每个被准入的请求花费
   28–46 ms，并在调度器线程和默认流上执行，所以每次
   准入都会让整个运行中的 decode 批次停顿那么久（并强制一次
   异步 decode 排空）。这是 Q-PR4 的目标：把编码器挪到
   LM 准入之前，放到它自己的线程/流上并批量化。
4. **build 到 queued 的间隙。** 并发 ≥ 8 时为 16–30 ms，来自 FIFO
   构建结果排空和准入锁路径；小但可见，同样在
   Q-PR5 范围内。

## 环境

指纹嵌入在每个由
`--fingerprint` 产出的结果 JSON 中：一个客户端块（git SHA、依赖冻结哈希、驱动、
GPU、缓存的模型版本——描述基准测试进程）和一个服务器
块（目标服务器对自身的报告）。在上面的运行中，
客户端和服务器共享同一主机和 checkout，所以客户端 git 状态同样
标识服务器代码。原始的按请求记录和 profiler 事件 JSONL
存放在测量主机上的结果文件旁边。

上述运行的摘要：NVIDIA H100 80GB HBM3、驱动 580.126.20、
torch 2.11.0+cu130、sglang 0.5.16、transformers 5.12.1、模型快照
`7278e1e70fe206f11671096ffdd38061171dd6e5`、依赖冻结
`49652ce3ea5a7720…`。current 侧的数字产出自本
分支的 commit；注意指纹的 `git.dirty` 标志同样会计入
未跟踪的运行输出目录。
