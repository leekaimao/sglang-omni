# 基于 CUDA MPS 的同 GPU 数据并行

> TL;DR：基于 CUDA MPS 的同 GPU DP 可以显著提升吞吐量。在下面锁定的 TTS 测试中，饱和的 DP2 与 DP3 配置达到了调优后单副本吞吐量的 1.4 到 2.1 倍。

常见的数据并行部署为每个副本分配一块 GPU。当调优后的副本仍留有大量 GPU 余量时，把多个副本共置到同一块 GPU 上可以提升每 GPU 吞吐量。

同 GPU 数据并行在一块 GPU 上运行多个完整的服务副本，并让 [CUDA MPS](https://docs.nvidia.com/deploy/mps/index.html) 在它们之间共享 GPU。这是一项有条件的、持续演进的优化。我们很高兴把它分享出来，并呼吁社区一起探索。

![Multiple host chains plus CUDA MPS filling the idle GPU](../_static/image/same-gpu-dp-mps.svg)

## 原生运行时支持（`--mps`）

运行时可以为**一条流水线**的进程自行管理 MPS。当一条流水线把两个或更多单 GPU
的阶段进程共置到一块 GPU 上时（例如 frontend 进程与 generation 进程相邻，或
进程级副本），传入 `--mps auto`：

```bash
sgl-omni serve --model-path <model> --mps auto
```

模式（CLI 的 `--mps` 或流水线配置的 `mps:`；默认 `off`）：

* `off`：绝不触碰 MPS。
* `auto`：在该流水线拥有两个或更多单 GPU、非 TP CUDA 进程的每块 GPU 上启用
  MPS。只有一个进程的 GPU 与 TP 组不使用 MPS。
* `on`：只需一个合格进程即可启用，且不支持 MPS 的平台会报硬错误而不是警告。
  同 GPU 数据并行请使用 `on`：同一块 GPU 上的每个 `serve --mps on` 都加入
  同一个守护进程。

`auto` 与 `on` 在获取 MPS 状态之前，若某个进程解析出的放置跨越了多块物理
GPU，则会拒绝启动；这种放置请使用 `mps=off`。工厂 CUDA 设备使用收窄后
worker 的本地命名空间：`cuda:0` 与任何单 GPU 放置兼容，而非零的 `cuda:N`
被排除，因为把 worker 收窄到一个 UUID 后，它唯一合法的 CUDA 序号就是
`cuda:0`。流水线边缘的传输仍由既有的 router 与 relay 层负责；它不参与 MPS
资格判定。

守护进程按物理 GPU 共享（以设备 UUID 为键）：MPS 只为一个 server 的客户端
合并 kernel，因此第一个 serve 创建守护进程，后续的 serve 加入它，最后一个
离开的 serve 排空客户端并退出它。逻辑 GPU 序号只针对父进程的 CUDA 可见性
解析一次，然后按物理 UUID 分组；`auto` 统计每个物理组中合并的客户端进程数。
在原生 MPS 启用期间，流水线或阶段的环境默认值不得覆盖 `CUDA_VISIBLE_DEVICES`
或 `CUDA_DEVICE_ORDER`；请改为在父命令上设置它们，或使用 `mps=off`。因此
同 GPU DP 不过就是 N 条 serve 命令：

```bash
sgl-omni serve --model-path <model> --mps on --mem-fraction-static 0.35 --port 8807
sgl-omni serve --model-path <model> --mps on --mem-fraction-static 0.35 --port 8808
```

逐个启动副本，并给每个副本一个显式的内存预算（`--mem-fraction-static` 或带
阶段限定的 `--<engine-stage>.engine.max_total_tokens`），理由与下面脚本方案中
描述的 KV 定容相同。用 [Omni Router](omni_router.md) 路由流量。

进程级副本可以改用字节来定容：每个阶段的 `engine.kv_cache_bytes` 与整个副本
占用的 `total_reserve_bytes`。这些预算会在任何副本启动之前对照显卡检查：
共置要求声明占用总量，保留量与比例之和必须放得下显卡，仅 KV 池之和也必须放得
下。`engine.kv_cache_bytes` 与 `engine.max_total_tokens` 在一个阶段上互斥，
因为更低的 token 上限会静默缩小按字节推导出的池。

运行时拥有完整的生命周期。每个受管进程在服务开始前都会对照守护进程的客户端
列表核验，因为错过 pipe 目录的进程会静默回退到时间分片。服务中途守护进程身份
或控制访问丢失时，watchdog 会使流水线失败。关停时重新评估当前客户端列表，
排空本次 serve 的客户端，只有在没有其他 serve 仍拥有它时才退出守护进程。

如果受管 worker 没有在关停超时之前退出，运行时会直接终止这个由它直接拥有的
子进程，并在 launcher 退出之前收割它——即使该直接拥有的 worker 同时也是 MPS
客户端。它不会基于 MPS 快照或客户端 PID 发出额外的信号，也绝不会自动向守护
进程、未知后代或 GPU 级进程集合发信号。进程所有权与共享的 MPS 状态独立处理：
如果在 worker 退出之后无法证明守护进程身份、客户端所有权或控制状态，owner
文件会被标记为 `retained`，其锁被释放，状态目录被保留。当前命令随后带详细的
非零错误退出，而不是让 CLI owner 保持存活。

脏状态绝不自动修复。加入需要原生的 `nvidia-cuda-mps-control.pid` 身份、可响应
的控制 socket，以及每个已发布的 owner lease 仍被持有。在硬杀（SIGKILL、OOM
kill、节点崩溃）之后，即使守护进程空闲、或某个共同 owner 已死，下一次启动也
会保留状态，并带 owner/客户端细节与安全的清理指引失败退出。未加锁或
retained 的 owner 会阻塞之后的所有启动，直到运维者检查并清理该状态。既有的
健康共同 owner 继续服务，但新 owner 无法加入，也没有进程自动重试清理。清理
之后重新启动。正常关停不会留下任何残余。

运维者注意事项：状态位于 `/tmp/sglang-omni-mps-<user>/<gpu-uuid>/`
（`SGLANG_OMNI_MPS_STATE_ROOT` 可覆盖）。要在同一块 GPU 上共享的多个 serve
必须使用相同的状态根，否则它们无法发现彼此的守护进程，会各自运行互相时间
分片的 MPS server。状态根以 `0700` 权限创建；已存在的根必须已经是由当前用户
拥有、权限为 `0700` 的非符号链接目录。原生 MPS 会拒绝父进程、流水线或阶段
环境中的 `CUDA_MPS_PIPE_DIRECTORY`，而不是覆盖或加入外部守护进程。它同样
拒绝这些位置的 `SGLANG_OMNI_WEIGHT_SHARE`：在一条流水线内部，CUDA IPC 权重
共享通过 `--weight-share on` 请求，由运行时自行分配副本角色；而下面
`examples/mps_dp/launch.sh` 方案为它自己独立的 serve 进程设置该变量。

## 部署

以下步骤是一个连续流程。我们提供 `examples/mps_dp/launch.sh` 来管理一次运行
的私有 MPS 守护进程与服务副本。它记录副本进程、端口与日志，顺序启动副本，
核验其 KV 容量与 MPS 挂接，并且只拆掉它记录的那次运行。详细说明如下：

1. **选择 GPU 与 NUMA 节点。**

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
export GPU_ID=0
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $GPU_ID)
BUS=${BUS,,}; BUS=${BUS:4}
NODE=$(cat /sys/bus/pci/devices/$BUS/numa_node)   # if -1, set the node explicitly
numactl -H | grep "node $NODE cpus"
```

选择一块空闲 GPU，然后从 PCI 总线 ID 找到它的 NUMA 节点（drm 显卡序号不一定
与 nvidia-smi 序号一致）。从该节点选择互不重叠的物理 CPU 核块，每个副本一块。

2. **启动副本。**

```bash
CONFIG=examples/mps_dp/configs/higgs_h100_dp3.yaml N=3 CORE_BLOCKS="0-9 10-19 20-29" bash examples/mps_dp/launch.sh up
```

上面的命令是经过验证的 H100 Higgs DP3 方案。经过验证的 H200 Higgs DP8 方案
使用：

```bash
CONFIG=examples/mps_dp/configs/higgs_h200_dp8.yaml N=8 CORE_BLOCKS="0-3 4-7 8-11 12-15 16-19 20-23 24-27 28-31" bash examples/mps_dp/launch.sh up
```

流水线配置提供模型与各副本的运行时设置。launcher 环境提供主机本地的放置，
包括 GPU、副本数与 CPU 块。launcher 解析 GPU 的 NUMA 节点，自动分配本地端口，
启动一个私有 MPS 守护进程，在启动下一个副本之前等待每个副本的健康检查，并
核验 MPS 挂接。上面的 CPU 块只针对被测试的主机；请根据你自己的 CPU 拓扑推导
正确的互不重叠的块。

两个 profile 都把 `mem_fraction_static` 设为 `0.85`；`MF` 可以覆盖它。当
`CONFIG` 未设置时，既有的 `MODEL` 与 `MAX_TOTAL_TOKENS` 接口仍然可用。顺序
启动副本可以避免启动期间的内存 profiling 与 CUDA graph 捕获相互重叠。

相同的 `--mem-fraction-static` 标志**并不**意味着相同的 KV 容量。
`--mem-fraction-static` 针对每个副本启动时可用的 GPU 显存为模型权重与 KV 池
做预算。粗略地说，profiling 得到的 KV 显存等于模型加载前测得的空闲显存乘以
请求的比例，再减去模型与固定的运行时分配。它是每副本的预算，不是显卡上可
加和的份额。因为副本顺序启动，先启动的已经保留了显存，后启动的看到的空闲池
更小，即使所有标志都相同也会分配更少的 KV token（一次运行中，三个顺序启动的
`mf=0.27` 副本分别获得 97,503 / 53,149 / 20,961 个 KV token）。

![What same-GPU DP spends in VRAM and what it reclaims](../_static/image/same-gpu-dp-vram.svg)

内存 profiling 不会在独立的副本进程之间协调 KV 分配。当 `N > 1` 时，每个副本
必须解析出相同的 KV 容量，它可以来自任一个定容旋钮，但绝不能同时来自两者：

* 流水线配置中的 `engine.kv_cache_bytes` 确定性地为每个副本的池定容；launcher
  核验所有副本解析出相同的容量，并拒绝同时出现的 `MAX_TOTAL_TOKENS`，因为
  更低的 token 上限会静默缩小按字节推导出的池。
* 没有字节预算时，launcher 要求一个共同的 `max_total_tokens` 值（来自流水线
  配置或 `MAX_TOTAL_TOKENS`），除非每个副本都精确解析出该容量，否则拒绝启动。

任一上限都独立作用于每个副本；它不会被分摊到整个池上。它也独立于请求级的
`max_new_tokens` 限制，并且不会在副本之间分发请求。

H100 Higgs DP3 profile 每副本使用 `100000` token。H200 Higgs DP8 profile
每副本使用 `30000` token，为非 KV 的运行时分配（包括共置的音频编码器与
声码器）保留 GPU 显存余量。这些值只针对其各自的配置，不是通用的硬件默认值。
更改模型、GPU、运行时、副本数、内存设置或 CUDA graph 设置之后，请重新计算
上限。如果某个副本无法分配共同的上限，请调低它或减少副本数。

H200 profile 的 `30000` token KV 池小于 64 个请求各自生成至多 2048 个新
token 的最坏情况需求，甚至还未计入输入 token。因此 `max_running_requests=64`
是准入上限，而不是 64 个长请求可以并发解码的保证。如果池被填满，SGLang 会
收回（retract）请求并把它们放回等待队列，直到有 KV 容量可用。

3. **把每个副本驱动到饱和。**

案例研究为每个副本使用一个专用客户端，并让所有副本并行运转。测量的目标是让
每个副本保持饱和。对等价的副本，随机或轮询路由可以把共享的入口流量分摊到
整个池上；先填满一个再填下一个是另一种可行策略，但该研究没有对它们做比较。
相等的 KV 容量让副本之间可以比较，但它本身并不能平衡它们的队列。请在你自己的
工作负载下核验路由策略与每个副本的饱和度。

4. **核验 MPS 挂接。**

MPS 应当被仔细核验。有四件事很容易混淆：设置了环境变量、守护进程在运行、
存在一个 MPS server，以及你启动的副本进程确实以客户端身份挂接。只有最后一点
才能让比较有效，而错过 pipe 目录的副本会不带任何错误地回退到时间分片。
launcher 把每个副本对照 MPS 客户端列表核验，把 server 到客户端的 PID 映射
写入 `mps_attach.txt`，只要有副本未挂接就会使启动失败。

5. **路由流量。**

为了便于部署，你可以把每个副本端点注册到 [Python Router](python_router.md)。
请保持 router 的 `--max-connections` 至少与总提供并发一样大。案例研究没有对
router 的调度策略做基准测试，因此请确认所选策略能让每个副本持续被驱动，并
满足你工作负载的延迟与吞吐要求。

6. **安全拆除。**

先停止新流量，然后运行 launcher 打印出的拆除命令：

```bash
bash examples/mps_dp/launch.sh down <RUN_ID>
```

在共享主机上，只触碰你自己启动的进程，绝不要把"GPU 空了"当作成功条件。
launcher 只停止为所选运行记录的副本进程，等待它们的 MPS 客户端脱离，然后
停止私有 MPS 守护进程。只要清理无法被确认，它就会保留该运行的状态。

搭建与拆除 MPS 比运行单个副本繁琐，但在锁定的 H100 Higgs 测试中，吞吐增益
显著。下表给出名义上的完成运行区间；包括失败与降级运行在内的完整记录见
案例研究。


| 配置 | 名义吞吐量 | 相对单副本 |
|---|---:|---:|
| 单副本 c96 | 21.7 到 22.1 qps | 1.0x |
| DP2 + MPS，2 × c64 | 31.5 到 37.7 qps | 1.4 到 1.7x |
| DP3 + MPS，3 × c64 | 39.9 到 46.9 qps | 1.8 到 2.1x |

上表与 H100 案例研究中的吞吐结果来自一块 80 GB H100 上的 Higgs。H200 DP8
profile 是在完整 SeedTTS 英语数据集上以每副本并发 64 单独验证的。把任一
profile 应用到不同硬件或工作负载之前，请重新评估副本数、CPU 分配、token
容量与饱和并发。


## 跨副本共享权重（可选，默认关闭）

默认情况下，每个副本加载自己完整的一份 AR 骨干（Higgs 3-4B 为 7.60 GiB——
约占 DP3 占用的三分之一）。由于所有副本在同一块 GPU 上运行相同的只读权重，
launcher 可以改为通过 CUDA IPC 只共享**一份**副本：

> 如果你的副本位于一条流水线内（`processes.<name>.num_replicas` 配合重复的
> `replica_devices` 条目），请优先使用 `sgl-omni serve --weight-share on`：
> 运行时分配 leader 与 follower 角色，围绕它们安排启动与关停顺序，且不需要
> 环境变量。参见[进程拓扑、副本与 GPU 共享](process_topology.md)。下面的
> `WEIGHT_SHARE=1` 方案覆盖另一种形态——由脚本监管的多个独立 `serve` 进程；
> 架构支持表对两种形态都适用。

**范围与契约。** 该功能是可选的（`WEIGHT_SHARE=1`，默认关闭），并限定在
**经过验证且 tp=pp=1** 的架构上；其他情况在任何资源创建之前即被拒绝，因为
一个把每请求状态写入共享参数的模型会损坏共置的副本。仅完成架构审计并不能
启用共享：一个模型只有在文档记录的 launcher 命令于当前修订上通过端到端验证
（MPS 下共享 N=2 启动、健康检查、挂接核验、并发请求正确性、干净拆除）之后
才受支持。`WEIGHT_SHARE=1` 要求 `CONFIG`，因此受支持配置的检查在预检中运行
——早于 MPS 守护进程、状态目录或任何副本的存在。每个受支持的架构都带有一份
共享策略：模型在服务时写入的已注册张量（每步 decode 的暂存 scratch）被归类为
**副本私有**——每个副本为它们保留自己的存储，只有不可变的权重别名到同一份
存储。leader 与 follower 从同一份策略推导分类，任何不一致都会失败关闭
（它是 manifest 的一部分）。共享是整组的生命周期：leader 必须比 follower
存活更久，重启时整个运行一起重启（绝不单独重启一个副本），共享生效期间拒绝
在线权重更新，且每个 follower 要求显式的 `MAX_TOTAL_TOKENS`（它的哑权重在
KV profiling 之前被释放，因此 KV 定容必须被钉死）。`autodp.sh` 定容的是
**最大*估算***的 DP（经过启动验证），不是绝对的安全上限；因为其定容假设了
共享，它默认 `WEIGHT_SHARE=1`，而 `launch.sh` 本身默认关闭。

一个模型要么**受支持**，要么**不受支持**；没有中间层级。受支持意味着在当前
修订上以下各项全部通过，命令与日志记录在 PR 中：文档记录的 `launch.sh` 命令
以 `WEIGHT_SHARE=1` 在私有 MPS 守护进程下启动 `N=2`，每个副本通过健康检查
与 MPS 挂接核验，follower 通过 CUDA IPC 挂接 leader 的权重，且启动后不持有
共享权重的第二份驻留副本，跨副本的并发请求返回正确的输出（TTS 与单副本基线
逐字节一致的音频；ASR 逐词一致的转写，其中时间戳字段在批处理下可能出现
抖动），拆除后不残留任何副本进程或 MPS 客户端。这是单块 H100 上的端到端
冒烟验证——当前修订上可执行的支撑证据，而不是长期稳定性或 CI 声明。

| 架构 | 状态 | 配置 | 副本私有 | 共享权重规模 |
|---|---|---|---|---|
| MOSS TTS delay（`MossTTSDelaySGLangModel`） | 受支持 | `moss_delay_h100_dp2.yaml` | `_decode_input_embedding.weight`（每步 decode 暂存） | Qwen3-8B 骨干、embedding、头（17.05 GiB） |
| Higgs TTS（`HiggsMultimodalQwen3ForConditionalGeneration`） | 受支持 | `higgs_h100_dp3.yaml` | 未发现 | 全部已注册参数/缓冲（7.55 GiB） |
| MOSS TTS local（`MossTTSLocalSGLangModel`） | 受支持 | `moss_local_h100_dp2.yaml` | `_decode_input_embedding.weight`（每步 decode 暂存） | AR 骨干、embedding、局部 transformer、rope 缓冲（8.44 GiB） |
| Whisper（`WhisperForConditionalGeneration`） | 受支持 | `whisper_h100_dp2.yaml` | 未发现 | 全部已注册张量（1.51 GiB） |
| MOSS Transcribe-Diarize（`MossTranscribeDiarizeForConditionalGeneration`） | 受支持 | `moss_td_h100_dp2.yaml` | 未发现 | 全部已注册张量（1.75 GiB） |
| Qwen3-ASR（`Qwen3ASRForConditionalGeneration`） | 受支持 | `qwen3_asr_h100_dp2.yaml` | 未发现 | 全部已注册张量（3.83 GiB） |
| FunASR Nano（`FunAsrNanoForConditionalGeneration`） | 受支持 | `fun_asr_h100_dp2.yaml` | 未发现 | 全部已注册张量（1.57 GiB） |

针对 #1401 Part 1 的验证更新：在代码修订
[`5e7a8c7`](https://github.com/sgl-project/sglang-omni/pull/1557/commits/5e7a8c717b9ec85b1f72aa7d3444f9f5ff7ec72e)
上，MOSS TTS local 与 MOSS TTS delay 在 H200 上通过了 `N=2`、`WEIGHT_SHARE=1`
的健康、MPS 挂接、leader/follower 字节一致性、请求完成与干净拆除。其他
"受支持"行未重新验证。

每个受支持的模型都用 `examples/mps_dp/configs/` 中其对应的配置与相同的命令
形态启动，例如：

```bash
CONFIG=examples/mps_dp/configs/moss_delay_h100_dp2.yaml N=2 WEIGHT_SHARE=1 CORE_BLOCKS="0-7 8-15" bash examples/mps_dp/launch.sh up
```

其他一切均**不受支持**，并在预检中被拒绝——早于 MPS 守护进程、状态目录、
句柄文件或任何副本进程的存在。对于这些模型，凡已有完成的架构审计处均已记录，
但支持工作仍在推进：

* Ming TTS（`MingTTSSGLangModel`）：审计完成；受 VRAM 阻塞（仅 16.8B 的
  leader 就触及了 80 GB 显卡的边缘），等待 H200 上的验证。
* Voxtral TTS（`VoxtralSGLangTTSModel`）、Fish S2-Pro（`S2ProSGLangTextModel`）：
  审计完成；探索性运行中观察到过共享启动，但并发请求正确性需要各模型自己的
  客户端，本次验证尚不具备。
* Qwen3-TTS（`Qwen3TTSTalker`）：审计完成。在代码修订
  [`cd45a47`](https://github.com/sgl-project/sglang-omni/commit/cd45a47a1838017c89fb2178f167aac0cd7412a3)
  上，确定性推理在 H200 上以 30,000 token 上限通过了 `N=2`、`WEIGHT_SHARE=1`
  的并发字节一致性资格验证。它仍不受支持，因为该结果要求
  `enable_deterministic_inference: true`——它会串行化预处理与声码器解码，
  并禁用 Talker 编译与初始声码器 CUDA graph，显著降低吞吐。默认模式的推理
  尚未通过字节一致性契约，因此 launcher 继续拒绝该流水线的权重共享。
* LLaDA2（`LLaDA2MoeModelLM`）：审计完成；其流水线未声明生成 SGLang 阶段，
  因此 launcher 无法以任何 `N` 驱动它。
* Qwen3-Omni（`Qwen3OmniThinkerForCausalLM`、`Qwen3OmniTalker`）：两个引擎
  的审计均已完成；语音流水线运行两个 SGLang 引擎，文本流水线未声明生成阶段，
  因此 launcher 两者都无法驱动。
* Ming-Omni thinker（`BailingMoeV2ForCausalLM`）与其他所有架构：没有完成的
  审计；新增一项需要加载后变异审计、一条策略条目以及上述完整的 launcher
  验证。

对 MOSS 而言，共享只覆盖 SGLang AR 引擎：预处理与声码器 codec 实例按设计仍
按副本加载（它们持有流式状态），因此既在共享范围之外，也不在其内存节省之列。

下面的测量数据是本 PR 验证活动的性能背景（一块 80 GB H100，启动后的 VRAM，
每格一次测量，除非另行注明）；上面的支持决定依据的是当前修订的端到端运行，
而不是这些格子。只展示受支持的模型。

| 模型 | 已验证 DP，IPC 关 | 已验证 DP，IPC 开 | VRAM，关 | VRAM，开 | 每个 follower 节省 | 吞吐量（跨副本聚合），关 vs 开 |
|---|---|---|---:|---:|---:|---|
| MOSS TTS delay | **DP1**（不共享的 DP2：副本 1 无法在剩余预算中容纳自己的权重副本） | **DP2** | n/a | 42.4 GB | 17.05 GiB | 单副本 5.7 对共享 DP2 聚合 7.5 qps（+32%，每副本 3.8，单次运行）；对齐历史的输出逐字节一致；leader 与 follower 进程分别为 29.9 与 12.4 GB |
| Higgs TTS 3-4B | DP3（100k 上限；不共享的 DP4 需要 98 GB） | **DP4**（空闲 74.9 GB，负载下 78.4 GB） | DP3 时 73.7 GB | DP3 时 58.1 GB | 7.55 GiB | 在这块 H100 与驱动上，共享 DP4 按轮次匹配比共享 DP3 高 +10% 到 +40%；另外，作者的 H200 系列显示每个 N 都持平 |
| MOSS TTS local | DP3 贴近显卡边缘，声码器 graph 部分 eager | DP3 带余量，完整 graph | 78.0 GB | 61.8 GB | 8.44 GiB | 实测持平：DP2 聚合 17.9 对 18.2（每副本 9.0 对 9.1），DP3 聚合 23.4 对 23.4（每副本 7.8）qps |
| Whisper large-v3-turbo | DP3（40k 上限） | **DP6**（19.7 GB） | DP3 时 14.1 GB | DP3 时 10.7 GB | 1.51 GiB | DP3 持平（聚合 67.8 对 68.0）；共享 DP6 达到聚合 95.3 冷启动 qps，比 DP3 高 +40% |
| MOSS Transcribe-Diarize | DP3（40k 上限） | DP3 | 23.9 GB | 20.2 GB | 1.75 GiB | 持平：聚合 75.7 对 70.9 冷启动 qps（每副本 25.2 对 23.6；192 个去重片段） |
| Qwen3-ASR 1.7B | DP3（40k 上限） | DP3 | 28.2 GB | 20.2 GB | 3.83 GiB | 持平：聚合 67.5 对 64.5（每副本 22.5 对 21.5） |
| FunASR Nano | DP2（30k 上限） | DP2 | 11.8 GB | 10.2 GB | 1.57 GiB | 持平：聚合 40.6 对 42.5 冷启动 qps（每副本 20.3 对 21.3） |

"已验证 DP"两列展示每种模式在这些运行中启动并服务过的最高配置，而不是被证明
的天花板：Whisper 一路扩展到 DP6 且拐点仍未找到，因此请把小模型的 DP 视为受
CPU 核数限制，而非受 VRAM 限制。对 ASR 模型，天花板受主机限制而非 VRAM 限制，
因此共享无法移动它；共享恰好在权重构成约束之处移动天花板：MOSS delay（DP1
到 DP2）与 MOSS local（贴边缘的 DP3 到可运行的 DP3）。固定 N 下的持平是机制
层面的预期（两种情况下都是相同的 kernel 作用于相同的权重值）；单次运行的
ASR 配对差异在 7% 以内且方向不一致，MOSS local 的重复轮次相互吻合，但请把
单次测量的格子视为有待重复的观察。吞吐增益来自被释放的显存所支撑的额外副本
或 KV（delay DP2 比其单副本 +32%，local DP3 比 DP2 +29%，Higgs DP4 比
DP3 +10% 到 +40%）。

冒烟验证的正确性范围：TTS 的字节一致性对把种子序列作为首个流量的副本成立，
对 leader 与 follower 皆然，且在对端副本处于并发负载之下时也成立；MOSS 的
采样器另外对自身服务历史敏感，共享开关与否速率相同（实测对照），因此字节
比较需要对齐的历史，没有证据指向跨副本的共享污染。FunASR：63/63 个可接受
片段完全一致；语料第 64 个片段超出模型自身的 30 秒 VAD 限制，被基线与两个
共享副本一致地拒绝。

每个 follower 节省的 VRAM 是 leader 导出、每个 follower 以别名代替分配的
字节数；关与开两列大致相差 (N-1) 倍该值。ASR 吞吐使用每副本 64 个去重的合成
片段，报告的是冷启动轮次（热身重跑会被缓存放大）。共享在所有做过配对的固定
N 上都让吞吐持平；收益在于放得下（MOSS delay DP2，此前不可能）、余量
（MOSS local DP3：13.3 GB 余量与完整 graph，对比 0.33 GB 与 graph 回退）
以及 follower 内存。

## 我们如何发现这一点

这一方案源自 [#907](https://github.com/sgl-project/sglang-omni/issues/907)
中的服务 profiling。我们的 profiling 在多个 omni 服务工作负载中发现了大量
闲置的 GPU 容量，并在被测试的 ASR 部署中找到了强烈的宿主派发受限证据。由此
我们在 [Higgs](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)
与 [Moss](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html)
TTS 模型上开展了同 GPU DP 实验。

![The bottleneck is host-side dispatch, not GPU compute](../_static/image/same-gpu-dp-host-bound.svg)

| 实验 | GPU 信号 | 受控观察 | 结果 | 解读 |
|---|---|---|---|---|
| ASR 单副本 | GPU 时间线 94.3% 空闲 | SM 时钟 0.455x 时吞吐 0.90x；宿主 CPU 接近 0.25x 时 0.31x | 对 CPU 敏感，对 GPU 计算不敏感 | 该 ASR 部署中强烈的宿主派发受限因果证据 |
| Higgs 调优单副本 | SM Active 约 29%，GPU 空闲约 71% | 吞吐进入平台期，worker 被完全驱动 | 归一化 1.00x | 明确存在可回收的 GPU 余量，但达不到 ASR 那样的因果闭环 |
| Higgs DP2 不加 MPS | SM Active 约 37 到 38%，GPU 空闲约 62 到 63% | 增加了第二个同卡 server 进程 | 归一化约 1.24x | 第二个进程回收了部分空闲缺口；宿主调度与长尾批处理都可能贡献力量 |
| Higgs DP 加 MPS | 见"评估"中锁定的案例研究 | 每个副本饱和，MPS 挂接已确认 | 名义 1.4 到 2.1x，可重复 | 启用 MPS 的饱和运行产出了后续锁定测试中观察到的最大增益 |

ASR 是最强的宿主受限证据。Higgs 起初处于灰色地带，但调优后的单副本显然留有
GPU 余量。以独立进程运行多个副本会改变宿主执行、调度与长尾行为，它与扩大
单个副本的 batch 不是一回事。没有 MPS 时，CUDA 上下文大多时间分片，只回收
部分空闲；MPS 让来自不同进程的 kernel 在资源允许时并发运行，后续启用 MPS
的饱和运行产出了锁定测试中观察到的最大增益。

## 常见问题

**吞吐量已经到平台期了，为什么 GPU 还是空闲的？**

服务吞吐量不只取决于 GPU 的峰值算力。它还取决于单个副本每一步能暴露多少
并行工作、宿主侧处理调度与阶段交接有多快，以及请求长度与批处理的分布。单个
Higgs 副本可以在请求队列全满的情况下仍停在约 29% SM Active；增加第二个独立
副本让 GPU 空闲与吞吐一起改善。所以一个进程的服务路径喂不饱显卡，但原因并不
是某个单一的 CPU 函数：多条宿主执行路径、批处理行为与受延迟约束的 decode
形态都可能贡献力量。

**复制权重要花费 VRAM。这买到了什么？**

同 GPU DP 不节省 VRAM；它花得更多。它按副本复制权重，并给每个副本一个更小
的 KV 池。它买到的是原本空闲的计算被回收。这笔交易只在以下情况划算：调优后
的单副本让 GPU 空闲（有闲置的 SM 可填），且模型小到其权重只占显卡的一小块，
两三个完整副本仍然放得下。对计算受限的模型，或大到放不下多份权重副本的模型，
额外副本买不到什么。（通过 CUDA IPC 的权重共享放宽了"放得下"这一约束
——follower 挂接 leader 的副本而不是自己加载——但放宽不了"计算空闲"这一
前提。）

**为什么这在 TTS 模型上划算，而在通用 LLM 服务上不划算？**

内存放得下是使能条件，不是原因。原因是单个引擎无法回收的空闲，而 TTS 风格的
AR 音频模型同时在两个轴上产生它：

- *受延迟约束的 batch 形态。* 流式首块延迟把每副本的 batch 钉小，0.6–4B 的
  talker 在该 batch 下运行低占用率的 kernel。通常的 LLM 解法——在单个引擎中
  加深 batch——会花掉产品赖以建立的延迟预算。
- *宿主密集的服务路径。* 采样器池、声码器调度、chunk 组装与 HTTP 流式都在
  每一步做与 GPU 步长时间相当的宿主工作，因此单进程会在两次 launch 之间暂时
  闲置显卡。N 个进程让一个副本的派发空泡与另一个副本的 kernel 重叠；这也是
  同 GPU DP 的扩展对每副本分配的 CPU 核敏感的原因。

一个大型稠密 transformer 把这一切都反过来：它的 decode batch 可以增长到
GEMM 饱和 SM 阵列（单引擎内的连续批处理已经把请求复用到同一份权重副本上），
服务 batch 大小下 SM 利用率很高，MPS 没有空闲可收割——只会增加竞争——而且
每份权重副本几十 GiB，同卡副本根本放不下。那里的扩展工具是单引擎内的
TP/PP/EP，而不是 MPS 之后的 DP。经验法则：当调优后的单副本在其延迟 SLO 下
保持大约 ≤60% SM-active、且 N 倍占用放得下（权重共享可扩展"放得下"）时，
共置副本；否则扩展 batch，而不是进程数。

## 复现结果

我们发布早期结果以及下面的复现指引。

### 准备基线

单副本基线决定了同 GPU DP 是否值得做，而未被充分驱动的基线会让 DP 显得比
实际更好。先调优并测量一个副本，然后把它的吞吐、延迟与 GPU 利用率当作每个
DP 配置都必须击败的数字。

* **把并发扫到平台期。** 提高客户端并发直到吞吐不再上升，并在每一步读取
  调度器日志行（`#running-req`、`#queue-req`），而不是假设出一个良好的
  工作点。
* **了解准入上限。** Higgs 默认以 `max_running_requests=64` 与
  `cuda_graph_max_bs=64` 服务；两者都可通过
  `sgl-omni serve --max_running_requests N --cuda_graph_max_bs N` 调高
  （CUDA graph 捕获范围必须覆盖准入上限，调高它会消耗捕获显存）。默认上限
  是否构成约束取决于运行时，因此请检查队列，不要假设。
* **客户端与服务器分离。** 客户端并发不是活跃的生成 batch：超出准入上限的
  请求在调度器队列中等待，请求也会在其他流水线阶段花费时间。
* **前置条件。** NVIDIA CUDA MPS 可用且 GPU 计算模式为 `Default`，使每用户
  守护进程无需 root；足够的 GPU 显存放下每个副本的共同 KV 上限加大约固定的
  每副本开销（权重、codec、MPS 上下文）；GPU 所在 NUMA 节点上互不重叠的
  CPU 核块，每副本一块（在 SMT 机器上，逻辑 CPU `N` 与 `N + ncores` 常是
  同一个物理核，请检查 `lscpu -e=CPU,CORE,NODE`）；以及足以让每个副本
  （而不只是整个池）饱和的提供并发。


### 评估

同 GPU DP 是否有效很容易被测错，因此对每种配置都保持同样的实验纪律：

| 控制项 | 为什么重要 |
|---|---|
| 把单副本调优到其吞吐平台期 | 避免基线被人为削弱 |
| 保持 GPU 与 CPU 总资源固定 | 把副本拆分与单纯增加资源区分开 |
| 给每个副本专用 CPU 核 | 避免副本争抢宿主派发 |
| 分别让每个副本饱和 | 避免把 DP 池喂不饱 |
| 钉死软件与运行时设置 | 让比较可复现 |
| 报告延迟与不成功的运行 | 避免只展示最好的吞吐 |

## H100 上 Higgs TTS 模型案例研究

一块 H100 80 GB（驱动 580.126.20 / CUDA 13），sglang-omni `a78de4cb`，
sglang `0.5.12.post1`，`bosonai/higgs-tts-3-4b`（快照 `7556c17e`），
`/v1/audio/speech`，seed-tts-eval EN，每客户端 300 样本，默认
`max_running_requests=64` / `cuda_graph_max_bs=64`，GPU 所在 NUMA 节点的
32 个服务核按副本拆分，每副本一个客户端跑在 SMT 兄弟核上，每次运行使用全新
服务器，在共享主机上交错执行。报告每一次尝试的运行。

| 配置 | 名义吞吐量 | 相对单副本 | 运行结果 |
|---|---:|---:|---|
| 单副本 c96 | 21.7 到 22.1 qps | 1.0x | 4/4 完成 |
| DP2 + MPS，2 × c64 | 31.5 到 37.7 qps | 1.4 到 1.7x | 5 次尝试中 3 次名义完成 |
| DP3 + MPS，3 × c64 | 39.9 到 46.9 qps | 1.8 到 2.1x | 4 次尝试中 2 次名义完成、1 次降级 |

失败情况：一次 DP2 基准运行遇到 `cudaErrorMpsRpcFailure`，一个 DP2 与一个
DP3 副本启动失败，均与宿主负载尖峰同时发生。一次 DP3 运行完成了全部请求但
只有 13.3 qps，因此被标记为降级而不是被排除。钉核的单副本在所有运行中都
保持在几个百分点之内，DP3 并没有明显可重复地优于 DP2。

注意：`MAX_TOTAL_TOKENS` 设置让每副本的 KV 定容更明确、更可比。它不是
`cudaErrorMpsRpcFailure` 的直接修复，启用它之后的启动与运行时失败率也没有
重新测量；表中的失败反映的是记录时的运行。

#907 的 profiling、本重复案例研究与下面的评审者验证是三次独立的测量系列。
它们运行在不同的日期与负载上，某些情况下软件也不同，因此不应按绝对 QPS
比较；大约 61、21 与 29.9 qps 之间的差异没有归因到任何单一原因。

> 在同一锁定软件修订上的另一次评审者验证测得单副本、DP2 与 DP3 分别为
> 29.9、59.7 与 64.5 qps。两个运行环境之间的绝对吞吐不同，包括观察到的准入
> 行为差异，因此两个系列不应合并。尽管如此，两者都表明一旦每种配置都被
> 饱和，DP 就有明确增益。

要测量你自己的部署，请在采用 DP 之前检查一个调优副本在你的真实工作负载下
是否低于 GPU 饱和：

```bash
nvidia-smi dmon -i $GPU_ID -s um -d 5                        # coarse utilization
nsys profile --gpu-metrics-devices $GPU_ID --gpu-metrics-set gh100 \
  -d 60 -o one_replica -f true sleep 63                      # device-level SM-active
```

调优单副本处于峰值时较低的 SM 活动可能指示存在可回收的余量；在依赖它之前，
请用受控的 DP 对比加以确认。如果 SM 活动已经接近天花板，到此为止。

## 限制与后续工作

1. **通用性未完全验证。** 除锁定的 H100 Higgs 案例研究外，我们还在 H200 上
   做了相关实验，并用 SGLang 直接服务 Qwen3-4B；两条工作线都大致确认了同
   GPU DP 的增益。篇幅与时间限制了我们在此呈现这些结果的完整程度，测量也
   尚未达到我们期望的打磨程度。我们相信同 GPU DP 对显存与算力余量充足的小
   模型是一个有前景的方向，但实验覆盖仍不完整。

2. **KV 定容与硬件及工作负载相关。** launcher 通过共同的 `MAX_TOTAL_TOKENS`
   强制每副本 KV 容量相等。一套能跨模型、运行时与 GPU 配置泛化的定容流程
   仍需进一步研究。

3. **router 与调度器仍需更深入的研究。** router 与 SGLang Omni 调度器都还
   需要进一步优化。router 侧，共置池明显需要更好的路由策略。调度器侧，一个
   更有雄心的问题是能否借鉴 LLM prefill–decode（PD）分离的思路：保留一个
   大型共享 KV cache，让多个副本共享它。这个方向极具挑战性，我们相信相应的
   潜在收益也很大。

同 GPU DP 加 MPS 今天就能在宿主或派发受限的服务上回收空闲的 GPU 时间，但更
广泛的验证与上述工作尚未完成。如果这个方向让你感兴趣，或者你有来自其他模型、
GPU 或工作负载、能够确认或挑战这些发现的结果，我们希望与你合作。
