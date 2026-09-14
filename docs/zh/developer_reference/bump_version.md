# 升级 SGLang 版本锁定

SGLang-Omni 固定（pin）一个 SGLang 发布版本，以及该发布版本所固定的整套技术栈。移动这个 pin 就是一次版本升级 PR。本页介绍哪些内容需要一起变动、Omni 在公开 API 之外依赖 SGLang 的哪些地方、CI 镜像如何更新，以及在 PR 令人信服之前必须测量什么。请先阅读 [main.md](./main.md) 以了解 stage / 调度器 / model-runner 的整体图景；下文提到的各个接缝正是那一页所引入的。

Omni 并不是 SGLang 的一个薄调用方。`OmniScheduler` 借用上游 `Scheduler` 中它没有覆写的方法并把它们放在自己身上运行，`SGLModelRunner` 继承自 `ModelRunner`，执行桥从 Omni 自己的事件循环驱动 `ForwardBatch`，而引擎构建器在引擎生命周期的固定节点修改 `ServerArgs`。以上每一项都是与某一个 SGLang 发布版本的契约，而上游可以在不改动任何公开签名的情况下移动其中任何一项。这就是为什么一次版本升级要按契约变更来评审，而不是当作一次 pin 编辑。

## 哪些内容一起变动

pin 集合就是上游 `python/pyproject.toml` 在目标 tag 上固定的全部内容。先把它与当前 tag 做 diff，并把该 tag 对应的 `lmsysorg/sglang` 镜像解析到一个 digest；这次升级的规模取决于 torch、CUDA 或 Python 是否发生了变动，而不是取决于 SGLang 的版本号。

Omni 在 `pyproject.toml` 中镜像了这些 pin：`sglang`、`torch`、`torchvision`、`torchcodec`、`flashinfer_python[cu13]`、`flash-attn-4`、`kernels`、`numba`、`transformers`，以及当上游变动时随之变动的 `torchaudio`。`sglang-kernel` 不由 Omni 固定；它随镜像一起提供。每一处说明某个 pin 与 sglang 技术栈一致的注释，都标记了一行需要重新检查的内容。

版本还存在于其他一些地方：

| 文件 | 承载的内容 |
|---|---|
| `docker/Dockerfile` | `SGLANG_IMAGE`（新 tag 的 cu13 manifest 的 digest）、FlashInfer 重装版本、JIT 缓存路径 `/root/.cache/flashinfer/<version>`、`FLASHINFER_CACHE_IMAGE` |
| `.github/workflows/*.yaml` | 每一行 `image:`，均按 digest 固定 |
| `docker/cpu.Dockerfile` | `SGLANG_IMAGE`（新 tag 的 `-xeon` manifest 的 digest） |
| `docker/xpu.Dockerfile` | `SGLANG_XPU_BRANCH`（该 tag）和 `SGL_KERNEL_XPU_REF`（该 tag 之前最后一个 `sgl-kernel-xpu` 提交） |
| `pyproject_cpu.toml`、`pyproject_xpu.toml`、`scripts/cpu/install_cpu.sh`、`scripts/xpu/install_xpu.sh` | 已验证的 SGLang tag；各提供商的 pyproject 无法固定 `sglang`，因为每个 wheel 都会拉取 CUDA 版 torch |
| `docs/get_started/installation.md`、`docs/get_started/installation_cpu.md`、`docs/get_started/installation_xpu.md`、`docs/basic_usage/tts.md`、`docs/cookbook/*.md`、各模型 README | 安装说明中的版本名 |
| `sglang_omni/` 中的注释 | 永远不写具体版本名；应陈述代码所依赖的不变量，使文字在下一次升级后依然成立 |

在整个代码树中搜索旧版本号和旧镜像 digest；上表就是过往升级所触及的内容。

Intel CPU 和 XPU 技术栈固定它们各自的 SGLang tag 和基础镜像，它们的 CI 工作流在每个触及 `sglang_omni/` 的 PR 上都会构建 `docker/cpu.Dockerfile` 和 `docker/xpu.Dockerfile`。`sglang_omni/platforms/` 会在 import 时导入被固定发布版本的模块，因此在这两套技术栈随之变动之前，这些工作流会在升级时失败：`-xeon` 镜像 digest、XPU tag 及其 `sgl-kernel-xpu` 修订，以及各提供商 pyproject、安装脚本和安装文档中已验证的 tag。该 tag 处上游的 `docker/xpu.Dockerfile` 会指名任何新的构建前提。

ROCm、NPU 和 MUSA 技术栈（`docker/rocm.Dockerfile`、`pyproject_rocm.toml`）固定它们各自的 SGLang tag 和基础镜像。没有任何项目 CI 构建它们，因此升级 PR 不去动它们，并在自己的描述中说明这一点，同时把新 tag、匹配的提供商镜像 digest（如果存在），以及在 `sglang_omni/platforms/` 中所做的任何平台分发变更移交给对应的提供商负责人。

## Omni 在哪些地方依赖 SGLang

Omni 与 SGLang 的契约并不都能从 import 列表中看到。下面这些面（surface）是过往升级不得不重新检查的，每一项都附有一个它如何损坏的例子。逐一检查它们，并料到某个发布版本会新增一个本列表没有提到的面。

**Import。** `sglang_omni/` 和 `tests/` 中的每一条 `from sglang... import`。大多数文件直接 import 上游；`sglang_omni/vendor/sglang/` 重新导出 Omni 要打补丁或希望拥有单一 import 入口的层、分布式辅助工具和核心类型。检查模块和符号是否仍然存在、重新导出是否解析到同一个来源，以及签名或 dataclass 字段是否未变。位于 `except ImportError` 之后的 import 也要接受同样的检查：MOSS-TTS 的 flash attention import 就是这样被保护的，而当 `sglang.jit_kernel` 变成 `sglang.kernels.ops` 时，它会一声不响地回退到 SDPA。

**借用的类与子类。** `OmniScheduler` 通过 `__getattr__` 查找它没有覆写的上游 `Scheduler` 方法，并以自身作为 `self` 运行它们，同时用上游自己的 kwargs 构建这些方法所期望的调度器组件（`SchedulerDPAttnAdapter`、`SchedulerLoadInquirer`、logprob 处理器、`ParallelState`、`NewTokenRatioTracker`）；`SGLModelRunner` 继承自 `ModelRunner`。对 Omni 覆写的每个方法以及它调用的每个借用方法做函数体 diff，并留意新的上游函数体所读取、而 `OmniScheduler.__init__` 从未赋值的 `self.<attr>`。当构造器增加或删除字段时，传入新的形状；一个按签名过滤 kwargs 或按字段布局分支的辅助函数会让两个版本同时存活。

**vendor 层。** `sglang_omni/vendor/sglang/layers.py` 给 `RMSNorm.forward_cuda` 打补丁，`models.py` 给 `apply_qk_norm` 打补丁；模块 docstring 说明了每个补丁的作用，`models.py` 还说明了移除它的条件。检查被包裹的上游函数体是否仍具有补丁所假定的形状（上游的一次分发重写可以让调用绕过被补丁的方法而不报错），以及 `tests/unit_test/vendor/` 是否仍然固定它。

**`ServerArgs` 变更。** `build_sglang_server_args` 构造该记录并解析一次，因此构建器与发布之间的每个读取者看到的都是解析所决定的结果，而不是原始输入。Omni 在构建器之后通过一个接缝修改引擎配置：`sglang_omni/vendor/sglang/server_args.py::override_server_args`。由上游决定一次变更在每个生命周期阶段的含义；如今，一条已解析但尚未发布的记录把这种修改当作延迟声明，而一条已发布的记录是只读的，其值存放在 runtime-context 包上。每个调用点都有所属阶段，而每个后续读取者都必须从当前发布版本存放该值的地方读取。引入只读记录的那次升级把若干先写后读的位置变成了硬错误。

**兼容覆盖层（compat overlay）。** `sglang_omni/models/dots_tts/compat.py`、`sglang_omni/models/qwen3_tts/compat.py` 和 `sglang_omni/models/qwen3_omni/components/vision_compat.py` 在被固定的第三方包与被固定的技术栈之间架桥，并且各自在其 docstring 中写明了移除条件。对照新技术栈阅读该条件，条件满足时就删除覆盖层。一个新的覆盖层是一个模块，该包的每一次 import 都经过它，并且条件被写在其中。

**上游代码的副本。** Omni 在需要不同形状的地方重新实现了少数几个上游辅助函数，例如 `moss_tts/sampling_kernels.py` 和 `qwen3_tts/sampling_kernels.py` 中的带种子采样变换。副本不 import 任何东西，因此没有任何 import 或签名 diff 会标记它；通过指名上游来源的注释找到它们，并手工对照新 tag 做 diff。当上游改动某个数值细节（一个 clamp、一种累加顺序）时，在每个副本中同步该改动，并为每个副本用一个测试固定边界。

**测试替身与 monkeypatch 目标。** `tests/unit_test/fakes.py` 和测试本地的 fake 建模的是已解析的上游形状。patch 上游名称的测试在该名称消失时会大声失败；而仍然接受上游已删除字段的 fake 不会，于是测试通过而代码不行。

**防御式访问。** 针对被固定发布版本静态定义的状态使用 `getattr`、`hasattr` 和 `except AttributeError`，等于让第二个版本因意外而存活。升级不新增这类代码，并转换它所触及文件中的既有代码。

## 阅读上游增量

SGLang 采用 squash 合并，因此每个上游 PR 都是一个 first-parent 提交，两个 tag 之间的 first-parent 日志就是完整增量。其中大部分触及 Omni 从未接触的文件。把日志范围限定在上面 import 面与子类面背后的文件，阅读剩下的每个提交，记下旧行为、新行为，以及该差异在 Omni 中第一次可见的位置。另外单独检查上一个 tag 携带而新 tag 没有的发布线补丁。

有些上游模块是作为行为而不是作为名称被借用的，无论限定后的日志怎么说，都要当作完整 diff 来阅读：`managers/scheduler.py` 和 `scheduler_components/`、`schedule_batch.py`、`schedule_policy.py`、`model_executor/model_runner.py`、`forward_batch_info.py`、CUDA graph runner、连同 runtime context 的 `server_args.py`、`sampling/`、vendor 模块中指名的那些层、Omni 用以确定 KV 规模的 `mem_cache/`，以及用于发现默认值变化的 `environ.py`。

通过借用的 `Scheduler` 方法抵达 Omni 运行时的变更（准入顺序、树缓存驱逐、从 KV 预算中扣除的内存预留、stop 与 `max_new_tokens` 在同一步骤上相遇时谁获胜）属于上游行为。Omni 不固定也不修补它们；它们作为继承变更进入 PR，并写明用户可见的影响，由 A/B 来测量。

## 技术栈带来的后果，而非 SGLang 的

torch 或 Python 的变动会带来没有任何上游提交描述过的变化。以下是耗费过时间的例子：

- 传递性 pin。torch 固定它自己的 NCCL；某一个 NCCL 发布版本把通信器创建时失败的 NVLS 多播绑定变成了致命错误，而上一个版本只是记录日志并回退，于是在绑定失败的主机上每个 TP>=2 的引擎都会死掉。修复方式是在 TP 工作流中设置一个环境变量并在注释里写明原因，但要找到它，需要的是 torch 发布版本的依赖列表，而不是 SGLang 的。

- 内核 hub。Transformers 从 `kernels` hub 提供某些注意力实现，该 hub 按 torch 和 CUDA 版本构建。当新的版本组合没有构建产物时，模型会无声回退到 eager。对于 Omni 捕获进 CUDA graph 的路径，SGLang 自己的注意力类随 pin 一起发布，没有这个问题。

- 模型包中的 import 守卫。某个包在 import 时检查 torch 与 torchaudio 的版本，拒绝了一个 torchaudio 从未与之匹配的 torch。这正是兼容覆盖层的用途。

- 浮点程序。一次 Transformers 升级改变了一个视觉位置编码插值的中间 dtype 和归约顺序。所有 API 和形状都匹配，而基准测试分数下降了。对预训练模型而言，解释其权重的算术是契约的一部分；覆盖层保留旧的计算序列，并在中间结果上验证，而不仅仅在最终分数上。

- 缓存。新的 Inductor、Triton、FlashInfer 和 DeepGEMM 版本会一次性使所有编译产物失效，因此全新镜像中的第一轮测量的是编译而不是服务。SGLang 在 `SGLANG_CACHE_DIR` 之下构建 DeepGEMM、Triton 及其自己的 JIT 内核；CI setup action 把它指向持久 CI 挂载点上跨 PR 共享的目录，因此升级之后只有第一个任务付出构建成本。

## CI 镜像

GPU CI 运行在 `hongccc/sglang-omni` 之内，在每个工作流中按 digest 固定。CI virtualenv 使用 Python 3.12 加系统 site-packages，并用 `site.addsitedir` 加载 `/opt/sglang/lib/python3.12/site-packages`，包括上游 SGLang 以可编辑安装方式装入的 `.pth` 文件。Torch、FlashInfer 和 SGLang 来自镜像，只有镜像缺少的东西才在其上安装；随后 `verify_omni_installed_pins.py` 会把 `pyproject.toml` 中的每个精确 pin 与实际安装的内容进行核对。

该镜像还安装了去掉冲突依赖的 Qwen-TTS、系统 SoX、Descript DAC 包，以及 Audar/CosyVoice 的附加依赖。解析这些包时应用项目的依赖覆盖。CI import 门禁在兼容补丁之后检查 Qwen-TTS、DAC 与 NeuCodec 的 import，以及 SoX 可执行文件。它会在 Torch 之前 import llama.cpp 以捕获系统 NCCL 冲突；该镜像让 Torch 的 NCCL 库优先。仅凭包清单不能证明运行时可用。

即使重装同一个 FlashInfer wheel，也会刷新捆绑头文件的 mtime。Dockerfile 只对字节完全相同的 FlashInfer 源保留缓存供体的 mtime；发生变化的源仍然更新，并使其对象失效。重建之后、GPU 验证之前，在被复制过来的 `cached_ops` 目录中以 `-n -d explain` 运行 Ninja，以捕获意外的对象重编译。

因此，一次升级要发布一个新镜像：在新 tag 的 `lmsysorg/sglang` digest 上构建 `docker/Dockerfile`，在 GPU 上为 CI 所运行的架构填充 FlashInfer JIT 缓存（Docker 构建没有 GPU，因此 Dockerfile 从上一个镜像复制缓存），推送它，并把新 digest 写入 Dockerfile 和各工作流。在工作流指向新镜像之前，分支上的 CI 毫无意义：在旧镜像上，setup 步骤会把新 torch 装进 virtualenv，之后的一切都不能反映实际发布的技术栈。

## 验证

有两件事可以证明这次升级：新技术栈上的完整单元测试套件，以及在当前 pin 与目标 pin 之间对每个模型所做的 A/B。当 A/B 使某个数字发生移动时，才轮到性能分析出场。

在新镜像内运行完整的单元测试套件，包括默认选择和加速器选择。失败是三种情况之一：测试 patch 了一个已不存在的上游名称、建模了旧形状的 fake，或一次真实的契约破坏。前两种是测试修复；第三种说明上面某个面被漏掉了。

A/B 在同一主机上以相同的数据集和并发，将当前镜像上合并基点处的 `main` 与新镜像上的分支进行比较，并从已预热的服务器开始（全新镜像中的第一轮被丢弃）。对工作流覆盖的家族运行 CI 预设和门禁；对它们未覆盖的家族做一次手动启动，包含一个非流式请求和一个流式请求，分别在 CI 并发和并发 1 下运行——单请求主机成本只有在并发 1 下才会显现。那些消费了别处不消费的东西的家族，例如 diffusion 运行时、dLLM 调度器或权重共享拓扑，最有可能在测试毫无察觉的情况下损坏。

按样本而不是按总分比较精度门禁：在同一臂的两次运行之间翻转的样本是噪声；在两条臂上各自稳定、而在两条臂之间不同的样本才是升级的影响。还要比较启动日志：为启用 CUDA graph 的家族捕获的 CUDA graph、同一组回退警告、KV 池大小，以及从热缓存到就绪的时间。`main` 也过不了的门禁不该由升级来重新调整；它属于一个单独的校准 PR。

当某个数字移动时，请求级事件记录器会指出差异属于哪个 stage，每个 stage 的 torch trace 把内核时间与主机时间分开，而针对 trace 所指向内核的微基准测试给出最终裁决；具体机制见 [profiler.md](./profiler.md)。当同一主机的 A/B 在固定 Omni 提交的情况下复现某个回归时，这个回归才属于这次升级。有一次被归咎于升级的 TTS 回归，二分后定位到分支存在之前三天合入的一个无关 PR；升级只是第一个以 CI 并发运行该配置的变更。

## PR 本身

PR 描述承载的是上游增量而不是 diff：变动的 pin 以及主导本次适配的那一个上游变更、按上文各面分组的修改、用户可观察到的每项继承上游变更的单独清单，以及 A/B——附带两条臂的提交和镜像 digest、样本数，以及对噪声之外任何增量的性能分析归因。测量与推断都按其本来面目标注。

GPU CI 需要 `run-ci` 标签，加上每个家族一个选择器（`run-higgs`、`run-moss`、`run-qwen3-tts`；`run-fun-asr`、`run-qwen3-asr`、`run-whisper-asr`），通过 `/tag-and-rerun-ci <selectors>` 施加。同一家族内的选择器互斥，因此在合并之前，每个预设都会在新镜像上得到自己的一次运行。

合并之后，所有人都要拉取新镜像并重建各自的 virtualenv；基于旧 pin 构建的环境无法运行 `main`。

合并之后必须进行一次完整的 CI 校准。新技术栈会改变吞吐量和延迟，因此工作流中的阈值描述的仍是旧镜像。在合并后的提交上重新校准每个家族，并各自放在独立的 PR 中。
