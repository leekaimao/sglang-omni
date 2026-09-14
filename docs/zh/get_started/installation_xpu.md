# 🚀 安装 — Intel XPU

为 **Intel GPU（XPU）**安装 `sglang-omni`。默认的[安装说明](./installation.md)固定使用仅限
CUDA 的 wheel，会破坏 `torch+xpu` 软件栈。与上游 SGLang 做法一致（[Intel XPU 文档](https://docs.sglang.io/docs/hardware-platforms/xpu)、
`docker/xpu.Dockerfile`），XPU 路径使用**独立的 `pyproject_xpu.toml`** 加上 PyTorch
XPU wheel 索引。

## 为什么需要独立的 pyproject

`pip install -e .` 会解析 CUDA 版的 [`pyproject.toml`](../../../pyproject.toml)，其中的 torch
系列和仅限 CUDA 的 wheel 会替换掉 `+xpu` 软件栈。
[`pyproject_xpu.toml`](../../../pyproject_xpu.toml) 编码了 XPU 的替换项。

核心依赖覆盖受支持的模型（Qwen3-ASR / TTS / Omni）以及 API 服务器；
`[eval]` 添加 SeedTTS/WER 工具，`[all]` 是它的别名。其他模型系列
（S2-Pro、Ming-Omni、Voxtral-TTS）仅支持 CUDA，不在此提供。

> **必须使用 `--no-build-isolation`** —— 否则 pip 会生成旧式的 in-tree
> `egg-info`，而不是 PEP 660 可编辑安装。安装器始终会传递该参数。
> 正因如此，pip 也不会安装构建依赖，所以该环境自身的
> `setuptools` 必须 **≥ 77.0.0**：更早的版本会以 ``invalid pyproject.toml config: `project.license` ``
> 拒绝 PEP 639 许可证元数据。安装器会在构建前检查这一点；可使用
> `pip install -U 'setuptools>=77.0.0'` 升级。

## 前置条件

- Python ≥ 3.10，以及 Intel GPU 驱动（存在 `/dev/dri/renderD*`）。
- 目标环境中的 `setuptools` ≥ 77.0.0（见上文说明）。
- **PyTorch XPU 软件栈**和 **XPU 版 SGLang 构建** —— 如果已有可正常工作的 `torch+xpu` 环境，可直接复用。关于 oneAPI 的注意事项，请参阅[运行时环境](runtime-environment-important)。

## 🐳 方案 A：Docker

```bash
docker build -f docker/xpu.Dockerfile -t sglang-omni:xpu .
docker run -it --device /dev/dri --shm-size 32g --ipc host --network host sglang-omni:xpu
```

基于 Intel Deep Learning Essentials 和 `+xpu` torch wheel 构建。它有意**不**
source oneAPI —— 参见[运行时环境](runtime-environment-important)。

固定 SGLang 版本并不会固定 SYCL 内核：其 XPU manifest 要求从 git 安装不带修订号的
`sgl-kernel-xpu`。因此 Dockerfile 自行固定了该提交，默认情况下重新构建是可复现的。
仅在有明确意图时才覆盖它：

```bash
docker build -f docker/xpu.Dockerfile \
  --build-arg SGL_KERNEL_XPU_REF=<sgl-kernel-xpu commit sha> \
  -t sglang-omni:xpu .
```

## 🛠️ 方案 B：安装到现有的 XPU 环境（此处推荐）

该辅助脚本会换入 `pyproject_xpu.toml`，使用 XPU 索引安装，然后再恢复 CUDA 版本：

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

# dry-run first — shows the commands, installs nothing
PYTHON=$(which python) scripts/xpu/install_xpu.sh --check

# editable install against the PyTorch XPU index
PYTHON=$(which python) scripts/xpu/install_xpu.sh
```

使用 `--extras` 选择 extras（逗号分隔）：

```bash
scripts/xpu/install_xpu.sh --extras eval           # core + SeedTTS/WER eval + tests
scripts/xpu/install_xpu.sh --extras all            # alias for eval
```

也可以手动完成（脚本自动执行的步骤相同）：

```bash
cp pyproject.toml .pyproject.cuda.bak
cp pyproject_xpu.toml pyproject.toml
pip install -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/xpu
cp -f .pyproject.cuda.bak pyproject.toml && rm .pyproject.cuda.bak   # restore CUDA pyproject
```

### SGLang（单独安装）

`sglang` 有意**不**固定版本，因此上述安装不会影响已有的 XPU 构建。
即使以版本区间的形式也无法固定：每个已发布的 wheel 都要求 `flashinfer_python[cu13]` 和
`nvidia-*` 运行时，因此**任何**版本说明符都会把 CUDA 软件栈拉到 `torch+xpu` 之上。请从源码构建：

```bash
git clone https://github.com/sgl-project/sglang && cd sglang
git checkout v0.5.19   # the pinned release
cd python && cp pyproject_xpu.toml pyproject.toml
pip install -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/xpu
```

请使用该提交：XPU 移植面向此 SGLang 版本的 API，不携带版本兼容垫片。VCS 依赖（`pip install "sglang @ git+…"`）**不**起作用：
pip 会读取 checkout 中的 `python/pyproject.toml`，其中固定的是 CUDA 版 torch；只有上面的替换操作才能选中
`+xpu`。

## 验证

```bash
# import works from anywhere now (package installed, not just cwd-on-path)
python -c "import sglang_omni, torch; print(sglang_omni.__file__, torch.__version__)"
which sgl-omni

# device-layer unit tests (CPU, no GPU) — needs pytest, which ships in the
# `[eval]` extra (install with `.[eval]`, or `pip install pytest` first)
pytest tests/unit_test/xpu/test_device_layer.py -v
```

## 启动服务

(runtime-environment-important)=
### 运行时环境（重要）

请直接在 **PyTorch-XPU 环境**中运行 —— **不要** `source /opt/intel/oneapi/setvars.sh`。
`+xpu` wheel 自带 oneCCL/SYCL/Level-Zero；系统 oneAPI 会将不同的 oneCCL/UCX
放到库路径上，与自带的 `libccl` 冲突，导致多 XPU 的 `xccl` 集合通信崩溃。

不需要额外的环境变量 —— XPU 后端会被自动检测。如果 Triton JIT
构建报告 `fatal error: sycl/sycl.hpp: No such file or directory`，请让编译器指向
`intel-sycl-rt` wheel 的头文件：
```bash
export CPATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
```

### Qwen3-ASR（语音转文本，单 XPU）

```bash
sgl-omni serve --model-path /path/to/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000
# transcribe:
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" -F "model=/path/to/Qwen3-ASR-1.7B"
```

### Qwen3-TTS（文本转语音，单 XPU）

Qwen3-TTS 需要上游的 `qwen-tts` 包。方案 A 已包含它；方案
B 需在此处安装，因为 `pyproject_xpu.toml` 有意不固定它。
两行命令都必须加 `--no-deps`：`qwen-tts` 固定了 Transformers 4.57.3，
会替换本项目的 5.12.1，而解析 `sox` 依赖会把 `numpy` 提升到超过
`numba==0.65.1` 的上限。参见
[docs/cookbook/qwen3_tts.md](../cookbook/qwen3_tts.md)。

```bash
apt-get update && apt-get install -y sox   # the Python sox package shells out to it
pip install --no-deps sox einops
pip install --no-deps qwen-tts==0.1.1
```

```bash
sgl-omni serve --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base --host 0.0.0.0 --port 8000
# Base checkpoint clones a reference voice — pass ref_audio (+ ref_text):
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/Qwen3-TTS-12Hz-1.7B-Base","input":"Hello from Intel XPU.",
       "voice":"default","ref_audio":"/path/to/ref.wav","ref_text":"reference transcript",
       "response_format":"wav"}' -o out.wav
```

### Qwen3-Omni（30B-A3B MoE，多 XPU 张量并行）

30B MoE 无法装入单张 24 GB 显卡；请使用张量并行将 Thinker 分片到多块 GPU 上。
`--text-only` 只服务 Thinker（聊天），不包含 Talker/语音阶段。纯文本配置通常会把每个阶段都放在
`pipeline` 进程中，因此在启用 TP 之前，请为 TP Thinker 指定一个原本未被占用的进程名：

```bash
# thinker across 8 cards (TP=8). Large shards over shared storage load slowly, so give
# startup more headroom than the default 600 s.
export SGLANG_OMNI_STARTUP_TIMEOUT=1800
sgl-omni serve --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --text-only --thinker.process thinker \
  --thinker.tp_size 8 --thinker.gpu "[0, 1, 2, 3, 4, 5, 6, 7]" \
  --host 0.0.0.0 --port 8000
# chat:
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/Qwen3-Omni-30B-A3B-Instruct",
       "messages":[{"role":"user","content":"What is Intel XPU?"}],"max_tokens":64}'
```

对上述任意服务进行健康检查：`curl http://localhost:8000/v1/models`。

> **XPU 上的预期现象：** `Failed to import mooncake` / `Failed to import nixl` 警告无害
> —— 这些仅限 CUDA 的传输后端已被省略；张量改经 `shm` 中继传输。

> ✅ 支持状态：**Qwen3-ASR、Qwen3-TTS 和 Qwen3-Omni 均可在 Intel XPU 上端到端地提供服务**
> （ASR 单卡、TTS 单卡、Qwen3-Omni 的 Thinker 通过张量并行分布到 8 张卡上）。
