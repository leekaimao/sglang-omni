# 🚀 安装

当前稳定版本：**v0.1.5**，发布于 [PyPI](https://pypi.org/project/sglang-omni/)。

请根据你的平台选择安装路径。NVIDIA CUDA 推荐 Docker —— UCX、flash-attn、SGLang 和 CUDA
均已完成预构建。Apple Silicon 在下方有专门的源码安装器。

> **Intel GPU（XPU）？** 对于 Intel Arc GPU，请参阅 [安装 — Intel XPU](./installation_xpu.md)，该页面使用 [`pyproject_xpu.toml`](../../../pyproject_xpu.toml) + PyTorch XPU wheel 索引，而不是下文仅针对 CUDA 的固定版本。

> **Intel CPU？** 同样不是本页。请参阅 [安装 — Intel CPU](./installation_cpu.md)，该页面使用 [`pyproject_cpu.toml`](../../../pyproject_cpu.toml) + PyTorch CPU wheel 索引。

> **昇腾 NPU？** 请参阅 [安装 — 昇腾 NPU](./installation_npu.md)，了解支持的软件栈、前置条件和安装辅助工具。

## 🐳 方案 A：Docker（推荐）

**1. 拉取镜像**

```bash
docker pull hongccc/sglang-omni:dev
```

目前仅发布了 `dev` 标签。它会随 main 分支一起移动 —— 如需可复现的运行，请按 digest 固定：

```bash
docker pull lmsysorg/sglang-omni@sha256:<digest>
```

**2. 运行容器**

```bash
docker run -it \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    hongccc/sglang-omni:dev \
    /bin/zsh
```

**3. 在容器内安装 `sglang-omni`**

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.1.5"
```

<a id="macos-apple-silicon"></a>

## 🍎 方案 B：macOS Apple Silicon 安装器

```bash
git clone https://github.com/sgl-project/sglang-omni.git && cd sglang-omni
./install.sh
source .venv-apple/bin/activate
```

该脚本是幂等的：创建（或复用）`.venv-apple`，安装 Homebrew formula `ffmpeg@7` 和 `uv`（仅当尚无可用的
git 时才安装 `git`），从源码安装 SGLang `v0.5.19` 及其 `all_mps`
extra，并使用 `uv pip` 安装当前 checkout。SGLang 的可选 Rust
扩展在此 Apple Silicon 路径中并不需要，因此会被跳过。
选用 `ffmpeg@7` 是有意为之：`torchcodec==0.15.0` 仅自带 FFmpeg 4 至 8
的加载器，而不带版本号的 formula 会安装 FFmpeg 9。运行时请暴露其库：

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

运行脚本前必须已安装 Homebrew。如果缺少 `brew`，脚本会打印错误并退出；请自行从
[brew.sh](https://brew.sh) 安装，然后重新运行。该安装器绝不会调用 `sudo` 或
Homebrew 的引导程序。在 CI 中可使用 `--non-interactive`（或 `NONINTERACTIVE=1`）禁用
Homebrew 自动更新，使用 `SGLANG_OMNI_VENV=/path/to/venv` 选择虚拟环境，并使用
`SGLANG_OMNI_EXTRAS=audar-tts,fun-cosyvoice3` 启用可选 extras。持久化的 SGLang 源码
checkout 默认位于 `~/.cache/sglang-omni/sglang-v0.5.19`，可通过
`SGLANG_SOURCE_DIR` 更改。缓慢或使用代理的网络可通过 `UV_HTTP_TIMEOUT` 和
`UV_HTTP_RETRIES` 覆盖安装器的 uv 默认设置。

此路径目前仅支持 `arm64` 架构上的 macOS 14 及以上版本（固定版本的 `torch==2.13.0`、
`torchvision==0.28.0` 和 `torchcodec==0.15.0` wheel 是为 `macosx_14_0_arm64` 构建的），面向
Apple-Silicon 上的 Qwen3-ASR MLX/Torch-MPS 路径。其他平台应使用下方的
Docker、手动或 Intel XPU 说明。常见故障包括：`PATH` 上缺少 Homebrew/uv、Python 3.12
工具链不可用，或启动音频服务器时忘记导出 `DYLD_LIBRARY_PATH`。

### 从托管安装器运行

该脚本也支持下载后运行或 `curl | bash` 的调用方式：当它不在 sglang-omni checkout
内部时，会将由 `SGLANG_OMNI_REPO` 和 `SGLANG_OMNI_REF` 指定的仓库克隆到缓存中，并安装该
checkout。建议先下载并审查固定版本的脚本，然后再运行：

```bash
curl -fsSLo /tmp/sglang-omni-install.sh \
  https://raw.githubusercontent.com/sgl-project/sglang-omni/<commit>/install.sh
less /tmp/sglang-omni-install.sh
chmod +x /tmp/sglang-omni-install.sh
SGLANG_OMNI_REF=<commit> /tmp/sglang-omni-install.sh
```

将远程脚本直接通过管道传给 Bash 会在没有审查步骤的情况下执行代码；
仅在接受这种权衡时才使用：

```bash
curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang-omni/<commit>/install.sh \
  | SGLANG_OMNI_REF=<commit> bash
```

对于 fork 或内部镜像，请显式设置 `SGLANG_OMNI_REPO` 和
`SGLANG_OMNI_REF`。托管模式默认将项目 checkout 存储在
`~/.cache/sglang-omni/sglang-omni-<ref>`；可通过
`SGLANG_OMNI_PROJECT_DIR` 覆盖。

## 🛠️ 方案 C：手动安装

请先构建前置依赖：

- **UCX 1.20.x**（含 CUDA + verbs）—— 见[上游仓库](https://github.com/openucx/ucx)，或复用 [`docker/Dockerfile`](../../../docker/Dockerfile) 中的构建标志。
- **flash-attn-4** `>=4.0.0b18`，需与 `torch==2.13.0` 以及 SGLang 0.5.19 对 `nvidia-cutlass-dsl` 4.6.2 的固定版本相匹配。

然后：

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.1.5"
```

若不固定版本，安装索引上的最新版：`uv pip install --prerelease=allow sglang-omni`。

### 从源码安装

适用于开发或尚未发布的更改：

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow -v -e .   # drop -e for a non-editable install
```
