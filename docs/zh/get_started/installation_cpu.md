# 🚀 安装 — Intel CPU

为**纯 CPU 推理**安装 `sglang-omni`。默认的[安装说明](./installation.md)面向
CUDA，因此本路径使用独立的 [`pyproject_cpu.toml`](../../../pyproject_cpu.toml) 和 PyTorch CPU wheel 索引。

> 可编辑安装 `sglang-omni` 时**必须使用 `--no-build-isolation`**。

## 为什么需要独立的 pyproject

`pip install -e .` 会解析 [`pyproject.toml`](../../../pyproject.toml)。在面向 CUDA 的
checkout 中，这可能拉取仅限 CUDA 的 wheel 并替换掉 CPU 版 torch 软件栈。
[`pyproject_cpu.toml`](../../../pyproject_cpu.toml) 将 torch 系列固定为 CPU wheel，
并省略仅用于加速器的包。

## 前置条件

- Python >= 3.10,<3.13
- `uv`
- 来自对应上游发布版本的 SGLang CPU 构建。
- 标准音频运行时库，例如 `ffmpeg` 和 `libsndfile`。

## 🐳 方案 A：Docker

```bash
# Clone the SGLang-omni repository
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni

# Build the docker image
docker build -f docker/cpu.Dockerfile -t sglang-omni:cpu .

# Initiate a docker container
docker run -it --shm-size 32g --ipc host --network host sglang-omni:cpu
```

该镜像使用上游 SGLang 的 CPU pyproject 安装 SGLang，然后使用 `pyproject_cpu.toml` 安装
`sglang-omni`。它为运行时设置了 `SGLANG_USE_CPU_ENGINE=1`。

## 🛠️ 方案 B：手动安装

先创建并激活环境：

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
OMNI_DIR="$(pwd)"

uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install --upgrade pip "packaging>=24.2" "setuptools>=77.0.0" wheel
```

安装对应的 CPU 版 SGLang 构建：

```bash
git clone https://github.com/sgl-project/sglang ../sglang
cd ../sglang
git checkout v0.5.19

cd python
cp pyproject_cpu.toml pyproject.toml
uv pip install -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cpu

cd sglang/kernels/aot
cp pyproject_cpu.toml pyproject.toml
uv pip install -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cpu
```

使用 CPU pyproject 安装 `sglang-omni`：

```bash
cd "$OMNI_DIR"
bash scripts/cpu/install_cpu.sh
```

## 验证

```bash
python -c "import sglang_omni, torch; print(sglang_omni.__file__, torch.__version__)"
which sgl-omni
```

torch 版本应解析为 CPU 构建。CPU 专属的单元测试位于同一个目录中，因此
CI（以及你自己）可以选择它们而不会触及加速器测试套件：

```bash
SGLANG_USE_CPU_ENGINE=1 pytest tests/unit_test/cpu -v
```

> **运行时必须设置 `SGLANG_USE_CPU_ENGINE=1`。** 否则平台层会报告 `device_type == "cpu"`，
> 而 `is_cpu()` 仍为 `False`，导致按平台分支判断的代码悄悄走加速器路径。

### 音频解码因 `libtorchcodec_core*.so` 而失败

`utils/audio.py` 通过 `torchcodec` 进行解码，后者在导入时会加载
FFmpeg 的共享库。不受支持的 FFmpeg 版本可能表现为缺乏信息量的
`Could not load this library` 错误。

`torchcodec` 仅自带适用于 FFmpeg 主版本 4–8 的加载器；FFmpeg 9 不满足其中任何一个。
如有需要，请固定到较早的主版本。Docker 用户会通过 `apt` 获得受支持的主版本。
