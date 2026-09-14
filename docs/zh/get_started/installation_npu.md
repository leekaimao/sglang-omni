# 安装 — 昇腾 NPU

在安装 `sglang-omni` 之前，请先安装昇腾软件栈和 NPU 版本的 SGLang。辅助脚本
[`install_npu.sh`](../../../scripts/npu/install_npu.sh) 只安装
`sglang-omni`，不会安装或修改下表中的任何前置依赖。

## 前置条件

请按照各链接文档，为你的昇腾硬件选择相互兼容的版本。已验证的配置为 Python 3.11。

| 组件 | 版本 | 是否必需 | 需手动安装 | 安装说明 |
|-----------|---------|----------|---------------------|--------------|
| CANN toolkit | 兼容版本 | 是 | 是 | [官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0008.html) |
| HDK（驱动与固件） | 与硬件及 CANN 版本匹配 | 是 | 是 | [官方文档](https://www.hiascend.com/hardware/firmware-drivers/community) |
| PyTorch 与 `torch_npu` | 相互匹配的版本 | 是 | 是 | [官方文档](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=177&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C) |
| `triton-ascend` | 与所选 PyTorch 和 CANN 版本匹配 | 是 | 是 | [官方文档](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/en/quick_start.md) |
| `sgl-kernel-npu` | 与 PyTorch、Python、CANN、硬件及架构匹配 | 是 | 是 | [官方文档](https://github.com/sgl-project/sgl-kernel-npu/releases) |
| `memfabric-hybrid` | 兼容版本 | 否（仅 PD 分离部署需要） | 是 | [官方文档](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu) |
| NPU 版 SGLang | `v0.5.18` | 是 | 是 | [官方文档](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu) |

## 安装 sglang-omni

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查环境并显示安装命令，不修改任何文件。
bash scripts/npu/install_npu.sh --check

# 以可编辑模式安装 sglang-omni。
bash scripts/npu/install_npu.sh
```

预检（precheck）接受 SGLang `0.5.18` 发布线，包括数字版本段以 `0.5.18` 开头的开发版、预发布版、后发布版和本地构建，例如 `0.5.18.dev7+g<git-sha>`。其他发布线（包括更新的版本）会被拒绝。版本不匹配时，它会同时报告支持的发布线和已安装的版本。预检还会校验必需的 Python 包、`torch` 与 `torch_npu` 的大版本-小版本匹配、NPU 可用性，以及一次小型 NPU 矩阵乘法。运行 `bash scripts/npu/install_npu.sh --help` 可查看可选附加组件、非可编辑安装，以及构建期间有意不暴露设备的环境的处理方式。

## 为 TTS 模型安装 Torcodec

TTS 类模型使用 **`torchcodec`** 实现高效的原生流式音频解码，直接解码为 PyTorch 张量。

辅助脚本 `scripts/npu/install_npu_torchcodec.sh` 会自动安装：
* **音频编解码器：** `ffmpeg`
* **CANN 9.1.0 栈：** `toolkit`、`A3-ops`、`nnal`
* **PyTorch 2.11 栈：** `torch`、`torchvision`、`torchaudio`、`torch_npu`、`torchcodec`

运行安装：

```bash
# 第一个参数指定设备类型（910b 或 A3）
bash scripts/npu/install_npu_torchcodec.sh A3
```
