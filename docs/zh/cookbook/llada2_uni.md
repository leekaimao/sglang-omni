# LLaDA2.0-Uni

[LLaDA2.0-Uni](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) 是一个接受文本与图像输入的多模态模型。本 SGLang-Omni 实战指南覆盖其实验性的文本输出服务路径。

## 亮点

- 统一的 dLLM-MoE 骨干网络 —— 基于 LLaDA 2.0 构建，统一多模态理解与生成。
- 顶尖的理解与生成能力 —— 在视觉问答与文档理解上匹敌专用 VLM，同时生成高质量图像。
- 交错生成与推理 —— 依托统一的离散表示，解锁交错生成与推理。

## 架构

![LLaDA2.0-Uni Architecture](../_static/image/llada2.0_uni_architecture.png)

LLaDA2.0-Uni 把多模态理解与生成统一到一个简单的掩码 token 预测（Mask Token Prediction）范式中。视觉输入由 SigLIP-VQ 分词器编码为离散语义 token，然后与文本 token 一起在统一的掩码预测目标下映射到骨干隐藏状态。输出的 token 经文本去分词器（Text De-Tokenizer）解码回文本，或经 Diffusion Decoder 重建为高保真图像。依托统一的离散表示，它能轻松处理复杂的交错生成，并解锁高级的交错推理：通过交错 `<|image|>...<|/image|>` 块，在单一连贯的框架内实现端到端的训练与推理。

## 前置条件

按照[安装指南](../get_started/installation.md)安装 `sglang-omni`。

## 服务器配置

LLaDA2.0-Uni 在单块 GPU 上运行 4 阶段流水线
（`preprocessing → image_encoder → thinker → decode`）。在这条实验性的
DLLM 路径上，thinker 默认禁用 CUDA graph。

```bash
sgl-omni serve --model-path inclusionAI/LLaDA2.0-Uni --port 8000
```

## 文本输入

发送纯文本提示词，获得文本响应。

**cURL**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-Uni",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "inclusionAI/LLaDA2.0-Uni",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 256,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

## 图像与文本输入

发送一张图像加文本提示词，获得文本响应。

**cURL**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-Uni",
    "messages": [{"role": "user", "content": "Briefly describe the cars in this image."}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "inclusionAI/LLaDA2.0-Uni",
        "messages": [{"role": "user", "content": "Briefly describe the cars in this image."}],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

也可以使用 OpenAI 的多内容格式内联传入图像：

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "inclusionAI/LLaDA2.0-Uni",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "tests/data/cars.jpg"}},
                    {"type": "text", "text": "Briefly describe the cars in this image."},
                ],
            }
        ],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

## 请求参数

下表列出 LLaDA2.0-Uni 的 `/v1/chat/completions` 端点接受的所有参数。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | `null` | 模型标识符 |
| `messages` | list | （必填） | chat 消息列表，每条含 `role` 与 `content` |
| `modalities` | list | `["text"]` | 输出模态（只支持 `["text"]`） |
| `images` | list | `null` | 图像文件路径列表（本地路径或 URL） |
| `max_tokens` | int | `null` | 生成的 token 上限 |

### 即将支持

- 文生图
- 带思考的文生图
- 交错生成

## 已知限制

- 支持文本与图像输入的文本输出。图像生成与交错生成尚未接入 OpenAI 兼容的响应
  路径。
