# Qwen3-Omni

[Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) 是一个多模态模型，
接受文本、图像、音频和视频输入，并可产生仅文本或文本 + 音频输出。本页涵盖所有受支持的
服务器配置 —— 使用生成器获取适合你硬件的精确启动命令，然后查看表格确认你的组合受支持。

## 前置条件

```bash
docker pull hongccc/sglang-omni:dev
docker run -it --shm-size 32g --gpus all hongccc/sglang-omni:dev /bin/zsh
```

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.1.5"
```

Docker 镜像摘要与源码安装请参见[安装](../get_started/installation.md)。

## 服务器配置

使用下面的选择器为你的配置生成精确的启动命令。

```{raw} html
<div id="sgl-server-gen-mount"></div>
```

## 兼容性矩阵

共置式（colocated）拓扑需要 `--config examples/configs/qwen3_omni_colocated_h20.yaml`
（在 H200 上为 `qwen3_omni_colocated_h200.yaml`）来设置各阶段的 GPU 显存预算。

| 模式 | 拓扑 | Thinker TP | 精度 | 状态 |
|---|---|---|---|---|
| 仅 Thinker | — | — | BF16 | ✅ |
| 仅 Thinker | — | — | FP8 | ✅ |
| 仅 Thinker | — | — | AutoRound INT4 | ✅ |
| Thinker-Talker | 分离式 | TP=1 | BF16 | ✅ |
| Thinker-Talker | 分离式 | TP=1 | FP8 | ✅ |
| Thinker-Talker | 分离式 | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |
| Thinker-Talker | 分离式 | TP=2 | BF16 | ✅ |
| Thinker-Talker | 分离式 | TP=2 | FP8 | ✅ |
| Thinker-Talker | 分离式 | TP=2 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |
| Thinker-Talker | 共置式 | TP=1 | BF16 | ✅ |
| Thinker-Talker | 共置式 | TP=1 | FP8 | ✅ |
| Thinker-Talker | 共置式 | TP=1 | AutoRound INT4 thinker + BF16 talker/code2wav | ✅ |

## 输入 / 输出模态

所有输入模态组合在仅文本服务器和语音服务器上均可工作。
`modalities: ["text", "audio"]` 需要**语音模式服务器**（省略 `--text-only`）。

| 输入 | 输出 | 语音服务器 | 最小请求体 | 说明 |
|---|---|---|---|---|
| 文本 | 文本 | 否 | `{"messages": [{"role": "user", "content": "..."}], "modalities": ["text"]}` | — |
| 图像 + 文本 | 文本 | 否 | `{"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text"]}` | — |
| 音频 | 文本 | 否 | `{"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text"]}` | 查询为语音时 content 必须为 "" |
| 图像 + 音频 | 文本 | 否 | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]}` | 查询为语音时 content 必须为 "" |
| 图像 | 文本 | 否 | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text"]}` | 查询来自图像时 content 必须为 "" |
| 视频 + 文本 | 文本 | 否 | `{"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text"]}` | — |
| 视频 + 音频 | 文本 | 否 | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text"]}` | 查询为语音时 content 必须为 "" |
| 视频 | 文本 | 否 | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text"]}` | 查询来自视频时 content 必须为 "" |
| 文本 | 文本 + 音频 | **是** | `{"messages": [{"role": "user", "content": "..."}], "modalities": ["text", "audio"]}` | — |
| 图像 + 文本 | 文本 + 音频 | **是** | `{"messages": [{"role": "user", "content": "..."}], "images": ["path/or/url"], "modalities": ["text", "audio"]}` | — |
| 音频 | 文本 + 音频 | **是** | `{"messages": [{"role": "user", "content": ""}], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | 查询为语音时 content 必须为 "" |
| 图像 + 音频 | 文本 + 音频 | **是** | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | 查询为语音时 content 必须为 "" |
| 图像 | 文本 + 音频 | **是** | `{"messages": [{"role": "user", "content": ""}], "images": ["path/or/url"], "modalities": ["text", "audio"]}` | 查询来自图像时 content 必须为 "" |
| 视频 + 文本 | 文本 + 音频 | **是** | `{"messages": [{"role": "user", "content": "..."}], "videos": ["path/or/url"], "modalities": ["text", "audio"]}` | — |
| 视频 + 音频 | 文本 + 音频 | **是** | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "audios": ["path/or/url"], "modalities": ["text", "audio"]}` | 查询为语音时 content 必须为 "" |
| 视频 | 文本 + 音频 | **是** | `{"messages": [{"role": "user", "content": ""}], "videos": ["path/or/url"], "modalities": ["text", "audio"]}` | 查询来自视频时 content 必须为 "" |

### 采样参数

标准采样参数作用于 thinker 阶段。当 `modalities` 包含 `"audio"` 时，下面额外的 talker
专属参数会独立控制语音生成。

| 参数 | 类型 | 默认值 | 作用于 |
|---|---|---|---|
| `temperature` | float | `1.0` | Thinker |
| `top_p` | float | `1.0` | Thinker |
| `top_k` | int | `-1` | Thinker |
| `min_p` | float | `0.0` | Thinker |
| `repetition_penalty` | float | `1.0` | Thinker |
| `max_tokens` | int | `2048` | Thinker |
| `max_completion_tokens` | int | `null` | Thinker；`max_tokens` 的 OpenAI 兼容别名 |
| `stop` | str \| list | `null` | Thinker |
| `seed` | int | `null` | Thinker |
| `stream` | bool | `false` | 两者 |
| `audio` | dict | `null` | 语音响应格式配置，例如 `{"format": "wav"}` |
| `talker_temperature` | float | `0.9` | Talker（仅音频输出） |
| `talker_top_p` | float | `1.0` | Talker（仅音频输出） |
| `talker_top_k` | int | `50` | Talker（仅音频输出） |
| `talker_repetition_penalty` | float | `1.05` | Talker（仅音频输出） |
| `talker_max_new_tokens` | int | `4096` | Talker（仅音频输出） |
| `stage_sampling` | dict | `null` | 逐阶段采样覆盖 |
| `stage_params` | dict | `null` | 逐阶段非采样参数 |
| `video_fps` | float | `null` | 视频输入的帧采样率（未设置时使用服务器默认值） |
| `video_max_frames` | int | `null` | 从视频中采样的最大帧数 |
| `video_min_pixels` | int | `null` | 每个视频帧的最小像素数 |
| `video_max_pixels` | int | `null` | 每个视频帧的最大像素数 |
| `video_total_pixels` | int | `null` | 所有视频帧的总像素预算 |

### 已知限制

- **`modalities: ["text", "audio"]` 对仅文本服务器无效。** 不会抛出错误 —— 只是响应中不含音频。要获得音频输出，请使用语音模式服务器（不带 `--text-only`）。
- **当查询完全位于 `audios`、`videos` 或 `images` 中时，`content` 必须为 `""`。** 将文本查询留在 `content` 中并与音频并存会导致模型同时处理二者，这通常不是你想要的。
- **共置式拓扑不支持 `--thinker.tp_size 2`。** 服务器在启动时抛出 `ValueError`（"Qwen Phase 1 colocation does not support thinker TP"）。TP=2 请使用分离式（disaggregated）拓扑。
- **超出模型上下文长度的请求会被拒绝并报错。** 当提示 token 数量单独达到或超过 `max_seq_len`，或 `prompt tokens + max_new_tokens ≥ max_seq_len` 时，预处理器会抛出 `ValueError`。请减少输入长度或降低 `max_tokens` 以保持在限制之内。
