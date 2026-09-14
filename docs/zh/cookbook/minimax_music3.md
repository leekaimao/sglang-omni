# MiniMax Music 3

[MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) 是 MiniMax 推出的文生音乐模型。给它歌词和风格描述，它会返回 32 kHz 的立体声歌曲音频，人声与伴奏都在其中。

它不是一个"用音乐音色说话"的 TTS 模型。生成分为两个阶段：Qwen3 骨干网络在每个解码步预测一个音频帧，每帧深达 8 层 RVQ codebook——骨干网络自身的 `lm_head` 先输出 `c0`，再由一个四层的 depth decoder 依次走完 `c1..c7`，每个 code 以前一个为条件。每 200 帧交给一个 flow-matching DIT 求解一个 VAE 隐变量，再由 DAC 风格的解码器转成波形。因此那些影响 TTS 请求的旋钮在这里毫无作用——没有参考说话人、没有 `voice`、没有 `temperature`。你能控制的是歌词、描述（caption）、seed 和长度。

| 组件 | 规格 |
|---|---|
| 骨干网络 | Qwen3 decoder（36 层，hidden=4096，GQA 32/8，词表 200k） |
| Depth decoder | 4 层，hidden=4096，16 个注意力头；7 个残差 codebook，每码本 1024 |
| 帧 | 8 个 codebook；`c0` 在骨干网络词表中，`c1..c7` 在 depth decoder 中 |
| AR 引导 | 无分类器引导（CFG），scale 1.5，`c0` 掩码到条件分支的 top 50 |
| 声学阶段 | Flow-matching DIT（36 层，dim=2048，32 个头）+ DAC 解码器，512 倍上采样 |
| 求解器 | Euler，30 步，声学 CFG scale 1.7 |
| 窗口 | 每个声学分块 200 帧，步进 100 帧 |
| 帧率 | 每秒 25 帧 |
| 上下文长度 | 10,240 token |
| 输出 | 32 kHz 立体声 WAV |

## 前置条件

按照[安装指南](../get_started/installation.md)从源码安装 `sglang-omni`。

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install -v -e .   # 非可编辑安装可去掉 -e
```

**单 GPU**（两个阶段放在一起）：

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000
```

**双 GPU**（AR 在第一块设备上，DIT/DAV 在第二块上）：

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000
```

默认开启、无需额外参数的优化：骨干解码 CUDA graph、RVQ depth CUDA graph、编译后的 DIT block、编译后的 DAV 解码器，以及带 seed 的批量采样。

两个阶段都默认开启无分类器引导，且没有开关参数。它会带来什么代价请参见 [引导](guidance) 一节，因为 AR 那一半会改变单个请求的占用量。

## 生成音乐

请求由两个字段承载：`input` 是歌词；`instructions` 是描述风格、配器、节奏与情绪的 caption。两者都必填，也都重要：caption 决定流派与编曲，歌词是被唱出来的内容。

歌词中的结构标签（`[Verse]`、`[Chorus]`、`[Bridge]`、`[Outro]`）会引导编曲，属于提示词契约的一部分。

**标签必须独占一行。** 归一化只保留以标签开头的行中的标签，丢弃该行其余内容，因此写在标签旁边的歌词会被静默丢弃：

```text
"[Verse]\nWalking down the street"   ->   [start] [verse] Walking down the street
"[Verse] Walking down the street"    ->   [start] [verse]
```

第二种写法生成的歌曲会缺失那一行歌词，而且没有任何警告。

### 第一首歌

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Verse]\nSunday morning, quiet and slow\nSunlight through the half-drawn blinds\nCoffee cooling by the window\nNothing on my mind\n[Chorus]\nTake it easy, take it slow\nWe got nowhere else to go",
    "instructions": "A melancholic lo-fi hip-hop track with a mellow piano riff, soft vinyl crackle, and a slow steady drum beat at 85 BPM",
    "seed": 42,
    "max_new_tokens": 750
  }' \
  --output song_1.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964122/song_1.wav" type="audio/wav">
</audio>

`max_new_tokens` 按每秒 25 帧统计音频帧，因此 750 帧**至多** 30 秒。它是上限而非目标：模型自己输出音频结束 token 时就会结束歌曲。上面的请求通常会顶满上限（约 30 秒）。若想给更早的自然结尾留出空间，可以调高上限；响应没有顶满上限是模型自然收尾，而不是被截断。

响应体是 32 kHz 立体声 WAV。

用 Python：

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "MiniMaxAI/MiniMax-Music3",
        "input": "[Verse]\nSunday morning, quiet and slow\nSunlight through the half-drawn blinds\n[Chorus]\nTake it easy, take it slow\nWe got nowhere else to go",
        "instructions": "A melancholic lo-fi hip-hop track with a mellow piano riff at 85 BPM",
        "seed": 42,
        "max_new_tokens": 750,
    },
    timeout=600,
)
resp.raise_for_status()
with open("song_2.wav", "wb") as f:
    f.write(resp.content)
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964204/song_2.wav" type="audio/wav">
</audio>


由于这是标准的 speech 端点，也可以使用 OpenAI 客户端：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

with client.audio.speech.with_streaming_response.create(
    model="MiniMaxAI/MiniMax-Music3",
    voice="default",
    input="[Verse]\nSunday morning, quiet and slow\nSunlight through the half-drawn blinds\n[Chorus]\nTake it easy, take it slow",
    instructions="A melancholic lo-fi hip-hop track with a mellow piano riff at 85 BPM",
    response_format="wav",
    extra_body={"seed": 42, "max_new_tokens": 750},
) as response:
    response.stream_to_file("song_3.wav")
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964231/song_3.wav" type="audio/wav">
</audio>


### 撰写 caption

caption 是你手中最强的控制手段。含糊的 caption 只能得到平庸的编曲；点明乐器、速度与制作质感才能得到你想要的曲目。

```bash
# 流派、配器、速度，以及一条制作说明
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Chorus]\nWe are the fire that never dies\nBurning bright against the sky",
    "instructions": "An energetic arena rock anthem with distorted electric guitars, punchy live drums and a soaring male vocal at 130 BPM, wide stereo image, lightly compressed",
    "seed": 7,
    "max_new_tokens": 750
  }' \
  --output rock_1.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964267/rock_1.wav" type="audio/wav">
</audio>

长的结构化 caption 同样有效。模型训练时使用的描述涵盖整体属性、情绪推进与人声细节，它会用到所有这些信息：

```python
import requests

caption = """Basic Attributes: bpm is 92, key is E minor, Electric Blues / Blues Rock.
Emotional Progression: confident and gritty from the outset, building tension through
call-and-response verses before releasing into an extended lead guitar solo.
Sonics: live and organic, warm mid-range, tube grit, relatively uncompressed so the
drums keep their natural attack.
Vocal: male baritone, gravelly and textured, conversational phrasing in the verses
turning melodic in the refrain."""

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "MiniMaxAI/MiniMax-Music3",
        "input": "[Verse]\nI came up on a dirt road, nothing but a name\n[Chorus]\nAnd the thunder rolls the same",
        "instructions": caption,
        "seed": 11,
        "max_new_tokens": 1500,
    },
    timeout=600,
)
resp.raise_for_status()
with open("blues_1.wav", "wb") as f:
    f.write(resp.content)
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964323/blues_1.wav" type="audio/wav">
</audio>

caption 与歌词在分词前会被确定性地清洗——去除 Markdown 残留，把 `<|tag value|>` 形式改写为 `tag is value`——分词后的提示词上限为 5,000 token。

### 纯器乐与短片段

歌词必填且不能为空，因此请在 caption 里声明器乐曲，并把歌词行压到最简：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Intro]\n(instrumental)",
    "instructions": "An instrumental ambient piece, no vocals: warm analog pads, slow evolving texture, distant piano, 70 BPM",
    "seed": 3,
    "max_new_tokens": 250
  }' \
  --output ambient_1.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30964363/ambient_1.wav" type="audio/wav">
</audio>

250 帧把片段限制在 10 秒内，这是在正式渲染完整长度之前试听 caption 效果最快的方式。短上限通常会被顶满而非提前结束，因此这类片段会按完整时长返回。

### 可复现性与变体

请求对 seed 是确定性的：相同的歌词、caption、seed 与长度会返回逐字节一致的音频。只改 seed 相当于同一首歌的另一次演绎——这是在不动提示词的前提下探索其他版本的途径。

```python
import requests

for seed in (1, 2, 3):
    resp = requests.post(
        "http://localhost:8000/v1/audio/speech",
        json={
            "model": "MiniMaxAI/MiniMax-Music3",
            "input": "[Verse]\nCity lights are calling out my name",
            "instructions": "A dreamy synthwave track with analog pads and a driving bassline at 110 BPM",
            "seed": seed,
            "max_new_tokens": 500,
        },
        timeout=600,
    )
    resp.raise_for_status()
    with open(f"take_{seed}.wav", "wb") as f:
        f.write(resp.content)
```

省略 `seed` 时使用 `0`，仍然是确定性的——它是固定 seed，不是随机 seed。


<audio controls>
  <source src="https://github.com/user-attachments/files/30964419/take_3.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="https://github.com/user-attachments/files/30964420/take_2.wav" type="audio/wav">
</audio>
<audio controls>
  <source src="https://github.com/user-attachments/files/30964418/take_1.wav" type="audio/wav">
</audio>

### 参考输出

五种流派各一条 caption，在单张 H200 上仅使用本页所述默认参数渲染。每个请求都完整给出，粘贴回去即可复现对应片段。

五条中有四条顶满 750 帧上限（30.0 秒）。J-pop 那条在 26.2 秒自行停止——这正是 `max_new_tokens` 作为上限而非目标的体现：模型自己结束了歌曲。

**Lo-fi hip-hop**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "$(cat <<'EOF'
{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Verse]\nWalking down the empty street at midnight\nStreetlights flicker like a broken dream\nI've got nothing but the sound of my own heartbeat\nEchoing through the silent concrete stream\n[Chorus]\nAnd I keep on walking\nTill the morning finds me\nLeave the night behind me",
  "instructions": "A melancholic lo-fi hip-hop track at 85 BPM in F minor: mellow Rhodes piano riff, soft vinyl crackle, dusty boom-bap drums with a laid-back swing, warm upright bass. Intimate bedroom production, gentle tape saturation, no bright cymbals.",
  "seed": 1,
  "max_new_tokens": 750,
  "response_format": "wav",
  "stream": false
}
EOF
)" \
  --output 00_lofi_hiphop.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30963578/00_lofi_hiphop.wav" type="audio/wav">
</audio>

**J-pop**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "$(cat <<'EOF'
{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Verse]\nMorning light is spilling through the curtain\nEvery colour waking up with me\n[Chorus]\nRun into the day and never look back\nEverything we wanted is ahead of us",
  "instructions": "A cheerful J-pop song at 128 BPM in C major: bright acoustic piano, chiming electric guitar, punchy four-on-the-floor drums, and a clear female lead vocal. Polished modern pop production, wide stereo, energetic and uplifting.",
  "seed": 2,
  "max_new_tokens": 750,
  "response_format": "wav",
  "stream": false
}
EOF
)" \
  --output 01_jpop_bright.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30963575/01_jpop_bright.wav" type="audio/wav">
</audio>

**Synthwave，器乐**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "$(cat <<'EOF'
{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Intro]\n(instrumental)\n[Outro]\n(instrumental)",
  "instructions": "A moody synthwave instrumental at 100 BPM in D minor: pulsing analog bass arpeggio, gated reverb drum machine, wide atmospheric pads, and a soaring lead synth melody. Retro 1980s production, heavy chorus effect, cinematic and nocturnal.",
  "seed": 3,
  "max_new_tokens": 750,
  "response_format": "wav",
  "stream": false
}
EOF
)" \
  --output 02_synthwave_moody.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30963577/02_synthwave_moody.wav" type="audio/wav">
</audio>

**原声民谣**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "$(cat <<'EOF'
{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Verse]\nI came up on a dirt road, nothing but a name\nCarried all my summers in a canvas bag\n[Chorus]\nAnd the river keeps on running\nLike it never learned to stay",
  "instructions": "A gentle acoustic folk ballad at 76 BPM in G major: fingerpicked steel-string guitar, soft brushed snare, subtle cello underneath, and a warm male vocal close to the microphone. Sparse and organic, natural room sound, very little compression.",
  "seed": 4,
  "max_new_tokens": 750,
  "response_format": "wav",
  "stream": false
}
EOF
)" \
  --output 03_acoustic_folk.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30963579/03_acoustic_folk.wav" type="audio/wav">
</audio>

**电影感管弦乐**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "$(cat <<'EOF'
{
  "model": "MiniMaxAI/MiniMax-Music3",
  "input": "[Intro]\n(instrumental)\n[Chorus]\nRise above the ashes of the fallen sky\nWe were never meant to say goodbye",
  "instructions": "An epic cinematic orchestral piece at 90 BPM in E minor: sweeping string ostinato, powerful brass swells, timpani and taiko percussion, and a distant choir. Wide concert-hall reverb, dynamic build from restrained to triumphant, no drum kit.",
  "seed": 5,
  "max_new_tokens": 750,
  "response_format": "wav",
  "stream": false
}
EOF
)" \
  --output 04_orchestral_epic.wav
```

<audio controls>
  <source src="https://github.com/user-attachments/files/30963576/04_orchestral_epic.wav" type="audio/wav">
</audio>


(guidance)=
## 引导

两个阶段都运行无分类器引导，且都无法关闭。它没有请求字段，也没有 serve 参数：scale、掩码宽度和求解器自身的 scale 都固定在模型里，因为参考实现就是按这种方式对该 checkpoint 采样的，而去掉引导会让模型对你的 caption 与歌词的遵循程度明显下降。

这两半是恰好同名的两种独立机制：

- **AR 阶段**对每个请求在每个解码步运行两次。一行看得到你的真实提示词；另一行看到的是同一段提示词，但整个 caption 与歌词区间（含分隔符）被覆写为 `<|audio_cfg|>`——只有对话标记和 `<|audio_start|>` 得以保留。随后的每个 code 都从 `uncond + (cond - uncond) * 1.5` 中抽取。对 `c0` 而言，在常规 top-k 采样之前，候选集还会额外被掩码到条件行自身的 top 50，因此带引导的抽取不会漂移到条件分支认为合理的范围之外。
- **声学阶段**在 30 个求解步的每一步内部应用它自己的引导，scale 为 1.7。

你必须为其规划的后果在 AR 侧：**一个请求在引擎中占用两行，而不是一行。**两行在整首歌期间都持有各自的 KV cache，因此一个请求的 KV 开销是无引导时的两倍。随之而来的可见影响见 [并发](concurrency) 一节。

它*不*改变的是请求契约和随机性来源：带引导的抽取仍由同一个按请求、按帧位置的 seed 决定，因此 seed 的行为与上文描述一致。引导确实会改变*抽到哪些* code，因此没有引导的构建产出的音频无法逐次对齐——同一个 seed 下得到的会是不同的编曲和不同的长度，而不是同一首歌的"净化版"。

(concurrency)=
## 并发

服务器持续批处理。准入默认为 16 个并发请求（`max_running_requests=16`），即 **32 个解码行**，因为引导让每个请求多占一行。可在 serve 时通过 `--minimax_music3_ar.engine.max_running_requests` 调高准入；行数、解码 CUDA graph 与 RVQ depth graph 都由它推导而来：

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000 \
  --minimax_music3_ar.engine.max_running_requests 32
```

这样 32 个并发请求会以 64 行提供服务。此处不要传 `engine.cuda_graph_max_bs`：该模型会自行计算上限，使 graph 始终覆盖翻倍后的 batch，你传入的值会被丢弃而不是生效。

请让客户端并行而非串行发送：

```python
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPTS = [
    ("[Verse]\nMorning breaks over the harbour", "A gentle acoustic folk song with fingerpicked guitar at 90 BPM"),
    ("[Chorus]\nDance until the record stops", "A disco house track with four-on-the-floor drums and funk guitar at 122 BPM"),
    ("[Verse]\nSnow falls quiet on the pines", "A cinematic orchestral piece with strings and soft horns at 60 BPM"),
]


def render(index_and_prompt):
    index, (lyrics, caption) = index_and_prompt
    resp = requests.post(
        "http://localhost:8000/v1/audio/speech",
        json={
            "model": "MiniMaxAI/MiniMax-Music3",
            "input": lyrics,
            "instructions": caption,
            "seed": index,
            "max_new_tokens": 750,
        },
        timeout=900,
    )
    resp.raise_for_status()
    with open(f"track_{index}.wav", "wb") as f:
        f.write(resp.content)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(render, enumerate(PROMPTS)))
```

并发是这个模型效率最高的场景。无论 batch 中是一个请求还是八个请求，depth decoder 每步的开销都一样，因此并发越高，分摊到每个请求的成本越低。

这种平坦性也正是引导"时间上便宜、显存上昂贵"的原因。AR 步的瓶颈在于读取骨干网络权重而非算术运算，因此第二行可以搭同一趟便车，额外墙钟时间很少，但每个请求持有的 KV 翻倍。请按行数而非请求数来规划资源池。

## 请求参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 已服务的模型 | 已服务的模型标识符 |
| `input` | string | （必填） | 歌词，不能为空。`[Verse]` / `[Chorus]` / `[Bridge]` / `[Outro]` 需独占一行 |
| `instructions` | string | （必填） | 描述流派、配器、速度、情绪与制作的 caption，不能为空 |
| `seed` | int | `0` | 非负 64 位整数。固定给定请求的输出 |
| `max_new_tokens` | int | `9000` | 音频帧上限，每秒 25 帧，最大 9,000（六分钟）。模型可能提前结束 |
| `response_format` | string | `"wav"` | 输出容器格式 |
| `stream` | bool | `false` | 必须为 `false`；该模型的外部 API 不支持流式 |

顶到上限时返回已生成的音频；以音频结束 token 收尾则提前停止歌曲。

## 模型会拒绝的参数

请求会被直接拒绝而非静默忽略，因此出错会立刻可见：

| 参数 | 原因 |
|---|---|
| `temperature`、`top_p`、`top_k`、`repetition_penalty` | 采样是固定的：先做 scale 1.5 的引导，再做 top-k 50 的按请求 seed 抽取。显式设置会被拒绝 |
| `voice` | 没有可选择的话人；人声来自 caption |
| `ref_audio`、`ref_text`、`language`、`task_type` | 该契约中没有参考音频条件，也没有语言标签 |
| `speed` | 只能为 `1.0`。速度属于 caption，如 "at 92 BPM" |
| `stream: true` | 不支持。该模型的外部 API 不支持流式 |

## 说明

**时长与成本。** 生成时间随 `max_new_tokens` 增长。10 秒片段是迭代 caption 最便宜的方式；风格确定后再渲染完整歌曲。

**提示词敏感性。** 提示词的 token id 与 `<|audio_start|>` 的位置会作为骨干 KV cache 的种子，因此空白与标签改写绝非表面文章——它们会改变音频。若需要可复现的结果，请保持歌词与 caption 逐字节一致。

**显存。** `mem_fraction_static` 默认为 `0.50`，且只为 SGLang 骨干 KV cache 预算；声学阶段的 DIT 与 DAV 权重不占该比例。它是总设备显存的份额，因此在较小的卡上 KV 池会自动缩小。调低它并不可靠——实测 `0.35` 时流水线约慢 6%。请按行数而非请求数做预算：引导意味着每个并发请求要在池中容纳两条序列，因此调高 `max_running_requests` 的 KV 代价是数字所示的两倍。

**注意力后端。** DIT 通过 dit_dav 阶段的 `factory` 组接受 `auto`、`torch_sdpa`、`fa` 与 `sage_attn`；`torch_sdpa` 是默认值，且实测在这些形状下最快。`cache_dit` 是默认关闭的近似选项，以音质换速度；把求解器的 `dit_steps` 降到 30 以下实测会带来可衡量的音质损失，并非免费的提速。
