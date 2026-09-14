SGLang-Omni
=======================

SGLang-Omni 是一个面向 Omni 与多模态模型的高性能推理服务框架，基于 `SGLang <https://github.com/sgl-project/sglang>`_ 构建。它旨在以低延迟编排多阶段流水线，并提供 OpenAI 兼容的 API。

现代 Omni 模型——例如带语音输出的 LLM 与多模态生成系统——可以分解为多个计算特性截然不同的异构阶段：计算密集型的 thinker、访存密集型的 talker，以及对延迟极其敏感的 codec。SGLang-Omni 围绕**以计算为中心的设计**构建：每个阶段运行在独立的调度器上，并针对其瓶颈进行调优；各阶段之间通过共享的 inbox/outbox 抽象通信，并通过零拷贝共享内存传输张量。这避免了任一阶段拖累其他阶段，同时允许新模型只需声明流水线拓扑即可接入框架，而无需从零构建一套推理系统。

关于
-----

核心特性：

- **多阶段流水线**：灵活的框架，可跨进程和 GPU 编排预处理、AR 引擎、codec 与 vocoder 等阶段。
- **原生 SGLang 集成**：为 AR 骨干网络利用 SGLang 的 RadixAttention、连续批处理（continuous batching）以及 CUDA Graph 优化。
- **OpenAI 兼容服务器**：开箱即用的 ``/v1/audio/speech``、``/v1/audio/transcriptions``、``/v1/audio/translations`` 与 ``/v1/chat/completions`` 端点，并支持实时流式输出。
- **广泛的模型支持**：TTS（Higgs、Fish S2-Pro、Voxtral、Qwen3-TTS、MOSS-TTS / Local、Ming-Omni-TTS、dots.tts、ZONOS2）、音乐（MiniMax Music 3）、ASR（Qwen3-ASR、Fun-ASR、ARK-ASR、Whisper、MOSS-Transcribe-Diarize）、Omni（Qwen3-Omni、Ming-Omni）以及 LLaDA2.0-Uni。

支持的模型
----------------

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - 模型
     - 类型
     - 备注
   * - `boson-sglang/higgs-audio-v3-tts-4b-base <https://huggingface.co/boson-sglang/higgs-audio-v3-tts-4b-base>`_
     - TTS
     - 语音克隆、流式输出、支持 100+ 语言
   * - `fishaudio/s2-pro <https://huggingface.co/fishaudio/s2-pro>`_
     - TTS
     - 语音克隆、流式输出
   * - `mistralai/Voxtral-4B-TTS-2603 <https://huggingface.co/mistralai/Voxtral-4B-TTS-2603>`_
     - TTS
     - 具名音色、流式输出、支持 9 种语言
   * - `Qwen/Qwen3-TTS-12Hz-Base <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base>`_
     - TTS
     - 语音克隆、流式输出、支持 10 种语言、0.6B / 1.7B
   * - `OpenMOSS-Team/MOSS-TTS-v1.5 <https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5>`_
     - TTS
     - Delay-pattern 版 MOSS-TTS；语音克隆、流式输出、支持 31 种语言
   * - `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 <https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5>`_
     - TTS
     - Local-transformer 版 MOSS-TTS；48 kHz 立体声、流式输出
   * - `inclusionAI/Ming-omni-tts-16.8B-A3B <https://huggingface.co/inclusionAI/Ming-omni-tts-16.8B-A3B>`_
     - TTS
     - 文本转语音与零样本语音克隆
   * - `dots-studio/dots.tts-mf <https://huggingface.co/dots-studio/dots.tts-mf>`_
     - TTS
     - 48 kHz 连续隐变量 TTS；另有 ``dots.tts-soar`` / ``dots.tts-base``
   * - `Zyphra/zonos2 <https://huggingface.co/Zyphra/zonos2>`_
     - TTS
     - MoE TTS、9 个 DAC codebook、语音克隆
   * - `MiniMaxAI/MiniMax-Music3 <https://huggingface.co/MiniMaxAI/MiniMax-Music3>`_
     - Music
     - 文生音乐；歌词 + 描述 → 32 kHz 立体声歌曲
   * - `Qwen/Qwen3-ASR-1.7B <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>`_
     - ASR
     - 多语言转写，支持 30 种语言提示
   * - `FunAudioLLM/Fun-ASR-Nano-2512-hf <https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf>`_
     - ASR
     - 多语言 Fun-ASR-Nano
   * - `AutoArk-AI/ARK-ASR-3B <https://huggingface.co/AutoArk-AI/ARK-ASR-3B>`_
     - ASR
     - 多语言 ARK-ASR
   * - `OpenMOSS-Team/MOSS-Transcribe-Diarize <https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize>`_
     - ASR
     - 多说话人转写 + 说话人分离 + 时间戳
   * - `openai/whisper-large-v3 <https://huggingface.co/openai/whisper-large-v3>`_
     - ASR
     - 实验性的转写与语音译英路由；参见 `音频翻译支持矩阵 <basic_usage/audio_translations.html>`_
   * - `Qwen/Qwen3-Omni-30B-A3B-Instruct <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct>`_
     - Omni
     - 文本、图像、音频、视频 → 文本 + 音频
   * - `inclusionAI/Ming-flash-omni-2.0 <https://huggingface.co/inclusionAI/Ming-flash-omni-2.0>`_
     - Omni
     - 流式 TTS
   * - `inclusionAI/LLaDA2.0-Uni <https://huggingface.co/inclusionAI/LLaDA2.0-Uni>`_
     - Multimodal
     - 文本 + 图像的理解与生成


.. toctree::
   :maxdepth: 1
   :caption: 快速开始

   get_started/installation.md
   get_started/installation_npu.md
   get_started/installation_xpu.md
   get_started/installation_cpu.md


.. toctree::
   :maxdepth: 1
   :caption: 实战指南

   cookbook/higgs_tts.md
   cookbook/voxtral_tts.md
   cookbook/fishaudio_s2_pro.md
   cookbook/qwen3_tts.md
   cookbook/ming_tts.md
   cookbook/moss_tts.md
   cookbook/moss_tts_local.md
   cookbook/dots_tts.md
   cookbook/minimax_music3.md
   cookbook/zonos2.md
   cookbook/qwen3_asr.md
   cookbook/fun_asr.md
   cookbook/arkasr.md
   cookbook/moss_transcribe_diarize.md
   cookbook/whisper_asr.md
   cookbook/qwen3_omni.md
   cookbook/ming_omni.md
   cookbook/nemotron_voicechat.md
   cookbook/llada2_uni.md
   cookbook/fun_cosyvoice3.md
   cookbook/auk.md


.. toctree::
   :maxdepth: 1
   :caption: 通用用法

   basic_usage/qwen3_omni.md
   basic_usage/audio_translations.md
   basic_usage/tts.md
   basic_usage/process_topology.md
   basic_usage/tts_process_topology.md
   basic_usage/process_topology_migration.md
   basic_usage/omni_router.md
   basic_usage/mps_dp.md


.. toctree::
   :maxdepth: 1
   :caption: 基准测试

   benchmarks/relay.md


.. toctree::
   :maxdepth: 1
   :caption: 开发者参考

   developer_reference/main.md
   developer_reference/apiserver_design.md
   developer_reference/pipeline.md
   developer_reference/config.md
   developer_reference/adding_parameters.md
   developer_reference/communication.md
   developer_reference/reference_encode_service.md
   developer_reference/profiler.md
   developer_reference/qwen3_asr_concurrency_profile.md
   developer_reference/rl_admin_control.md
   developer_reference/bump_version.md
