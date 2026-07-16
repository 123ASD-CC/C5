#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║             C5 文本转语音 (TTS) 模块                          ║
║   Fun-CosyVoice3-0.5B + 音色克隆 + 双鱼支持                  ║
╚══════════════════════════════════════════════════════════════╝

【模块定位】
  在整个多模态翻译系统中，C5 是最后一环（输出端）：
    源语音 → C3/C4(翻译) → 文本 → C5(TTS) → 合成语音
  C5 读取 C3 或 C4 的翻译结果，用原始说话人的声音把译文读出来。

【核心能力】
  1. 中文 TTS：instruct2 模式，零样本音色克隆
  2. 英文 TTS：cross_lingual 模式，<|en|> 语言标签
  3. 批量生成：JSON 驱动，支持 --limit 控制数量
  4. 推理耗时统计：每条记录 latency / RTF / audio_duration

【接口约定（与 C3/C4 对齐）】
  输入：JSON 数组，每条包含 id / audio / 翻译文本字段
  输出：wavs/<id>.wav（以输入 id 命名，可追溯）
        c5_results.json（统计汇总）

【运行示例】
  python c5_tts.py --input ../c4/outputs/c4_results.json --limit 20
  python c5_tts.py --input ../c3/outputs/cascade_results.json --text_field translation
"""

import os
import sys
import json
import time
import argparse
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional

# ═══════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════

# CosyVoice3 源码路径（推理框架，不是模型权重）
# CosyVoice3/ 目录下是阿里开源的 Python 推理代码
# Matcha-TTS 是 Flow 模块的依赖（梅尔频谱生成）
COSYVOICE_REPO = "/root/siton-tmp/multimodal/CosyVoice3"
sys.path.insert(0, COSYVOICE_REPO)
sys.path.insert(0, os.path.join(COSYVOICE_REPO, "third_party", "Matcha-TTS"))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel  # cosyvoice 包的唯一入口

# 项目路径
PROJECT_ROOT = "/root/siton-tmp/multimodal"
MODEL_DIR = os.path.join(PROJECT_ROOT, "Fun-CosyVoice3-0.5B")  # 预训练权重
ASSET_DIR = os.path.join(PROJECT_ROOT, "C5_TTS", "asset")       # 默认音色参考音频

# 默认的音色参考音频（当源音频不可用时的兜底方案）
# 这段音频是一段 2.15 秒的中文女声，campplus 从中提取 512 维 speaker embedding
DEFAULT_PROMPT_WAV = os.path.join(ASSET_DIR, "zero_shot_prompt.wav")
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"  # 与 prompt_wav 内容匹配

# ═══════════════════════════════════════════════════════════════
# CosyVoice3 特殊标记
# ═══════════════════════════════════════════════════════════════

# <|endofprompt|> 是 CosyVoice3 训练时定义的句子分隔符（token_id=151646）
# 模型内部的 llm_job() 函数有 assert 检查，没有这个标记会直接报错：
#   assert 151646 in text, '<|endofprompt|> not detected!'
# 所以所有传给模型的文本都必须包含这个标记
END_OF_PROMPT = "<|endofprompt|>"

# ═══════════════════════════════════════════════════════════════
# 超参数
# ═══════════════════════════════════════════════════════════════

TRAILING_SILENCE_SEC = 1.0  # 末尾静音时长（秒）
# 为什么加静音？
# instruct2 有时在句子末尾提前终止，最后一个字的尾音没发完
# 拼接 1 秒静音让音频不会突然截断，听感上更自然

MIN_CHARS = 3  # 最短字符数，低于此值跳过（卷积核最小尺寸要求）

# 双语测试的固定文本
ZH_TEST_TEXT = "今天天气很好，适合出门散步。"
EN_TEST_TEXT = "The weather is nice today, perfect for a walk."


# ═══════════════════════════════════════════════════════════════
# 日志工具
# ═══════════════════════════════════════════════════════════════

def setup_logging(log_dir: str) -> logging.Logger:
    """
    同时输出到文件和控制台。
    日志文件：{log_dir}/c5_run.log，每次运行追加写入。
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("C5_TTS")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 文件输出
    fh = logging.FileHandler(os.path.join(log_dir, "c5_run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    # 控制台输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ═══════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════

def load_model(model_dir: str, logger: logging.Logger):
    """
    加载 Fun-CosyVoice3-0.5B 预训练模型。

    AutoModel 内部流程：
    1. 读取 config.json → 确定模型类型（CosyVoice3Model）
    2. 加载三个子模型：
       - llm.pt       → 大语言模型（文本 → 语音 token）
       - flow.pt      → Flow Matching（语音 token → 梅尔频谱）
       - hift.pt      → HiFiGAN 声码器（梅尔频谱 → 波形）
    3. 加载 campplus.onnx  → 说话人特征提取器（音频 → 512 维向量）
    4. 加载 speech_tokenizer → 语音离散化编解码

    加载耗时约 60 秒，GPU 显存占用约 4GB。
    """
    logger.info(f"Loading model from {model_dir} ...")
    t0 = time.time()
    model = AutoModel(model_dir=model_dir)
    logger.info(f"Model loaded in {time.time()-t0:.1f}s, sample_rate={model.sample_rate}")
    return model


# ═══════════════════════════════════════════════════════════════
# 输入解析
# ═══════════════════════════════════════════════════════════════

def load_input(json_path: str, logger: logging.Logger) -> List[Dict]:
    """
    读取 C3 或 C4 的 JSON 输出文件。

    兼容两种格式：
    - 数组格式（C4）：[{id:..., audio:..., end2end_translation:...}, ...]
    - 字典格式（C3）：{"results": [{...}, ...]}

    接口字段：
    - C4 用 "end2end_translation"，C3 用 "translation"
    - 通过 --text_field 参数指定，默认 "end2end_translation"
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # C3 格式：{"results": [...], "summary": {...}}
        data = data.get("results", [data])
    elif not isinstance(data, list):
        raise ValueError("Input JSON must be a list or dict with 'results' key")

    logger.info(f"Loaded {len(data)} samples from {json_path}")
    return data


# ═══════════════════════════════════════════════════════════════
# 音频后处理
# ═══════════════════════════════════════════════════════════════

def add_trailing_silence(
    speech: torch.Tensor, sample_rate: int, silence_sec: float
) -> torch.Tensor:
    """
    在语音末尾拼接一段静音。

    原理：
    - 采样率 24000 Hz → 1 秒 = 24000 个采样点
    - torch.zeros(1, 24000) 创建一个全零张量（即静音）
    - torch.cat 沿时间轴拼接

    为什么需要：
    instruct2 在句末有时会提前输出 EOS（结束符），导致最后一个字的
    尾音没有完整发出。加 1 秒静音让音频不会突然截断，听感更自然。
    """
    silence_samples = int(sample_rate * silence_sec)
    silence = torch.zeros(
        speech.shape[0],        # 通道数（通常是 1，单声道）
        silence_samples,        # 静音长度（采样点数）
        dtype=speech.dtype,     # float32
        device=speech.device    # CPU 或 CUDA
    )
    return torch.cat([speech, silence], dim=-1)  # dim=-1 沿最后一维拼接


# ═══════════════════════════════════════════════════════════════
# 语言检测
# ═══════════════════════════════════════════════════════════════

def is_english_text(text: str) -> bool:
    """
    简单启发式判断文本是否为英文。
    统计 ASCII 字母占全部字符的比例，超过 50% 判为英文。

    为什么不直接用 item['target_lang']？
    因为 C4 数据的 target_lang 都是 "zh"（中文目标），
    但文本内容本身可能是中英混合的。所以用文本特征判断更可靠。
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
    return ascii_chars > len(text) * 0.5


# ═══════════════════════════════════════════════════════════════
# 合成引擎 — 中文音色克隆（核心）
# ═══════════════════════════════════════════════════════════════

def synthesize_voice_clone(
    model, text: str, prompt_wav: str, logger: logging.Logger, sr: int,
    speed: float = 1.0
) -> Optional[torch.Tensor]:
    """
    使用 instruct2 模式实现零样本音色克隆。

    【原理】
    inference_instruct2 是 CosyVoice3 最灵活的推理接口，它接受：
      - tts_text：要合成的文本（必须包含 <|endofprompt|>）
      - instruct_text：风格指令（我们传 "" 避免被读出来）
      - prompt_wav：参考音频（从中提取说话人特征）

    内部过程：
      源音频 ──→ [campplus.onnx] ──→ 512维 speaker embedding
                                             │
      中文文本 ──→ [LLM] ──→ 语音 token ──┼──→ [Flow] ──→ [HiFiGAN] ──→ .wav

    【为什么 instruct_text="" ？】
    如果写 "请用自然的语气朗读"，模型会把这句话也合成到音频开头，
    导致前几秒是废话。空字符串只做音色克隆，不添加额外指令。

    【为什么用 instruct2 而不是 zero_shot ？】
    zero_shot 要求 prompt_text 和 prompt_wav 内容严格匹配，
    且 prompt 长度影响合成语速。13 字 prompt → 5 字文本会被拉长到 5-6 秒。
    instruct2 没有这个限制，语速自然。
    """
    if len(text) < MIN_CHARS:
        logger.warning(f"Text too short (len={len(text)}), skip")
        return None

    # CosyVoice3 要求在文本前加 <|endofprompt|> 标记
    tts_text = f"{END_OF_PROMPT}{text}"

    # === 方案 A：instruct2 + 空指令（主方案）===
    try:
        for result in model.inference_instruct2(
            tts_text,           # 例："<|endofprompt|>别忘了外套!"
            "",                 # 空指令 = 不额外读任何文字
            prompt_wav,         # 源音频路径 → campplus 自动提取音色
            stream=False,       # 非流式，等全部生成完再返回
            speed=speed         # 语速缩放（1.0=原速，1.3=快30%）
        ):
            # result 是 dict，包含 "tts_speech" 键（torch.Tensor）
            speech = result["tts_speech"]
            dur = speech.shape[-1] / sr

            # 检查是否异常短（可能截断）
            # 正常中文约 3-5 字/秒，所以 len*0.25 秒是合理下限
            expected_min = max(1.0, len(text) * 0.25)
            if dur < expected_min:
                logger.warning(
                    f"Voice clone output short: {dur:.2f}s "
                    f"(expected >{expected_min:.2f}s)"
                )
            return speech
    except Exception as e:
        logger.warning(f"instruct2 failed: {e}")

    # === 方案 B：cross_lingual 降级 ===
    # 当 instruct2 失败时（极少发生），尝试 cross_lingual
    try:
        for result in model.inference_cross_lingual(
            tts_text, prompt_wav, stream=False
        ):
            return result["tts_speech"]
    except Exception as e:
        logger.warning(f"cross_lingual fallback failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 合成引擎 — 英文跨语言合成
# ═══════════════════════════════════════════════════════════════

def synthesize_english(
    model, text: str, prompt_wav: str, logger: logging.Logger, sr: int,
    speed: float = 1.0
) -> Optional[torch.Tensor]:
    """
    使用 cross_lingual 模式合成英文语音。

    【原理】
    cross_lingual 是 CosyVoice3 的跨语言合成接口：
      - 用中文说话人的音色（从 prompt_wav 提取）
      - 生成指定语言的语音（由 <|en|> 标签控制）

    <|en|> 是 CosyVoice3 训练时使用的语言标记，
    模型根据这个标记选择对应的语言 embedding。

    【已知限制】
    由于 CosyVoice3 训练数据以中文为主，英文发音会带有中式口音。
    这不是 bug，是训练数据分布决定的模型特性。
    """
    if len(text.split()) < 3:
        logger.warning(f"EN text too short (words={len(text.split())}), skip")
        return None

    # 拼装：分隔符 + 英文标签 + 英文文本
    tts_text = f"{END_OF_PROMPT}<|en|>{text}"

    try:
        for result in model.inference_cross_lingual(
            tts_text, prompt_wav, stream=False, speed=speed
        ):
            speech = result["tts_speech"]
            dur = speech.shape[-1] / sr

            # 英文约 2-3 词/秒
            expected_min = max(1.5, len(text.split()) * 0.3)
            if dur < expected_min:
                logger.warning(
                    f"EN output short: {dur:.2f}s "
                    f"(expected >{expected_min:.2f}s)"
                )
            return speech
    except Exception as e:
        logger.warning(f"EN cross_lingual failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 合成引擎 — 默认零样本（兜底方案）
# ═══════════════════════════════════════════════════════════════

def synthesize_default(
    model, text: str, prompt_wav: str, prompt_text: str,
    logger: logging.Logger, speed: float = 1.0
) -> Optional[torch.Tensor]:
    """
    标准的零样本 TTS（zero-shot），作为最后的兜底方案。

    当以下情况发生时使用：
    1. 源音频文件不存在
    2. instruct2 和 cross_lingual 都失败了
    3. 用户指定了 --no_voice_clone

    使用固定的默认音色（zero_shot_prompt.wav），不从源音频克隆。
    """
    tts_text = f"{END_OF_PROMPT}{text}"

    # prompt_text 也需要包含 <|endofprompt|>，与 tts_text 格式一致
    full_prompt = (
        prompt_text
        if END_OF_PROMPT in prompt_text
        else f"{END_OF_PROMPT}{prompt_text}"
    )

    try:
        for result in model.inference_zero_shot(
            tts_text, full_prompt, prompt_wav, stream=False, speed=speed
        ):
            return result["tts_speech"]
    except Exception as e:
        logger.error(f"zero-shot failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 音频保存
# ═══════════════════════════════════════════════════════════════

def save_wav(
    speech: torch.Tensor, path: str, sr: int, logger: logging.Logger
) -> float:
    """
    将 torch.Tensor 保存为 WAV 文件。

    torchaudio.save() 内部使用 libsndfile，支持自动选择编码格式。
    对于 24000Hz 单声道 float32，输出 PCM 16-bit WAV。
    返回音频时长（秒）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, speech.cpu(), sr)  # .cpu() 确保在 CPU 上保存
    dur = speech.shape[-1] / sr
    logger.info(f"Saved: {os.path.basename(path)} ({dur:.2f}s)")
    return dur


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    # ── 命令行参数 ──
    parser = argparse.ArgumentParser(
        description="C5 TTS — Fun-CosyVoice3-0.5B 双语音色克隆"
    )

    # 输入输出
    parser.add_argument("--input", required=True,
                        help="输入 JSON 路径（C3 或 C4 的结果文件）")
    parser.add_argument("--text_field", default="end2end_translation",
                        help="JSON 中文本字段名（C4: end2end_translation, C3: translation）")
    parser.add_argument("--outdir", default="outputs",
                        help="输出目录（将创建 wavs/ 和 c5_results.json）")

    # 模型和音色
    parser.add_argument("--model_dir", default=MODEL_DIR,
                        help="CosyVoice3 模型权重目录")
    parser.add_argument("--prompt_wav", default=DEFAULT_PROMPT_WAV,
                        help="兜底用的默认音色参考音频")
    parser.add_argument("--prompt_text", default=DEFAULT_PROMPT_TEXT,
                        help="兜底用的默认 prompt 文本")
    parser.add_argument("--speaker_prompt", default=None,
                        help="自定义音色参考音频（覆盖每条样本的源音频）")

    # 控制参数
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 条（0=全量）")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="语速缩放（0.8=慢, 1.0=正常, 1.5=快）")
    parser.add_argument("--trailing_silence", type=float,
                        default=TRAILING_SILENCE_SEC,
                        help="末尾静音时长（秒）")
    parser.add_argument("--no_voice_clone", action="store_true",
                        help="关闭音色克隆，统一用默认音色")
    parser.add_argument("--source", default="c4",
                        help="来源标记（c3/c4，仅用于记录）")

    args = parser.parse_args()

    # ── 创建输出目录 ──
    outdir = Path(args.outdir)
    wav_dir = outdir / "wavs"
    log_dir = outdir / "logs"
    for d in [wav_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 日志 ──
    logger = setup_logging(str(log_dir))
    logger.info("=" * 60)
    logger.info("C5 TTS — Voice Clone + Trailing Silence")
    logger.info(
        f"Input: {args.input}  |  Output: {outdir}"
    )
    logger.info(
        f"Clone: {'OFF' if args.no_voice_clone else 'ON'}  |  "
        f"Speed: {args.speed}  |  Silence: {args.trailing_silence}s"
    )
    if args.speaker_prompt:
        logger.info(f"Custom speaker: {args.speaker_prompt}")

    # ═══════════════════════════════════════════════════════════
    # 1. 加载模型
    # ═══════════════════════════════════════════════════════════
    model = load_model(args.model_dir, logger)
    sr = model.sample_rate if hasattr(model, "sample_rate") else 24000

    # ═══════════════════════════════════════════════════════════
    # 2. 读取输入
    # ═══════════════════════════════════════════════════════════
    samples = load_input(args.input, logger)
    if args.limit > 0:
        samples = samples[:args.limit]
        logger.info(f"Limited to first {len(samples)} samples")

    # ═══════════════════════════════════════════════════════════
    # 3. 校验路径
    # ═══════════════════════════════════════════════════════════

    # 默认 prompt 音频
    def_pwav = args.prompt_wav
    if not os.path.exists(def_pwav):
        alt = Path(__file__).parent.parent / args.prompt_wav
        if alt.exists():
            def_pwav = str(alt)
        else:
            logger.error(f"Default prompt wav not found: {args.prompt_wav}")
            sys.exit(1)

    # 自定义音色音频（如果指定）
    custom_spk = args.speaker_prompt
    if custom_spk and not os.path.exists(custom_spk):
        alt = Path(__file__).parent.parent / args.speaker_prompt
        if alt.exists():
            custom_spk = str(alt)
        else:
            logger.error(f"Speaker prompt not found: {custom_spk}")
            sys.exit(1)
    if custom_spk:
        logger.info(f"All samples will use custom speaker: {custom_spk}")

    # ═══════════════════════════════════════════════════════════
    # 4. 批量合成
    # ═══════════════════════════════════════════════════════════

    results = []            # 每条的结果记录
    t_start = time.time()   # 总计时
    ok = fail = skip = cloned = 0

    for idx, item in enumerate(samples):
        # ── 4.1 提取文本 ──
        sid = item.get("id", f"sample_{idx:04d}")
        text = item.get(args.text_field)
        # 降级尝试：如果指定字段为空，尝试其他常见字段
        if text is None:
            text = (
                item.get("end2end_translation")  # C4 字段
                or item.get("translation")        # C3 字段
                or item.get("output_text")        # 通用字段
            )

        if not text or not text.strip():
            logger.warning(f"[{sid}] Empty text, skip")
            results.append({
                "id": sid, "status": "skipped", "reason": "empty_text"
            })
            skip += 1
            continue

        # ── 4.2 文本预处理 ──
        text = text.strip()
        # 确保句末有标点，帮助模型正确结束句子
        if text[-1] not in "。！？.!?":
            text += "。"

        logger.info(f"[{idx+1}/{len(samples)}] {sid}: \"{text[:40]}\"")
        t0 = time.time()

        # ── 4.3 决定合成策略 ──
        # 音色来源优先级：
        #   1. --speaker_prompt（用户指定）→ 全部统一音色
        #   2. 源音频（C4 item.audio）     → 每条保留原始音色
        #   3. 默认 prompt                 → 兜底统一音色

        source_audio = item.get("audio", "")
        use_clone = (
            not args.no_voice_clone
            and source_audio
            and os.path.exists(source_audio)
        )

        if custom_spk:
            # 用户指定了统一音色 → 覆盖源音频
            use_clone = True
            source_audio = custom_spk

        # ── 4.4 语言路由 ──
        speech = None

        if is_english_text(text):
            # 英文 → cross_lingual（<|en|> 标签）
            logger.info(f"[{sid}] English detected → cross_lingual")
            speech = synthesize_english(
                model, text, def_pwav, logger, sr, speed=args.speed
            )
            if speech is not None:
                cloned += 1

        elif use_clone:
            # 中文 + 有源音频 → instruct2 音色克隆
            speech = synthesize_voice_clone(
                model, text, source_audio, logger, sr, speed=args.speed
            )
            if speech is not None:
                cloned += 1
            else:
                # instruct2 失败 → 降级为零样本
                logger.info(f"[{sid}] Clone failed, fallback to zero-shot")
                speech = synthesize_default(
                    model, text, def_pwav, args.prompt_text,
                    logger, speed=args.speed
                )

        else:
            # 中文 + 无源音频 → 默认零样本
            speech = synthesize_default(
                model, text, def_pwav, args.prompt_text,
                logger, speed=args.speed
            )

        # ── 4.5 后处理与保存 ──
        if speech is None:
            logger.warning(f"[{sid}] All synthesis methods failed")
            results.append({"id": sid, "status": "failed"})
            fail += 1
            continue

        # 记录原始语音时长（加静音前）
        orig_dur = speech.shape[-1] / sr
        # 末尾加静音
        speech = add_trailing_silence(speech, sr, args.trailing_silence)
        total_dur = speech.shape[-1] / sr
        elapsed = time.time() - t0

        # 保存文件
        wpath = str(wav_dir / f"{sid}.wav")
        save_wav(speech, wpath, sr, logger)
        rtf = elapsed / total_dur if total_dur > 0 else None

        # ── 4.6 记录单条统计 ──
        results.append({
            "id": sid,
            "status": "ok",
            "input_text": text,
            "audio_path": wpath,
            "audio_duration_sec": round(total_dur, 3),     # 含静音
            "speech_duration_sec": round(orig_dur, 3),     # 纯语音
            "synth_latency_sec": round(elapsed, 3),        # 合成耗时
            "rtf": round(rtf, 4) if rtf else None,        # 实时率
            "voice_cloned": use_clone,                      # 是否音色克隆
        })
        ok += 1

        # 定期清理 GPU 缓存，防止显存泄漏
        if idx % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # ═══════════════════════════════════════════════════════════
    # 5. 双语测试音频
    # ═══════════════════════════════════════════════════════════

    logger.info("Generating bilingual test audio (zh + en)...")

    # 中文测试
    zh_speech = synthesize_voice_clone(
        model, ZH_TEST_TEXT, def_pwav, logger, sr, speed=args.speed
    )
    if zh_speech is None:
        zh_speech = synthesize_default(
            model, ZH_TEST_TEXT, def_pwav, args.prompt_text,
            logger, speed=args.speed
        )
    if zh_speech is not None:
        zh_speech = add_trailing_silence(zh_speech, sr, args.trailing_silence)
        save_wav(zh_speech, str(wav_dir / "lang_zh.wav"), sr, logger)

    # 英文测试
    en_speech = synthesize_english(
        model, EN_TEST_TEXT, def_pwav, logger, sr, speed=args.speed
    )
    if en_speech is not None:
        en_speech = add_trailing_silence(en_speech, sr, args.trailing_silence)
        save_wav(en_speech, str(wav_dir / "lang_en.wav"), sr, logger)
    else:
        logger.warning("English test audio failed")

    # ═══════════════════════════════════════════════════════════
    # 6. 生成汇总报告
    # ═══════════════════════════════════════════════════════════

    total_t = time.time() - t_start
    ok_list = [r for r in results if r["status"] == "ok"]
    total_dur = sum(r.get("audio_duration_sec", 0) for r in ok_list)
    total_speech = sum(r.get("speech_duration_sec", 0) for r in ok_list)
    total_lat = sum(
        r.get("synth_latency_sec", 0)
        for r in results
        if "synth_latency_sec" in r
    )

    summary = {
        # 模块信息
        "module": "C5_TTS",
        "version": "voice-clone+bilingual",
        "model": "Fun-CosyVoice3-0.5B",

        # 输入来源（用于追溯）
        "input_json": args.input,
        "text_field": args.text_field,
        "source": args.source,

        # 合成数量
        "total_samples": len(samples),
        "ok": ok,
        "failed": fail,
        "skipped": skip,
        "voice_cloned": cloned,

        # 控制参数
        "trailing_silence_sec": args.trailing_silence,
        "speed": args.speed,
        "speaker_prompt": custom_spk if custom_spk else "source_audio",

        # 双语覆盖
        "languages_tested": ["zh", "en"],
        "bilingual_test_wavs": [
            "lang_zh.wav",
            "lang_en.wav"
        ],

        # 性能统计
        "total_time_sec": round(total_t, 3),
        "total_audio_duration_sec": round(total_dur, 3),
        "total_speech_duration_sec": round(total_speech, 3),
        "total_synth_latency_sec": round(total_lat, 3),
        "sample_rate": sr,

        # 每条明细
        "results": results,
    }

    # 保存 JSON
    out_path = outdir / "c5_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_path}")
    logger.info(
        f"Summary: total={len(samples)} ok={ok} fail={fail} "
        f"skip={skip} cloned={cloned}"
    )
    logger.info(
        f"Time: {total_t:.1f}s  Speech: {total_speech:.1f}s  "
        f"Total audio: {total_dur:.1f}s"
    )
    logger.info("C5 TTS — Done")

    # 最终清理
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
