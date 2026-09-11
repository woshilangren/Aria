"""语音识别和语音合成，走阿里百炼的 dashscope SDK。

识别用 qwen-audio-3.0-asr-flash-streaming：名字带 streaming，但 SDK 支持
"整段音频一次转"的非流式调用，正好对上现有的攒完整段再转的链路。
合成用 qwen-audio-3.0-tts-plus：支持情绪指令（比如"用轻快的语气说"），
对有人设的角色正好用得上。

情绪标注约定：回复开头的 [情绪] 标签和（语气描述）短句是给合成器的，
合成前拆出来转成 instruction，剥干净的正文才拿去念——
标签既不能被念出来，也不能进聊天记录。

key 没配的时候直接报错，别装作能用的样子，上层会自己降级成纯文字。
"""

import io
import os
import re
import struct
import tempfile
import wave

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
from dashscope.audio.tts_v2 import SpeechSynthesizer

from config.settings import get_settings

# 情绪标签白名单：prompt 里给过示例（开心/得意/生气/难过），再补几个常见的语气词。
# 收紧成白名单的好处：正文里的 [1]（如"第[1]点"）和普通括号（如"他（我朋友）"）
# 不会再被误当情绪标记剥掉——这是之前那个过宽正则的 bug。
EMOTION_WORDS = ("开心", "得意", "生气", "难过", "温柔", "惊喜", "无奈", "委屈", "兴奋", "害羞")

# [情绪] 标签：只认回复**开头**第一个，且必须是白名单里的词，比如 [开心]、[得意]
_EMOTION_TAG_RE = re.compile(r"^\s*\[(" + "|".join(EMOTION_WORDS) + r")\]")
# （语气描述）短句：只在情绪标签**紧邻之后**才认，全角/半角圆括号里 1~12 个字符
_ACTION_DESC_RE = re.compile(r"^\s*[（(]([^（）()]{1,12})[）)]")


def pcm_to_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
    """裸 PCM16 包一个 WAV 头。

    浏览器 <audio> 和 dashscope ASR 都只认带头的容器格式，
    前端 WebAudio 采上来的裸 PCM 进哪条链路前都先套上这层壳。
    """
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits
    )
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def sniff_audio_format(audio: bytes) -> str:
    """按文件头魔数猜音频格式，猜不出返回空串由调用方定默认值。"""
    if audio[:4] == b"RIFF":
        return "wav"
    if audio[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    if audio[:4] == b"OggS":
        return "ogg"
    if audio[:3] == b"ID3" or (len(audio) > 2 and audio[0] == 0xFF and audio[1] & 0xE0 == 0xE0):
        return "mp3"
    return ""


def split_emotion(text: str) -> tuple:
    """把**开头**的情绪标记从文本里拆出来：返回（干净正文，情绪词列表，描述列表）。

    只认开头的第一个 [情绪]（且在白名单里）以及紧随其后的（语气描述）——
    与 prompt 契约"标记整条只出现一次、且在开头"一致，正文里的方括号/圆括号不再误伤。
    """
    text = text or ""
    tags, descs = [], []
    rest = text
    m = _EMOTION_TAG_RE.match(rest)
    if m:
        tags.append(m.group(1))
        rest = rest[m.end():]
        d = _ACTION_DESC_RE.match(rest)
        if d:
            descs.append(d.group(1))
            rest = rest[d.end():]
    return rest.strip(), tags, descs


def strip_emotion_marks(text: str) -> str:
    """剥掉开头的情绪标签和语气描述，得到能直接展示、能落库的干净文本。"""
    clean, _tags, _descs = split_emotion(text)
    return clean


def _has_key(key: str) -> bool:
    """key 存在且不是占位符（占位符以 your- 开头，比如 your-xxx）。"""
    return bool(key) and not key.startswith("your-")


# 音色名没配的时候用这个兜底（百炼 qwen-tts 系的默认女声）
_DEFAULT_VOICE = "Cherry"


class _QuietCallback(RecognitionCallback):
    """同步调用用不上回调，但这版 SDK 构造时必须给一个，给个空实现。"""


def _extract_text(sentence) -> str:
    """兼容单句（dict）和多句（list）两种返回，拼成一段话。"""
    if not sentence:
        return ""
    if isinstance(sentence, dict):
        texts = [sentence.get("text") or ""]
    else:
        texts = [s.get("text") or "" for s in sentence if isinstance(s, dict)]
    return "".join(texts).strip()


class ASRTool:
    """语音转文字：给一段音频字节，还一段文字。"""

    def transcribe(self, audio: bytes, filename: str = "audio.wav",
                   fmt: str = "", sample_rate: int = 0) -> str:
        """fmt/sample_rate 由调用方显式传（前端固定发 16k 裸 PCM）；
        不传就按文件头魔数嗅探。裸 PCM 统一先包成 WAV 再送——
        dashscope 对 wav 的支持最稳，采样率也从头里读，不用靠猜。
        """
        cfg = get_settings()
        if not _has_key(cfg.asr_api_key):
            raise RuntimeError("语音识别服务没配置（ASR_API_KEY）")
        # 这版 SDK 的 key 不走构造参数，走全局设置（和 TTS 同一套路，调用前各设各的）
        dashscope.api_key = cfg.asr_api_key
        # 格式判定：显式参数 > 魔数嗅探 > 文件名后缀
        fmt = (fmt or "").lower().strip() or sniff_audio_format(audio) or (
            filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"
        )
        if fmt == "pcm":
            audio = pcm_to_wav(audio, sample_rate or 16000)
            fmt = "wav"
        # wav 就从文件头读真实采样率，其他格式按显式值或 16000 报
        sr = sample_rate or 16000
        if fmt == "wav":
            try:
                with wave.open(io.BytesIO(audio), "rb") as w:
                    sr = w.getframerate()
            except Exception:
                pass
        # 这版 SDK 的整段转写只收文件路径，先落临时文件再转
        tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        try:
            tmp.write(audio)
            tmp.close()
            recognition = Recognition(
                model=cfg.asr_model,
                callback=_QuietCallback(),
                format=fmt,
                sample_rate=sr,
            )
            result = recognition.call(file=tmp.name)
            return _extract_text(result.get_sentence() if result else None)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


class TTSTool:
    """文字转语音：给一段文字，还一段音频字节（mp3）。"""

    def synthesize(self, text: str, instruction: str = "") -> bytes:
        cfg = get_settings()
        if not _has_key(cfg.tts_api_key):
            raise RuntimeError("语音合成服务没配置（TTS_API_KEY）")
        # 文本里带的 [情绪] 和（语气描述）是给合成器的：拆出来转成指令，
        # 剥干净的正文才拿去念，标记绝不能被念出来
        clean, tags, descs = split_emotion(text)
        if not clean:
            clean = (text or "").strip()
        # 文本里带了标记就以它为准，没带才用调用方显式传的指令
        if not instruction and (tags or descs):
            mood = [t for t in tags if t][:2]
            desc = [d for d in descs if d][:2]
            bits = []
            if mood:
                bits.append(f"用{'、'.join(mood)}的语气")
            if desc:
                bits.append("，" + "，".join(desc))
            instruction = "".join(bits)
        # 这版 SDK 的合成器构造函数不收 key，key 走全局设置
        dashscope.api_key = cfg.tts_api_key
        kwargs = {
            "model": cfg.tts_model,
            "voice": cfg.tts_voice or _DEFAULT_VOICE,
        }
        # 情绪指令可选：比如"用轻快的语气，压低声音"，不传就正常念
        if instruction:
            kwargs["instruction"] = instruction
        synthesizer = SpeechSynthesizer(**kwargs)
        return synthesizer.call(clean)
