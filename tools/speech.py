"""语音识别和语音合成，走阿里百炼的 dashscope SDK。

识别用 qwen-audio-3.0-asr-flash-streaming：名字带 streaming，但 SDK 支持
"整段音频一次转"的非流式调用，正好对上现有的攒完整段再转的链路。
合成用 qwen-audio-3.0-tts-plus：支持情绪指令（比如"用轻快的语气说"），
对有人设的角色正好用得上。

情绪标注约定：回复开头的 [情绪] 标签和（语气描述）短句是给合成器的，
合成前拆出来转成 instruction，剥干净的正文才拿去念——
标签既不能被念出来，也不能进聊天记录。

key 没配的时候直接报错，别装作能用的样子，上层会自己降级成纯文字。

每一次 dashscope 调用都有明确的墙钟上限（J3，取值在 config.json 的 timeouts 段）：
ASR 走"专用线程池 + future.result(timeout)"（这版 SDK 的 Recognition.call 压根
不收超时参数），TTS 走 SDK 原生的 timeout_millis。没有上限的调用是 24/7 运行
最大的挂起风险——它挂住的是线程，不是协程，不会自己醒。
"""

import io
import os
import re
import struct
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
from dashscope.audio.tts_v2 import SpeechSynthesizer

from config.settings import get_settings, timeout_seconds
from tools.misc import has_key

# config.json 的 timeouts 段缺失/写坏时的兜底上限（秒），与 config/settings.py 的
# defaults 保持一致——两处都得有，因为 timeout_seconds 需要一个"配置全丢"时的落点
_ASR_TIMEOUT_DEFAULT = 60.0
_TTS_TIMEOUT_DEFAULT = 60.0

# dashscope 的同步调用一律丢进这个**专用**线程池，再在外面用 future.result(timeout)
# 卡住上限（J3）。
#
# 为什么不直接让它跑在 asyncio.to_thread 的默认池里：默认池是全项目共享的
# （SQLite 写回、文件读写、感知/画像那些慢调用全靠它 offload），而一个挂死的
# ASR 请求会占住一个池线程且**永不释放**；攒够 min(32, cpu+4) 个之后，所有
# offload 调用开始排队——表现是"整个服务卡死"，而日志里一条报错都没有。
# 隔离到专用池后，最坏也只是语音功能自己降级，写回那条命脉不受牵连。
#
# max_workers 给 4 而不是 1：超时之后底下那个线程可能仍被占着（见 _run_bounded
# 里的说明），单 worker 会让一次挂死之后的所有语音请求永久排队。4 个既有缓冲，
# 又给"最多同时挂着几个僵尸"封了顶。
_DASHSCOPE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dashscope")


def _run_bounded(fn, timeout_s: float, what: str):
    """在专用池里跑 fn，超过 timeout_s 就不再等它，抛 RuntimeError。

    必须说清楚的一点：超时只是"**我们不等了**"，底下那个线程可能还在跑。
    这版 dashscope 的 `Recognition.call()` 是同步阻塞的 websocket 收流循环，
    没有可以安全调用的取消口子——`stop()` 只翻 `_running` 标志、并不会中断
    `call()` 里正在迭代的那个响应生成器。所以超时后必须把这笔僵尸调用大声记下来：
    静默丢弃就等于把"专用池正在被蚕食"这件事藏起来，正是 J3 要治的那个病
    （兜底哲学：外部服务失败可以降级，但必须留痕）。

    统一抛 RuntimeError 而不是内建的 TimeoutError：本模块既有的失败契约就是
    RuntimeError（key 没配也是它），上层 skills.py / gateway.py 靠它降级成纯文字。
    """
    future = _DASHSCOPE_POOL.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except _FutureTimeout:
        # 还没被 worker 取走的话 cancel() 能拦下来；已经在跑的拦不住，只能记账
        still_running = not future.cancel()
        note = (
            "；底层调用仍挂在专用池线程里（已与 asyncio 默认池隔离，不会拖死写回）"
            if still_running else ""
        )
        print(f"[speech] {what} 超过 {timeout_s:.0f}s 未返回，已放弃等待{note}")
        raise RuntimeError(f"{what}超时（{timeout_s:.0f}s）") from None


# dashscope 的 key 是模块级全局（dashscope.api_key），而这版 SDK 的
# Recognition / SpeechSynthesizer 构造函数都不收 key，只能在调用前设全局。
# ASR 和 TTS 各配各的 key（ASR_API_KEY / TTS_API_KEY），两者又都跑在
# asyncio.to_thread 里 —— 并发时后设的 key 会覆盖先设的，先那个请求就拿到错 key。
# 所以"设 key + 发起调用"必须整段串行：只锁赋值不够，锁在调用返回前都得拿着。
#
# 代价：所有 dashscope 调用被串行化，ASR 和 TTS 也会互相等。
# 对单用户应用可接受（语音交互本来就是一轮一轮来的）；
# 真要并发上去，得换成每个请求独立 client（SDK 支持了再改这里，只动这一处）。
_DASHSCOPE_LOCK = threading.Lock()

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
        if not has_key(cfg.asr_api_key):
            raise RuntimeError("语音识别服务没配置（ASR_API_KEY）")
        # 这版 SDK 的 key 不走构造参数，走全局设置（和 TTS 同一套路，调用前各设各的）
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
            # 设 key -> 构造 -> 发起调用，整段持锁（见 _DASHSCOPE_LOCK 的注释）
            with _DASHSCOPE_LOCK:
                dashscope.api_key = cfg.asr_api_key
                recognition = Recognition(
                    model=cfg.asr_model,
                    callback=_QuietCallback(),
                    format=fmt,
                    sample_rate=sr,
                )
                # J3：这版 SDK 的 Recognition.call() 不收任何超时参数（1.27.2 的
                # 签名只有 file / phrase_id / 几个识别开关 + **kwargs，kwargs 全被
                # 透传成请求参数），内部又是同步阻塞的 websocket 收流循环——挂住就是
                # 永久挂住。所以只能从外面包一层：专用线程池 + future.result(timeout)。
                #
                # 已知代价（属于 J14 的范畴，这里只记账不动手）：超时后本函数会带着
                # 异常退出 with 块、把 _DASHSCOPE_LOCK 放掉，而底下那个僵尸调用可能
                # 还在跑；此时若另一个请求进来重设 dashscope.api_key，就出现了这把锁
                # 本来要防的"key 被覆盖"窗口。不放锁则是永久死锁，两害相权取其轻——
                # 窗口只有"僵尸还没读完请求参数"那么长，且专用池已经把爆炸半径限制
                # 在语音功能内。
                result = _run_bounded(
                    lambda: recognition.call(file=tmp.name),
                    timeout_seconds("asr_seconds", _ASR_TIMEOUT_DEFAULT),
                    "语音识别（ASR）",
                )
            return _extract_text(result.get_sentence() if result else None)
        finally:
            # 就算 ASR 超时留下了僵尸线程，这里删临时文件也是安全的：SDK 在
            # call() 的头几毫秒就把整个文件读进自己的队列再关掉了文件句柄，
            # 而超时是几十秒量级的事；万一真撞上（Windows 下删被占用的文件会
            # 抛 PermissionError），下面这个 OSError 兜底会把它咽掉，只漏一个临时文件。
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


class TTSTool:
    """文字转语音：给一段文字，还一段音频字节（mp3）。"""

    def synthesize(self, text: str, instruction: str = "") -> bytes:
        cfg = get_settings()
        if not has_key(cfg.tts_api_key):
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
        kwargs = {
            "model": cfg.tts_model,
            "voice": cfg.tts_voice or _DEFAULT_VOICE,
        }
        # 情绪指令可选：比如"用轻快的语气，压低声音"，不传就正常念
        if instruction:
            kwargs["instruction"] = instruction
        # 设 key -> 构造 -> 发起调用，整段持锁（见 _DASHSCOPE_LOCK 的注释）
        timeout_s = timeout_seconds("tts_seconds", _TTS_TIMEOUT_DEFAULT)
        with _DASHSCOPE_LOCK:
            dashscope.api_key = cfg.tts_api_key
            synthesizer = SpeechSynthesizer(**kwargs)
            try:
                # J3：这版 SDK 的 SpeechSynthesizer.call 原生就收 timeout_millis
                # （1.27.2 签名：call(text, timeout_millis=None)），超时抛 TimeoutError，
                # 并且 streaming_complete 的 finally 里会 __cleanup_task() 把 websocket
                # 关掉——比 ASR 那样从外面包线程池干净得多，所以能用原生的就用原生的。
                return synthesizer.call(clean, timeout_millis=int(timeout_s * 1000))
            except TimeoutError as exc:
                # 转成 RuntimeError：本模块既有的失败契约就是它（key 没配也是 RuntimeError），
                # 上层靠它降级成纯文字；把内建 TimeoutError 直接漏出去，只 catch
                # RuntimeError 的地方就会漏接。超时本身要留痕，别静默。
                print(f"[speech] 语音合成（TTS）超过 {timeout_s:.0f}s 未完成，本次按失败处理")
                raise RuntimeError(f"语音合成超时（{timeout_s:.0f}s）") from exc
