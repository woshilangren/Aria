"""交互层 - 消息收发网关

进来：把文字/语音统一成 InputMessage；
出去：把 FinalReply 整理成接口友好的形式，需要出声就合成音频。
只谈格式和搬运，理解了内容。

回复里可以自己标一段「语音版」：<voice>真正念出来的话</voice>。
后端的回复生成提示词教会模型这项能力，这里负责拆出来 + 合成音频（没标就纯文字）。
"""

import base64
import hashlib
import os
import re
from pathlib import Path

from config.settings import get_settings
from tools.misc import ClockTool
from shared.singletons import services
from shared.types import FinalReply, InputMessage

# 模型自己决定带语音时的封装标签：中途只可能有一个，多标签取下不取前
_VOICE_RE = re.compile(r"<voice>(.*?)</voice>", re.S)
_VOICE_CACHE = None


def _voice_cache_dir() -> Path:
    """语音缓存目录：同一段文字只合成一次，之后直接读缓存，不再烧 TTS。"""
    global _VOICE_CACHE
    if _VOICE_CACHE is None:
        _VOICE_CACHE = Path(os.environ.get("DATA_DIR", "storage")) / "voice"
        _VOICE_CACHE.mkdir(parents=True, exist_ok=True)
    return _VOICE_CACHE


def _voice_key(text: str) -> str:
    """缓存 key：文本 + 当前音色 + 当前模型。

    只按文本 md5 的话，换了音色（TTS_VOICE）还会命中旧音频，听起来"换了设置没生效"。
    把音色和模型一起算进 key，换配置即失效重合成（缓存本来就可丢弃）。
    """
    cfg = get_settings()
    raw = f"{text or ''}\x00{cfg.tts_voice}\x00{cfg.tts_model}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def get_cached_voice(text: str):
    """按内容取缓存音频；没有就返回 None（由调用方去合成并落缓存）。"""
    p = _voice_cache_dir() / _voice_key(text)
    try:
        return p.read_bytes() if p.exists() else None
    except Exception:
        return None


def save_cached_voice(text: str, audio: bytes):
    """把合成好的音频按内容落盘，供历史语音条免重复合成。"""
    try:
        (_voice_cache_dir() / _voice_key(text)).write_bytes(audio)
    except Exception:
        pass


def extract_voice(text: str) -> tuple:
    """把回复里模型自己标注的语音文字拆出来。

    返回 (voice_text, clean_text)：voice_text 是要合成的语音内容，
    clean_text 是去掉语音标签后给前端当文字备注的正文。
    找不到标签就都是纯文字（返回空语音）。顺带清掉可能的残缺标签，防止露馅。
    """
    text = text or ""
    m = _VOICE_RE.search(text)
    if not m:
        return "", re.sub(r"</?voice[^>]*>", "", text).strip()
    voice = (m.group(1) or "").strip()
    clean = (text[: m.start()] + text[m.end():]).strip()
    return voice, clean


class MessageReceiver:
    """消息接收入口，只做格式归一化，不理解内容。"""

    def receive(self, raw_text: str, session_id: str, image_url: str = "") -> InputMessage:
        """文字输入转标准消息。空文本不拦，交给后面的安全预检处理。"""
        return InputMessage(
            text=(raw_text or "").strip(),
            input_mode="text",
            session_id=session_id,
            image_url=image_url,
        )

    def receive_voice(self, audio_bytes: bytes, session_id: str,
                      fmt: str = "", sample_rate: int = 0) -> InputMessage:
        """语音输入先转文字再标准化。识别失败往上抛，调用方决定怎么回。"""
        text = services.get("asr").transcribe(audio_bytes, fmt=fmt, sample_rate=sample_rate)
        if not text:
            raise RuntimeError("语音识别没出文字")
        return InputMessage(
            text=text,
            input_mode="voice",
            session_id=session_id,
        )


class OutputRenderer:
    """回复渲染与发送。"""

    def render(self, reply: FinalReply, session_id: str) -> dict:
        """把最终回复整理成接口要返回的字典。

        模型在回复里标了 <voice> 语音版时：正文只留简短文字备注，语音版合成音频——
        语音是主、文字是备注，两者方向一致但不必一模一样。
        带图的回复顺带确认图片文件在不在，不在就降级成纯文字，
        免得前端拿个空路径干瞪眼。
        """
        voice_text, clean_text = extract_voice(reply.text)
        if not voice_text and (reply.voice_text or "").strip():
            # R14a：语音派生已有独立载体（FinalReply.voice_text）——正文里不再
            # 塞 <voice> 标签，旧协议的"塞回去再拆"到此为止。extract_voice
            # 保留是为了兼容仍带标签的 legacy 来源。
            voice_text = reply.voice_text.strip()
        data = {
            "text": clean_text or reply.text,
            "output_mode": reply.output_mode,
            "image_path": reply.image_path,
        }
        if voice_text:
            try:
                # R14b：voice_text 的同义性无法可靠核验（它由模型自标，可能
                # 混入正文没有的事实/承诺）——TTS 源直接用 canonical 正文
                # （8.11.1 契约明文允许），不靠 prompt 承诺。voice_text 降级为
                # "存在语音表达"的信号，不再作为合成源。
                spoken = clean_text or reply.text
                audio = self.render_voice(spoken, session_id)
                save_cached_voice(spoken, audio)  # 落盘，历史语音条免重复合成
                data["voice_audio"] = base64.b64encode(audio).decode()
                data["voice_text"] = spoken
            except Exception:
                # 语音合成只是锦上添花，失败就退回纯文字，别影响聊天
                pass
        if reply.output_mode == "image":
            from pathlib import Path

            if not reply.image_path or not Path(reply.image_path).exists():
                data["output_mode"] = "text"
                data["image_path"] = ""
        return data

    def render_voice(self, text: str, session_id: str) -> bytes:
        """文字转语音，返回音频字节。合成失败往上抛，调用方降级成纯文字。"""
        audio = services.get("tts").synthesize(text)
        if not audio:
            raise RuntimeError("语音合成没出音频")
        return audio


class ConnectionKeeper:
    """连接状态管理：谁在线、心跳什么时候跳的，就记这点事。"""

    def __init__(self):
        self._sessions: dict = {}

    def heartbeat(self, session_id: str) -> None:
        """记一次心跳，顺便把不在线的标记成在线。"""
        self._sessions[session_id] = ClockTool().now()

    def is_connected(self, session_id: str) -> bool:
        return session_id in self._sessions

    def on_disconnect(self, session_id: str) -> None:
        """断线就把记录摘掉，流资源由各端点自己清理。"""
        self._sessions.pop(session_id, None)
