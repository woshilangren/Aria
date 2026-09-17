"""能力层 - 语音通道和主动搭话

语音两条路：级联式（先转文字再合成）和端到端对话式。
key 没配的时候直接抛错让上层降级成纯文字，不装能用的样子。
"""

from shared.singletons import get_llm, services
from shared.types import MemoryBundle
from tools.misc import ClockTool
from tools.speech import strip_emotion_marks

# 主动搭话的备用话术，模型不给力就按时间轮换着用
class CascadeVoiceEngine:
    """级联式语音：录音先转文字，回复再合成音频，两步分开走。"""

    def transcribe(self, audio_bytes: bytes, fmt: str = "", sample_rate: int = 0) -> str:
        text = services.get("asr").transcribe(audio_bytes, fmt=fmt, sample_rate=sample_rate)
        if not text:
            raise RuntimeError("语音识别没出文字")
        return text

    def synthesize(self, text: str) -> bytes:
        audio = services.get("tts").synthesize(text)
        if not audio:
            raise RuntimeError("语音合成没出音频")
        return audio


class SpeechDialogEngine:
    """端到端语音对话：录音进来，转文字、生成回复、合成音频一条龙。"""

    def handle(self, audio_bytes: bytes, session_id: str = "default",
               fmt: str = "", sample_rate: int = 0) -> dict:
        """返回 {"user_text", "reply_text", "reply_audio"}。

        fmt/sample_rate 是前端音频的格式说明（现在固定 16k 裸 PCM），原样传给 ASR。
        任何一步失败都把异常抛给上层，上层决定降级成纯文字还是报错。
        """
        voice = CascadeVoiceEngine()
        user_text = voice.transcribe(audio_bytes, fmt=fmt, sample_rate=sample_rate)

        # 语音通道的回复也带人设，不能让角色突然说"作为一个AI"
        reply_text = self._reply(user_text)

        try:
            reply_audio = voice.synthesize(reply_text)
        except Exception:
            # 合成挂了就只回文字，前面的活不能白干
            reply_audio = b""

        # 标签只给合成器用：返回的文本剥干净，展示和记录都不带标记
        reply_text = strip_emotion_marks(reply_text)

        return {
            "user_text": user_text,
            "reply_text": reply_text,
            "reply_audio": reply_audio,
        }

    def _reply(self, user_text: str) -> str:
        """给转出来的文字生成一句带人设的回复。"""
        system = ""
        try:
            persona = services.get("kv_store").read("persona_config", "")
            system = persona.background_story
        except Exception:
            system = ""
        # 时间感知：语音通话里她看不见钟，时段分寸得喂给她
        clock = ClockTool()
        system += (
            f"\n\n【当前时间】现在是{clock.now()}（{clock.period()}）。{clock.time_guidance()}"
        )
        # 语音回复要带情绪：开头标 [情绪]，可带一句（语气描述），
        # 合成器会把标记拆出来转成情绪指令，正文拿去念，标记不会被念出来
        emotion_rule = (
            "【情绪标注】这条回复会被合成成语音："
            "开头先用一个方括号标出这段话的主情绪，比如 [开心]、[得意]、[生气]；"
            "标签后面可以用一个圆括号补一句不超过 6 个字的语气描述，比如（压低声音）。"
            "这两样标记整条回复里只出现一次，正文里不许再写任何括号补充或情绪标记，"
            "其余部分照常口语化聊天。"
        )
        system = f"{system}\n\n{emotion_rule}" if system else emotion_rule
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})
        return get_llm().chat(messages)


