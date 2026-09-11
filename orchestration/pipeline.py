"""调度层 - 对话管道（去 LangGraph，句级流式）

一份实现，两种消费：
- `astream(message, synthesize_voice, want_voice)` 是唯一实现，逐事件产出（SSE 用）；
- `handle(message)` 是它的**消费者**：内部把事件收起来重拼成 `FinalReply`，
  返回形状与改造前完全一致（text 保留带 `<voice>` 的原文），
  既有的 `renderer.render` 照旧去剥标签 + 合成语音，`/api/chat/send` 与 Gradio `/ui` 零变化。

一轮对话从安全预检进来、到记忆写回出去，中间按意图/情绪分流。
以前这是一张 LangGraph 状态图（一摊节点函数 + 一段建图接线）；
现在换成一条普通 async 管道：顺序调用若干方法 + if 分支 + 一个有界重写循环，
语义与原图逐条对齐，但不再依赖 langgraph。

分流语义（与原图一致）：
    预检不过            → refuse（不写回）
    工具类意图(weather/search/image/diary) → toolcall → respond → 句级审核
    其余意图            → compose → 流式生成 → 句级审核
    缺城市追问(toolcall 已写 final_reply) → 直接出文本（跳过 respond）
    句级审核过          → 发 sentence
    句级审核不过        → 整轮降级 refuse（不写回）
    首句只有一两个字的短句 → 先扣住、等下一句；若整轮就这么点，安全重写（最多 _MAX_REWRITE 次）

流式与原图的**有意差异**（后续批次/前端已知）：
- 原图是"整轮生成 → 整轮审核 → 最多重写 2 次 → 婉拒"；流式路径改成**句级审核**：
  某句一旦命中黑名单，已推出去的句子撤不回，所以不再逐句/整轮重写，直接整轮降级。
  这是**一次有意的能力下降**——换来的是首 token 一到就能开口说话。
- 单字崩不再按句判，改成"首句短缓冲 + 收尾安全重写"（重写时一个 token 都还没推，安全）。
"""

import asyncio
import base64
import itertools
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from shared.types import (
    EmotionResult,
    FinalReply,
    InputMessage,
    IntentResult,
    MemoryBundle,
    PromptPackage,
)

from capability.memory import (
    ConversationDistiller,
    MemoryRecaller,
    PortraitBuilder,
    RelationshipTracker,
    SessionMemoryKeeper,
)
from capability.perception import PerceptionPipeline, SafetyReviewer
from capability.persona_engine import PersonaEngine
from capability.quirks import MoodEngine, QuirkDirector
from capability.response_generator import build_chat_messages, generate, persona_wrap
from capability.toolcall import ToolCallOrchestrator

from config.settings import load_app_config

from orchestration.cancellation import TURN_REGISTRY, TurnCancelled
from orchestration.managers import FallbackController, InfoGapCoordinator
from shared.singletons import services
from tools.speech import EMOTION_WORDS, split_emotion, strip_emotion_marks

# 短期记忆的管家全项目就这一份：compose 要拿上下文、writeback 要写记录，
# 两处用的必须是同一个对象，不然短期记忆各记各的就对不上了
KEEPER = SessionMemoryKeeper()

# 人设引擎也共用一份，绑定上面那个管家
_ENGINE = PersonaEngine(KEEPER)

# 后置审核最多重写几次，超了就给兜底话，别死循环
_MAX_REWRITE = 2

# 工具类意图：命中就走工具链，其余走人设链
_TOOL_INTENTS = ("weather", "search", "image", "diary")

# 语音轮的情绪标注规矩：回复开头标 [主情绪]，可带一句（不超过 6 字的语气描述）。
# 标记是给语音合成器看的：合成时拆出来转成情绪指令，剥干净的正文才拿去念，
# 所以标记既不能被念出来，也不能进聊天记录（写回和展示前都会剥掉）。
_VOICE_EMOTION_RULE = (
    "【情绪标注（本条回复会被合成成语音）】"
    "回复开头先用一个方括号标出这段话的主情绪，"
    "比如 [开心]、[得意]、[生气]、[难过]；"
    "标签后面可以用一个圆括号补一句不超过 6 个字的语气描述，比如（压低声音）。"
    "这两样标记整条回复里只出现一次，正文里不许再写任何括号补充或情绪标记，"
    "其余部分照常口语化聊天。"
)

# 句级切分：命中这些标点就切一句（与前端 splitSentences 同一套）。
_SENT_BOUNDARY = "。！？；\n"
# 一段没有标点的超长文本也切一刀，别让一"句"无限长下去
_FORCE_CUT = 60
# 首句短缓冲阈值：首句去标点后不超过这么多实质字符就暂缓推出，等下一句裁决
_FIRST_HOLD_MAX = 4
# <voice> / </voice> 标签
_VOICE_OPEN = "<voice>"
_VOICE_CLOSE = "</voice>"
# 模型自标的"本能反应"前缀：看到就放行极简判定，推送前剥掉
_REACTION_TAG = "@r"


class _ReviewReject(Exception):
    """审核没过（句子或 <voice> 语音内容）。

    这一层只负责"报告没过"，**不自己决定**怎么处理——由上层按统一规则裁决：
    还没推出任何内容 → 安全重写（最多 _MAX_REWRITE 次）；已经推过 → 整轮降级。
    """


class _RefuseTurn(Exception):
    """整轮降级成兜底话，停止推流、不写回。"""

    def __init__(self, text: str):
        super().__init__(text)
        self.text = text


@dataclass
class TurnState:
    """单轮对话的流转状态（管道各步读取并补充字段）。"""

    # 输入
    user_text: str
    session_id: str
    voice_mode: bool = False
    user_image: str = ""

    # 各步产出
    safety_passed: bool = True
    intent: Optional[IntentResult] = None
    emotion: Optional[EmotionResult] = None
    subtext: str = ""                # 感知步骤读出的潜台词/弦外之音
    comfort_mode: bool = False
    think: bool = False              # 本轮是否触发"认真想"（按输入复杂度逐轮判定）
    reaction_tagged: bool = False    # 模型自标了 @r（这一句是本能反应）
    memory: Optional[MemoryBundle] = None
    prompt: Optional[PromptPackage] = None
    draft_reply: str = ""
    tool_results: list = field(default_factory=list)
    rewrite_count: int = 0
    review_passed: bool = False
    review_block_reason: str = ""
    final_reply: str = ""
    output_mode: str = "text"
    image_path: str = ""
    error: str = ""


# 纯 emoji / 无实质文字输入：小模型没话接，容易借着人设蹦单字。
# 这里只对"发给 LLM"的输入补一句语义，库里的原始 user 消息不受影响。
_TELEGRAPH_RE = None


def _enrich_telegraph(raw: str) -> str:
    import re as _re

    global _TELEGRAPH_RE
    if _TELEGRAPH_RE is None:
        _TELEGRAPH_RE = _re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
    t = (raw or "").strip()
    if not t:
        return "（对方安静地陪着你去）用一句自然的完整话回应他。"
    if not _TELEGRAPH_RE.search(t):
        return (
            f"{t}（对方只发来这个表情，这是一句很生动的情绪。好好接住他："
            "用一句完整的俏皮或关心的话回他，把话说完整，绝不只回单个字。）"
        )
    return t


def _subst_count(text: str) -> int:
    """数一句里有多少"实质字符"（汉字/字母/数字），标点空格不算。"""
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text or ""))


def _is_onechar_reply(reply: str) -> bool:
    """单字崩检测：整句去掉空格标点后只剩 1 个汉字/字母/数字，判为敷衍。
    被夸/被闹时小模型爱用"哼/啧/哈/你/哦"单字打发，这里硬拦。"""
    return _subst_count(reply) <= 1


# 「想」的触发正则：出现这些模式说明这条消息需要动脑（连问/追问原因/取舍/要观点）
_THINK_RE = re.compile(
    r"[?？]\s*.{0,40}[?？]"           # 多个问号 / 连问
    r"|为什么|怎么会|凭什么|咋就"        # 追问原因
    r"|怎么办|该怎么|要不要|还是|选哪个"  # 取舍 / 决策
    r"|帮我(想|分析|看看|比较|总结|梳理)"  # 明确要动脑
    r"|你觉得|你认为|你的看法|怎么看"     # 要观点
    r"|第一|第二|首先|其次|另外"         # 多要点
)


def _needs_thinking(state: TurnState) -> bool:
    """这一轮要不要让模型"认真想"。按输入复杂度逐轮判定，不是全局开关。

    反向规则比正向更重要：安抚和情绪回应绝对不想——这时候要温度，一分析就
    变成心理咨询师了（用户说的"开思考智商高情商低"就是这个）。
    """
    cfg = load_app_config()["thinking"]
    if not cfg.get("enabled", True):
        return False
    # 安抚轮 + 情绪轮：绝对不想
    if state.comfort_mode and cfg.get("never_on_comfort", True):
        return False
    emo = state.emotion.emotion if state.emotion else "neutral"
    if emo in (cfg.get("never_on_emotions") or []):
        return False
    # 短情绪消息：也不要想（比如"我今天好难过"——要的是陪伴）
    if emo in ("sad", "tired") and _subst_count(state.user_text) < 20:
        return False
    # 该想的：字数达到阈值，或命中复杂度正则
    if len(state.user_text) >= int(cfg.get("char_threshold", 60)):
        return True
    if _THINK_RE.search(state.user_text):
        return True
    return False


def _too_short_input(user_text: str) -> str:
    """单字崩的安全重写提示（与改造前 rewrite_node 的 too_short 分支一字不差）。"""
    return (
        f"{user_text}\n"
        "（系统提醒：你刚才只回了一个字，太敷衍了。请重新说一句完整的话，"
        "20~50 字，把此刻的情绪和想说的话好好说清楚，"
        "话都要说完整，不许用单个字符号打发人。）"
    )


def _clean_review_input(user_text: str) -> str:
    """句级审核没过、且还没推出任何内容时的安全重写提示。

    与单字崩那条重写走同一条统一路径：换个干净说法重新回答，
    不许违规、也不许把"被拦下"这件事说出去。
    """
    return (
        f"{user_text}\n"
        "（系统提醒：请换一个更得体、更安全的说法重新回答，"
        "不要包含任何违规或不适宜的内容，"
        "也不要提及刚才被拦下或重来的这件事，直接自然地把话说完即可。）"
    )


def _has_city(text: str) -> bool:
    """粗查用户话里有没有城市字样，有"天气"前面接地名就算。"""
    return bool(re.search(r"[\u4e00-\u9fa5]{2,8}(市)?的?天气", text or ""))


def _portrait_interval() -> int:
    """几轮聊天更新一次画像，从配置里读，读不到就 5 轮。"""
    try:
        from config.settings import load_app_config

        return int(load_app_config()["memory"].get("session_update_interval", 5)) or 5
    except Exception:
        return 5


class _ReplyStreamer:
    """把模型/文本流装配成可推的片段（同步、纯逻辑，不碰 I/O）。

    处理三件事：
    - 情绪前缀：仅语音轮，扣住开头 ≤24 字解析 `[情绪]（语气）`，内容不进正文；
    - `<voice>` 扣留：末尾未闭合的标签前缀一律扣住，绝不把 `</voice` 当正文推出去；
    - 句子切分 + 首句短缓冲：遇标点（或超长）切句；首句很短就先扣住等下一句裁决。

    feed()/finish() 产出片段，元素形如：
        ("emotion", 情绪词, 语气描述) / ("sentence", 文本) / ("voice", 文本)
    以及 finish() 才可能出现的 ("held", 文本)（整轮只剩一句短话，交上层决定重写）。
    """

    def __init__(self, voice_mode: bool):
        self.voice_mode = voice_mode
        self.emitted_content = False  # 是否已产出过 sentence/voice
        self.reaction_tagged = False  # 模型自标了 @r：这一句是本能反应
        self._tag_checked = False     # 是否已经检查过开头的 @r
        self._emotion_done = not voice_mode
        self._emo_buf = ""
        self._raw = ""  # 待处理原始文本（可能含被截断的标签）
        self._in_voice = False
        self._voice_buf = ""
        self._text_buf = ""
        self._held = None  # 首句短缓冲

    # ---- 反应标记 @r：只看每个片段的开头，剥掉并登记 ----
    def _maybe_strip_tag(self, s: str) -> str:
        if self._tag_checked or self.reaction_tagged:
            return s
        self._tag_checked = True
        t = s.lstrip()
        if t.startswith(_REACTION_TAG):
            self.reaction_tagged = True
            return t[len(_REACTION_TAG):].lstrip()
        return s

    # ---- 情绪前缀 ----
    def _emit_emotion(self, emo, desc, rest, out):
        self._emotion_done = True
        self._emo_buf = ""
        if emo:
            out.append(("emotion", emo, desc))
        self._raw += rest

    def _try_resolve_emotion(self, out) -> bool:
        s = self._emo_buf.lstrip()
        if not s:
            return False
        if s[0] != "[":
            self._emit_emotion("", "", self._emo_buf, out)
            return True
        close = s.find("]")
        if close == -1:
            if len(self._emo_buf) >= 24:
                self._emit_emotion("", "", self._emo_buf, out)
                return True
            return False
        word = s[1:close]
        after = s[close + 1:].lstrip()
        if word not in EMOTION_WORDS:
            self._emit_emotion("", "", self._emo_buf, out)
            return True
        # 可能是（语气描述）但右括号还没来：等一等，别把描述当正文
        if after[:1] in ("（", "(") and "）" not in after and ")" not in after:
            if len(self._emo_buf) >= 24:
                clean, tags, _d = split_emotion(self._emo_buf)
                self._emit_emotion(tags[0] if tags else "", "", clean, out)
                return True
            return False
        clean, tags, descs = split_emotion(self._emo_buf)
        self._emit_emotion(tags[0] if tags else "", descs[0] if descs else "", clean, out)
        return True

    # ---- 标签尾部扣留（比固定 8 字更准：只扣住"不完整的标签前缀"）----
    @staticmethod
    def _split_tail(text: str):
        i = text.rfind("<")
        if i == -1:
            return text, ""
        cand = text[i:]
        for tag in (_VOICE_OPEN, _VOICE_CLOSE):
            if len(cand) < len(tag) and tag.startswith(cand):
                return text[:i], cand
        return text, ""

    # ---- 片段产出（含首句短缓冲）----
    def _push_sentence(self, s, out):
        s = self._maybe_strip_tag(s)
        if self._held is not None:
            out.append(("sentence", self._held))
            self._held = None
            self.emitted_content = True
        if not self.emitted_content and _subst_count(s) <= _FIRST_HOLD_MAX:
            self._held = s  # 首句很短：先扣住，等下一句
            return
        out.append(("sentence", s))
        self.emitted_content = True

    def _push_voice(self, v, out):
        v = self._maybe_strip_tag(v)
        if self._held is not None:
            out.append(("sentence", self._held))
            self._held = None
            self.emitted_content = True
        out.append(("voice", v))
        self.emitted_content = True

    def _consume(self, work: str):
        out = []
        i, n = 0, len(work)
        while i < n:
            if self._in_voice:
                j = work.find(_VOICE_CLOSE, i)
                if j == -1:
                    self._voice_buf += work[i:]
                    i = n
                else:
                    self._voice_buf += work[i:j]
                    self._push_voice(self._voice_buf, out)
                    self._voice_buf = ""
                    self._in_voice = False
                    i = j + len(_VOICE_CLOSE)
            else:
                jv = work.find(_VOICE_OPEN, i)
                jb = -1
                for k in range(i, n):
                    if work[k] in _SENT_BOUNDARY:
                        jb = k
                        break
                if jv != -1 and (jb == -1 or jv < jb):
                    # <voice> 之前若有文字，先按一句结算（否则顺序会乱、标签前的话被吞）
                    self._text_buf += work[i:jv]
                    if self._text_buf:
                        self._push_sentence(self._text_buf, out)
                        self._text_buf = ""
                    self._in_voice = True
                    i = jv + len(_VOICE_OPEN)
                elif jb != -1:
                    self._text_buf += work[i:jb + 1]
                    self._push_sentence(self._text_buf, out)
                    self._text_buf = ""
                    i = jb + 1
                else:
                    self._text_buf += work[i:]
                    i = n
        return out

    def _force_cuts(self, out):
        while not self._in_voice and len(self._text_buf) >= _FORCE_CUT:
            cut = self._text_buf[:_FORCE_CUT]
            self._text_buf = self._text_buf[_FORCE_CUT:]
            self._push_sentence(cut, out)

    def feed(self, chunk: str):
        out = []
        if not self._emotion_done:
            self._emo_buf += chunk
            if not self._try_resolve_emotion(out):
                return out  # 情绪前缀还没攒够，先不处理
            chunk = ""  # 已并入 _raw
        self._raw += chunk
        work, tail = self._split_tail(self._raw)
        self._raw = tail
        out.extend(self._consume(work))
        self._force_cuts(out)
        return out

    def finish(self):
        out = []
        if not self._emotion_done:
            if not self._try_resolve_emotion(out):
                self._emit_emotion("", "", self._emo_buf, out)  # 强制判定：不再等了
        if self._raw:
            out.extend(self._consume(self._raw))
            self._raw = ""
        if self._in_voice:
            # 模型忘了闭合 </voice>：把攒的当语音处理，别丢
            self._push_voice(self._voice_buf, out)
            self._voice_buf = ""
            self._in_voice = False
        elif self._text_buf:
            self._push_sentence(self._text_buf, out)
            self._text_buf = ""
        if self._held is not None:
            out.append(("held", self._held))
            self._held = None
        return out


class DialoguePipeline:
    """单轮对话主流程（普通 async 管道，替代原 LangGraph 编排）。"""

    def __init__(self, turns=None):
        self._turns = turns or TURN_REGISTRY

    # ================= 消费者：非流式入口 =================
    async def handle(self, message: InputMessage) -> FinalReply:
        """处理一条用户消息，返回最终回复（主入口）。

        自己不做逻辑，只是 `astream` 的消费者：把事件收起来重拼成 FinalReply。
        text 保留带 `<voice>` 的原文，交给 renderer 去剥标签 + 合成语音，
        返回形状与改造前完全一致（/api/chat/send 与 Gradio /ui 零变化）。
        """
        frags = []  # (seq, kind, content)
        emotion = ""
        desc = ""
        output_mode = "text"
        image_path = ""
        cancelled = False

        async for ev in self.astream(message, synthesize_voice=False, want_voice=False):
            t = ev.get("type")
            if t == "emotion":
                emotion = ev.get("emotion") or ""
                desc = ev.get("desc") or ""
            elif t == "sentence":
                frags.append((ev.get("seq", 0), "text", ev.get("text", "")))
            elif t == "voice":
                frags.append((ev.get("seq", 0), "voice", ev.get("voice_text", "")))
            elif t == "image":
                image_path = ev.get("path", "") or ""
                output_mode = "image"
            elif t == "refuse":
                frags = [(0, "text", ev.get("text", ""))]
                emotion, desc = "", ""
                output_mode, image_path = "text", ""
            elif t == "done":
                output_mode = ev.get("output_mode") or output_mode
                image_path = ev.get("image_path") or image_path
            elif t == "cancelled":
                cancelled = True
            elif t == "error":
                return FinalReply(text=FallbackController().fallback_reply("llm"))

        if cancelled:
            # 取消不当异常兜底成 llm 兜底话：这轮当没发生过，给个空回复
            return FinalReply(text="")

        frags.sort(key=lambda x: x[0])
        body = "".join(
            f"{_VOICE_OPEN}{c}{_VOICE_CLOSE}" if kind == "voice" else c
            for _seq, kind, c in frags
        )
        prefix = f"[{emotion}]" + (f"（{desc}）" if desc else "") if emotion else ""
        return FinalReply(
            text=prefix + body,
            output_mode=output_mode or "text",
            image_path=image_path or "",
        )

    # ================= 唯一实现：流式入口 =================
    async def astream(
        self,
        message: InputMessage,
        synthesize_voice: bool = False,
        want_voice: bool = False,
    ) -> AsyncIterator[dict]:
        """处理一条用户消息，逐事件产出（SSE 端点用）。

        synthesize_voice=True 时由管道内部合成语音并通过 `voice` 事件带出音频；
        False 时 `voice` 事件只带文本（给 handle() 重拼原文用），合成交给上层 renderer。
        want_voice=True 时，若整轮一个 `<voice>` 都没出，补发一个 `voice` 事件兜底
        （用户明确点名要语音的场景）。
        """
        KEEPER.restore(message.session_id)
        state = TurnState(
            user_text=message.text,
            session_id=message.session_id,
            voice_mode=message.input_mode == "voice",
            user_image=message.image_url or "",
        )
        handle = self._turns.start(message.session_id)
        seq = itertools.count(1)
        display_parts = []
        voice_emitted = False

        try:
            yield {
                "type": "start",
                "turn_id": handle.turn_id,
                "session_id": message.session_id,
                "text": state.user_text,
            }

            # ① 安全预检
            await asyncio.to_thread(self._safety_precheck, state)
            self._check_cancel(handle)
            if not state.safety_passed:
                text = FallbackController().fallback_reply("blocked")
                state.final_reply = text
                yield {"type": "refuse", "text": text, "replace": True}
                yield {"type": "done", "full_text": text, "output_mode": "text", "image_path": ""}
                return

            # ② 感知（意图 + 情绪 + 潜台词 + 记忆召回）
            await asyncio.to_thread(self._perceive, state)
            # 逐轮判定要不要"认真想"（不是全局开关）
            state.think = _needs_thinking(state)
            self._check_cancel(handle)

            # ③ 分流：工具链 / 人设链
            if state.intent.intent in _TOOL_INTENTS:
                async for ev in self._astream_tool(state, handle, synthesize_voice, seq):
                    if ev.get("type") == "sentence":
                        display_parts.append(ev.get("text", ""))
                    elif ev.get("type") == "voice":
                        voice_emitted = True
                    yield ev
            else:
                self._check_cancel(handle)
                await asyncio.to_thread(self._compose, state)
                async for ev in self._astream_chat(state, handle, synthesize_voice, seq):
                    if ev.get("type") == "sentence":
                        display_parts.append(ev.get("text", ""))
                    elif ev.get("type") == "voice":
                        voice_emitted = True
                    yield ev

            # ④ 记忆写回（含取消检查点，保证迟到的取消不会写出半截账）
            self._check_cancel(handle)
            await asyncio.to_thread(self._writeback, state, handle)
            self._check_cancel(handle)

            # ⑤ _wants_voice 兜底：用户点名要语音但整轮没出 <voice>，用简短正文补一条
            if want_voice and not voice_emitted:
                spoken = "".join(display_parts).strip()
                if spoken:
                    ev = {"type": "voice", "seq": next(seq), "voice_text": spoken}
                    if synthesize_voice:
                        audio = await self._synthesize(spoken, handle)
                        if audio:
                            ev["audio_b64"] = base64.b64encode(audio).decode()
                    voice_emitted = True
                    yield ev

            # ⑥ 带图回复补一条 image 事件
            if state.output_mode == "image" and state.image_path:
                yield {"type": "image", "path": state.image_path}

            yield {
                "type": "done",
                "full_text": "".join(display_parts).strip(),
                "output_mode": state.output_mode or "text",
                "image_path": state.image_path or "",
            }
        except TurnCancelled:
            # 被更新的同一会话轮次取代：什么都不写（见 _writeback 注释）
            yield {"type": "cancelled", "reason": "superseded"}
        except _RefuseTurn as r:
            yield {"type": "refuse", "text": r.text, "replace": True}
            yield {"type": "done", "full_text": r.text, "output_mode": "text", "image_path": ""}
        except Exception as exc:
            yield {"type": "error", "code": "pipeline_error", "message": str(exc)}
        finally:
            self._turns.finish(message.session_id, handle.turn_id)

    # ---------- 人设链：流式生成 ----------
    async def _astream_chat(self, state, handle, synthesize_voice, seq, user_text=None, attempt=0):
        user_text = state.user_text if user_text is None else user_text
        extra = _VOICE_EMOTION_RULE if state.voice_mode else ""
        messages = build_chat_messages(
            state.prompt, _enrich_telegraph(user_text), extra=extra, user_image=state.user_image
        )
        streamer = _ReplyStreamer(state.voice_mode)
        raw_parts = []
        # 是否已经把"正文/语音内容"真正推给了上层。
        # 注意：不能用 streamer.emitted_content —— 那个是"装配器产出过片段"，
        # 一个片段只要被装配出来它就是真，哪怕紧接着就被审核拦下、根本没推出去；
        # 而审核重写的判据必须是"真推过"，否则首句被拦时会被误判成"已推过"直接降级。
        pushed = False
        try:
            async for chunk in services.get("llm").astream_chat(
                messages, enable_thinking=state.think
            ):
                self._check_cancel(handle)  # ① 每个 token 之后
                raw_parts.append(chunk)
                for frag in streamer.feed(chunk):
                    async for ev in self._emit_frag(frag, state, handle, synthesize_voice, seq):
                        if ev.get("type") in ("sentence", "voice"):
                            pushed = True
                        yield ev

            # 正常收尾：处理首句短缓冲（可能触发安全重写）
            state.reaction_tagged = streamer.reaction_tagged
            for frag in streamer.finish():
                if frag[0] == "held":
                    # 整轮只剩一句短话——此时一个 token 都还没推，重写是安全的。
                    #
                    # 判定分两层（见方案第六节）：
                    # ① 反应之后有没有下文——剥掉反应前缀后还有实质内容就放行，
                    #    不管多短（"？你说的是真的？"、"6，这也行？"都是正常形态）
                    # ② 只剩反应、没下文时，再看这一轮需不需要实质回应：
                    #    长输入 / 含明确提问 / 情绪求助 → 算敷衍，重写
                    #    短闲聊 / 小脾气 → 放行（"哼"是小脾气不是敷衍）
                    held_text = frag[1]
                    has_followup = _subst_count(held_text) > 1
                    if has_followup or state.reaction_tagged:
                        async for ev in self._emit_frags([("sentence", held_text)], state, handle, synthesize_voice, seq):
                            if ev.get("type") in ("sentence", "voice"):
                                pushed = True
                            yield ev
                        continue
                    needs_substance = (
                        len(state.user_text) >= int(load_app_config()["expression"].get("minimal_input_chars", 40))
                        or "?" in state.user_text or "？" in state.user_text
                        or state.comfort_mode
                    )
                    if not needs_substance:
                        # 闲聊/短陈述：极简放行，不算敷衍
                        async for ev in self._emit_frags([("sentence", held_text)], state, handle, synthesize_voice, seq):
                            if ev.get("type") in ("sentence", "voice"):
                                pushed = True
                            yield ev
                        continue
                    if attempt < _MAX_REWRITE:
                        async for ev in self._astream_chat(
                            state, handle, synthesize_voice, seq,
                            user_text=_too_short_input(user_text), attempt=attempt + 1,
                        ):
                            yield ev
                        return
                    raise _RefuseTurn(FallbackController().fallback_reply("review"))
                async for ev in self._emit_frag(frag, state, handle, synthesize_voice, seq):
                    if ev.get("type") in ("sentence", "voice"):
                        pushed = True
                    yield ev
            state.final_reply = "".join(raw_parts)
        except TurnCancelled:
            raise
        except _RefuseTurn:
            # 整轮降级（比如单字崩重写额度用尽）：直接交给上层，别当成"模型挂了"去跑兜底重发
            raise
        except _ReviewReject:
            # 句级 / <voice> 审核没过：_emit_frag 只负责报告，这里按"统一规则"裁决：
            if pushed:
                # 已经推过内容 → 撤不回，整轮降级成兜底话（与改造前行为一致）
                raise _RefuseTurn(FallbackController().fallback_reply("review"))
            if attempt < _MAX_REWRITE:
                # 一个 token 都还没推 → 换个干净说法重跑一次。
                # 与单字崩那条重写是同一条机制：共用 attempt 计数与 _MAX_REWRITE 上限。
                async for ev in self._astream_chat(
                    state, handle, synthesize_voice, seq,
                    user_text=_clean_review_input(user_text), attempt=attempt + 1,
                ):
                    yield ev
                return
            # 一个 token 都没推、重写额度也用尽 → 整轮降级
            raise _RefuseTurn(FallbackController().fallback_reply("review"))
        except Exception:
            # 主/备模型都挂了，或中途断流
            if streamer.emitted_content:
                # 已经推过内容：接不上，按现有内容收尾
                async for ev in self._emit_frags(streamer.finish(), state, handle, synthesize_voice, seq):
                    yield ev
                state.final_reply = "".join(raw_parts)
                return
            # 一个 token 都没推：退回 generate() 的兜底话（含人设覆盖），当成整段重发
            fallback = await asyncio.to_thread(
                generate, state.prompt, _enrich_telegraph(user_text), extra, state.user_image
            )
            streamer2 = _ReplyStreamer(state.voice_mode)
            async for ev in self._emit_frags(
                streamer2.feed(fallback) + streamer2.finish(), state, handle, synthesize_voice, seq
            ):
                yield ev
            state.final_reply = fallback
            return

    async def _emit_frags(self, frags, state, handle, synthesize_voice, seq):
        """把一批片段依次转成事件；出现既不该被推出去的 held 就当成普通句子（兜底）。"""
        for frag in frags:
            if frag[0] == "held":
                frag = ("sentence", frag[1])
            async for ev in self._emit_frag(frag, state, handle, synthesize_voice, seq):
                yield ev

    # ---------- 工具链：完整文本 -> 同一套句子装配器 ----------
    async def _astream_tool(self, state, handle, synthesize_voice, seq):
        # 注意：工具链不做审核重写，与 chat 链的差异是有意的。
        # 工具链的文本来自工具结果（天气/搜索结果等）拼装，不是模型自由发挥，
        # 命中审核只走"整轮降级"（由 _emit_frag 抛 _RefuseTurn），不重跑工具。
        await asyncio.to_thread(self._toolcall, state, handle)
        self._check_cancel(handle)
        if not state.final_reply:
            await asyncio.to_thread(self._respond, state)
            self._check_cancel(handle)
        text = state.final_reply or state.draft_reply
        streamer = _ReplyStreamer(state.voice_mode)
        for frag in streamer.feed(text) + streamer.finish():
            if frag[0] == "held":
                # 工具链路不做整轮重写（与 chat 路径一致的流式取舍）：直接当一句交给审核
                frag = ("sentence", frag[1])
            async for ev in self._emit_frag(frag, state, handle, synthesize_voice, seq):
                yield ev
        state.final_reply = text

    # ---------- 片段 -> 事件 ----------
    async def _emit_frag(self, frag, state, handle, synthesize_voice, seq):
        kind = frag[0]
        if kind == "emotion":
            yield {"type": "emotion", "emotion": frag[1], "desc": frag[2] or ""}
            return
        if kind == "held":
            # 正常不会走到这（_astream_chat 收尾时已拦下裁决）；兜底当普通句子
            frag = ("sentence", frag[1])
            kind = "sentence"
        if kind == "sentence":
            text = frag[1]
            if not text:
                return
            ok, _reason = await asyncio.to_thread(SafetyReviewer().review, text, "output")
            if not ok:
                # 只报告"没过"，怎么处理交给 _astream_chat 按统一规则裁决：
                # 还没推过内容 → 安全重写；已经推过 → 整轮降级。这里不自己拍板。
                raise _ReviewReject(text)
            self._check_cancel(handle)  # ② 准备推一句之前
            yield {"type": "sentence", "seq": next(seq), "text": text}
            return
        if kind == "voice":
            vtext = frag[1]
            # 语音内容同样要过审核：万一模型把违规话塞进 <voice>，绝不能拿去合成/推送。
            # 先审、后合成——审不过直接抛给上层裁决（与句子一致）。
            ok, _reason = await asyncio.to_thread(SafetyReviewer().review, vtext, "output")
            if not ok:
                raise _ReviewReject(vtext)
            ev = {"type": "voice", "seq": next(seq), "voice_text": vtext}
            if synthesize_voice and vtext.strip():
                audio = await self._synthesize(vtext, handle)
                if audio:
                    ev["audio_b64"] = base64.b64encode(audio).decode()
            yield ev
            return

    async def _synthesize(self, text: str, handle) -> bytes:
        """合成语音（dashscope 同步调用，扔线程池）。失败只是少声音，不打断流程。"""
        self._check_cancel(handle)
        try:
            return await asyncio.to_thread(services.get("tts").synthesize, text)
        except Exception:
            return b""

    @staticmethod
    def _check_cancel(handle) -> None:
        """协作式取消检查点：被顶掉就抛 TurnCancelled 让上层收场。"""
        if handle is not None and handle.is_cancelled():
            raise TurnCancelled(f"turn {handle.turn_id} cancelled")

    # ================= 各步：函数体与原 graph.py 节点逐条对齐 =================
    def _safety_precheck(self, state: TurnState) -> None:
        """安全预检。用户的话先过一遍黑名单。"""
        reviewer = SafetyReviewer()
        ok, _reason = reviewer.review(state.user_text, mode="input")
        state.safety_passed = ok

    def _perceive(self, state: TurnState) -> None:
        """感知 + 记忆召回并行。

        意图/情绪（一次 LLM 调用）和记忆召回（两次向量检索）互不依赖，
        串行等于把两条网络延迟加起来，回复慢一大截——扔线程池里同时跑。
        """
        pipeline = PerceptionPipeline()
        recent = KEEPER.get_context(state.session_id)[-4:]
        with ThreadPoolExecutor(max_workers=2) as pool:
            ft = pool.submit(pipeline.run, state.user_text, recent)
            fr = pool.submit(MemoryRecaller().recall, state.user_text, state.session_id)
            intent, emotion, subtext = ft.result()
            state.memory = fr.result()
        state.intent = intent
        state.emotion = emotion
        state.subtext = subtext
        # 危机信号或用户明说要安慰，都切安抚模式
        state.comfort_mode = bool(emotion.is_crisis) or intent.intent == "comfort"

    def _compose(self, state: TurnState) -> None:
        """人设组装。安抚模式换 comfort 语气，正常轮先掷一次"小动作骰子"。"""
        mode = "comfort" if state.comfort_mode else "chat"
        memory = state.memory or MemoryBundle()
        rel = memory.relationship or {}
        quirk = QuirkDirector().roll(
            memory,
            stage=rel.get("stage", "初识"),
            mood=rel.get("mood", "平常"),
            comfort_mode=state.comfort_mode,
        )
        emotion = state.emotion
        state.prompt = _ENGINE.compose(
            mode,
            memory,
            state.session_id,
            emotion_label=emotion.emotion if emotion else "",
            quirk=quirk,
            user_text=state.user_text,
            subtext=state.subtext,
        )

    def _toolcall(self, state: TurnState, handle) -> None:
        """工具管线。天气缺城市时先问人，别瞎查默认城市。"""
        intent = state.intent
        # 工具链路不走 recall 节点，档案直接从库里读，城市才兜底得上
        profile = {}
        try:
            profile = services.get("kv_store").read("profile", state.session_id) or {}
        except Exception:
            pass
        fallback_city = profile.get("city", "")

        # 天气意图但用户没提城市、档案里也没有：直接追问，不瞎查
        if intent.intent == "weather" and not fallback_city and not _has_city(state.user_text):
            state.final_reply = InfoGapCoordinator().ask("city", state.session_id)
            return

        out = ToolCallOrchestrator().run(
            state.user_text,
            intent.intent,
            fallback_city,
            session_id=state.session_id,
            should_cancel=(handle.is_cancelled if handle is not None else None),
        )
        if out.get("cancelled"):
            self._check_cancel(handle)  # 抛 TurnCancelled，交给上层收场
        state.tool_results = out["results"]
        # 模型自己先给的答案存成初稿，respond 步再拿去润成人设口吻
        state.draft_reply = out["final_text"]

    def _respond(self, state: TurnState) -> None:
        """把工具结果转述成人话。模型初答只是参考，要重说一遍。"""
        extra = _VOICE_EMOTION_RULE if state.voice_mode else ""
        reply = persona_wrap(
            state.tool_results or [],
            _enrich_telegraph(state.user_text),
            initial_text=state.draft_reply,
            extra=extra,
        )
        state.final_reply = reply.text
        state.output_mode = reply.output_mode
        state.image_path = reply.image_path

    def _writeback(self, state: TurnState, handle) -> None:
        """记忆写回。短期记忆、长期沉淀、亲密度、画像，一样别落。

        取消检查点插在每个子步骤之前：迟到的取消（被新轮顶掉）不会写出半截账。
        注意：一旦取消，本方法**什么都不写**——不写 KEEPER、不写聊天记录、
        不蒸馏、不涨亲密度、不刷画像。这偏离了设计文档里
        "chat_log 可补一条 cancelled 标记"的说法，是**有意选择**：
        写任何东西都会让"这轮当没发生过"不彻底，而 chat_log 的两个用途
        （写日记、重启后恢复短期记忆）都不需要一条中途被丢弃的痕迹。
        实际也很少出现"取消但没被新轮接管"——TurnRegistry.start 的会话级
        单轮不变式会让新轮先取消旧轮，旧轮本来就不该留痕。
        """
        session_id = state.session_id
        user_text = state.user_text
        # 普通聊天的回复还停在初稿里，先定稿再写回
        reply = state.final_reply or state.draft_reply
        state.final_reply = reply
        # 语音轮：短期记忆和聊天记录只收剥掉情绪标记的正文；
        # final_reply 保持原样（带标签），上层拿它去合成语音才有情绪可拆
        clean_reply = strip_emotion_marks(reply) if state.voice_mode else reply
        emotion = state.emotion

        # 短期记忆追加这一轮的问答
        self._check_cancel(handle)
        KEEPER.append(session_id, "user", user_text)
        if clean_reply:
            KEEPER.append(session_id, "assistant", clean_reply)

        # 聊天记录落库（chat_log 表）：日记生成、重启恢复都靠这份数据
        # 危机轮也要记——对话发生过就该留痕，只是不涨亲密度不沉淀记忆
        self._check_cancel(handle)
        intent = state.intent
        emotion_label = emotion.emotion if emotion else ""
        kv = services.get("kv_store")
        kv.write(
            "session",
            session_id,
            {
                "role": "user",
                "text": user_text,
                "intent": intent.intent if intent else "",
                "emotion": emotion_label,
                "mode": "voice" if state.voice_mode else "text",
            },
        )
        if clean_reply:
            kv.write(
                "session",
                session_id,
                {
                    "role": "assistant",
                    "text": clean_reply,
                    "intent": intent.intent if intent else "",
                    "emotion": emotion_label,
                    "mode": state.output_mode,
                },
            )

        # 危机或婉拒的轮次不沉淀、不涨亲密度，这些轮次不算正常互动
        if emotion and emotion.is_crisis:
            return

        self._check_cancel(handle)
        distiller = ConversationDistiller()
        distiller.distill_turn(session_id, user_text)

        self._check_cancel(handle)
        tracker = RelationshipTracker()
        rel = tracker.update(session_id, emotion.emotion if emotion else "neutral")

        # 心情状态机：数值之外，角色此刻的情绪也往前走一格，下轮 compose 要用
        rel["mood"] = MoodEngine().update(
            rel, emotion.emotion if emotion else "neutral", state.comfort_mode
        )
        try:
            kv.write("relationship", session_id, rel)
        except Exception:
            pass

        # 聊够几轮才重新画像，别一句话就给人家贴标签
        self._check_cancel(handle)
        interval = _portrait_interval()
        if rel.get("interaction_count", 0) % interval == 0:
            PortraitBuilder().refresh(session_id, KEEPER.get_context(session_id)[-6:])
