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
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from shared.types import (
    EmotionResult,
    ExternalServiceError,
    FinalReply,
    InputMessage,
    IntentResult,
    MemoryBundle,
    PipelineInternalError,
    PerceptionExtras,
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
from capability.quirks import QuirkDirector
from capability import self_identity
from capability.response_generator import (
    build_chat_messages,
    fallback_line,
    generate,
    persona_wrap,
)
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
# 首句短缓冲阈值：首句去标点后不超过这么多实质字符就暂缓推出，等下一句裁决
_FIRST_HOLD_MAX = 4
# <voice> / </voice> 标签
_VOICE_OPEN = "<voice>"
_VOICE_CLOSE = "</voice>"
# 反应前缀与强切长度不再硬编码：以前 config.json 里的 reaction_tag / force_cut_chars
# 是摆设（代码用常量，改配置不生效）。现在统一从 expression 配置读，读不到落默认值。


def _expression_cfg() -> dict:
    """expression 配置段的安全读取：读不到给空 dict，调用方各自兜底。"""
    try:
        return load_app_config().get("expression", {}) or {}
    except Exception:
        return {}


def _strip_reaction_tag(text: str) -> str:
    """剥掉回复开头的反应前缀（@r 之类），落库和短期记忆里不许留标记。

    _ReplyStreamer 推送前会剥一次，但 state.final_reply 存的是模型原始输出——
    写回前必须再过一遍，否则标记会进 chat_log 和短期记忆污染后续上下文。
    """
    tag = (_expression_cfg().get("reaction_tag") or "@r").strip()
    t = (text or "").lstrip()
    if t.startswith(tag):
        return t[len(tag):].lstrip()
    return text


def _log_exc(where: str) -> None:
    """把当前异常连 traceback 记下来（J2）。

    为什么必须有：本文件的泛型 `except Exception` 以前是**完全静默**的。
    A2 的 5 处裸 `session_id` NameError 就是被它吞掉的——任何编程错误都会被
    转成一次兜底重发，看起来像"模型挂了"，查不出真因。
    这也是 A1（日记子系统静默死亡）能隐身一整个提交周期的同一个病。

    优先走 services 里的 Logger（J1 之后带 RotatingFileHandler，能落盘事后查）；
    拿不到就退回 stderr。**这个函数自己绝不能抛异常**——它只在错误路径上被调用，
    再抛一次就把原始异常盖掉了。
    """
    tb = traceback.format_exc()
    try:
        lg = services.get("logger")
        if lg is not None and hasattr(lg, "error"):
            lg.error(f"[pipeline] {where}:\n{tb}")
            return
    except Exception:
        pass
    print(f"[pipeline] {where} 异常：\n{tb}", file=sys.stderr)


class _ReviewReject(Exception):
    """审核没过（句子或 <voice> 语音内容）。

    这一层只负责"报告没过"，**不自己决定**怎么处理——由上层按统一规则裁决：
    还没推出任何内容 → 安全重写（最多 _MAX_REWRITE 次）；已经推过 → 整轮降级。

    R27a：异常里**只带有限原因码，绝不带被拦原文**。以前 raise _ReviewReject(text)
    让 str(exc) 一路漏进 state.review_block_reason → refuse.reason 推给前端——
    等于把要拦的内容换个键原样发出去。原因码（review_output_blocked）要能和
    服务失败（pipeline_error）、输入预检（"输入未过安全预检"）区分开。
    """

    def __init__(self, reason_code: str = "review_output_blocked"):
        super().__init__(reason_code)
        self.reason_code = reason_code


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
    extras: Optional[PerceptionExtras] = None  # 感知附加产出（心事/语气反馈，S2）
    comfort_mode: bool = False
    think: bool = False              # 本轮是否触发"认真想"（按输入复杂度逐轮判定）
    reaction_tagged: bool = False    # 模型自标了 @r（这一句是本能反应）
    memory: Optional[MemoryBundle] = None
    prompt: Optional[PromptPackage] = None
    draft_reply: str = ""
    tool_results: list = field(default_factory=list)
    rewrite_count: int = 0
    review_passed: bool = False
    review_block_reason: str = ""   # 本轮被审核/降级拦下的原因（refuse 事件带出，排查误杀用）
    was_poor: bool = False          # 本轮触发了敷衍/违规重写（写回时落账，下一轮强制"认真想"）
    final_reply: str = ""
    # R14a 正文契约：canonical 是唯一正式正文（审核通过片段的拼接，剥净协议），
    # voice_text 是 <voice> 语音派生（单独走，不拼回正文）。final_reply 在
    # R17b 写回迁移前仍是既有路径的载体，但两段拼接已终结。
    canonical_text: str = ""
    voice_text: str = ""
    review_status: str = "accepted"  # accepted / rejected / unavailable
    reason_code: str = ""            # 有限原因码（复用 R27 体系）
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
    # 配置里的标签先过一遍白名单：历史上混进过"委屈"这种非法标签，永远匹配不上
    valid_emotions = ("neutral", "happy", "sad", "angry", "tired", "anxious", "crisis")
    never_emos = {e for e in (cfg.get("never_on_emotions") or []) if e in valid_emotions}
    if emo in never_emos:
        return False
    # 短情绪消息：也不要想（比如"我今天好难过"——要的是陪伴）
    if emo in ("sad", "tired") and _subst_count(state.user_text) < 20:
        return False
    # force_if_last_poor（F8 接线）：上一轮被判敷衍/出戏，这一轮强制认真想。
    # 该配置以前是死配置，提交信息声称实现了实际没有。
    if cfg.get("force_if_last_poor", True):
        rel = (state.memory.relationship if state.memory else None) or {}
        if rel.get("last_poor"):
            return True
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


def _repeat_input(user_text: str) -> str:
    """逐字复读守卫（S12-1）的重写提示：和上一条一字不差 = 大概率复读。"""
    return (
        f"{user_text}\n"
        "（系统提醒：你刚才的回复和上一条一字不差，他在等你新的回应——"
        "换一种说法，顺着对话往前走，禁止复读。）"
    )


def _last_assistant_first_sentence(session_id: str) -> str:
    """上一条她说过的话里的第一句实质句（复读守卫的比较基准）。取不到返回空。"""
    try:
        context = KEEPER.get_context(session_id)
    except Exception:
        return ""
    for msg in reversed(context):
        if msg.get("role") != "assistant":
            continue
        text = (msg.get("content") or "").strip()
        for piece in re.split(r"[。！？\n]", text):
            piece = piece.strip()
            if _subst_count(piece) >= 2:
                return piece
        return ""
    return ""


# 后台写回任务（真机反馈："回复很久"）：写回里的 LLM 重活（画像刷新、身份冻结）
# 原来阻塞在 done 事件之前——她的话说完了，你还要等她"做完笔记"才看到完成。
# 挪到后台后 done 立刻到；强引用集合防 GC（S12-5），done 回调里查未捕获异常。
_BG_TASKS: set = set()


def _on_bg_done(task) -> None:
    _BG_TASKS.discard(task)
    if not task.cancelled():
        exc = task.exception()
        if exc is not None:
            print(f"[bg] 后台写回异常: {type(exc).__name__}: {exc}")


def _schedule_bg_writeback(deferred) -> None:
    """把 writeback 返回的后台待办逐个调度（必须在持有 running loop 的一侧调用）。"""
    for fn in deferred or []:
        task = asyncio.create_task(asyncio.to_thread(fn))
        _BG_TASKS.add(task)
        task.add_done_callback(_on_bg_done)


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
        cfg = _expression_cfg()
        self.emitted_content = False  # 是否已产出过 sentence/voice
        self.reaction_tagged = False  # 模型自标了反应前缀：这一句是本能反应
        self._tag_checked = False     # 是否已经检查过开头的反应前缀
        self._reaction_tag = (cfg.get("reaction_tag") or "@r").strip() or "@r"
        self._force_cut = int(cfg.get("force_cut_chars", 60) or 60)
        self._emotion_done = not voice_mode
        self._emo_buf = ""
        self._raw = ""  # 待处理原始文本（可能含被截断的标签）
        self._in_voice = False
        self._voice_buf = ""
        self._text_buf = ""
        self._held = None  # 首句短缓冲

    # ---- 反应标记（@r 之类，配置可改）：只看每个片段的开头，剥掉并登记 ----
    def _maybe_strip_tag(self, s: str) -> str:
        if self._tag_checked or self.reaction_tagged:
            return s
        self._tag_checked = True
        t = s.lstrip()
        if t.startswith(self._reaction_tag):
            self.reaction_tagged = True
            return t[len(self._reaction_tag):].lstrip()
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
        # <voice> 的内容独立剥一遍反应前缀：_maybe_strip_tag 带一次性开关
        # （_tag_checked），正文先跑会把它用掉，轮到 <voice> 时直接放行，
        # 导致 "你好。<voice>@r 你好呀</voice>" 里的 @r 进 voice_text 被朗读。
        # 这里改用无状态的 _strip_reaction_tag（只看开头），不受开关限制；
        # 真剥到了就登记 reaction_tagged（"模型自标了本能反应"的信号，上层要用）。
        stripped = _strip_reaction_tag(v)
        if stripped != v:
            self.reaction_tagged = True
            v = stripped
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
        while not self._in_voice and len(self._text_buf) >= self._force_cut:
            cut = self._text_buf[:self._force_cut]
            self._text_buf = self._text_buf[self._force_cut:]
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
        error_code = ""  # R27b：error 事件的有限错误码，循环后统一裁决
        refused = False        # R14a：审核拒绝（refuse 事件）→ FinalReply.review_status
        refuse_reason = ""     # R14a：有限原因码（refuse.reason，已是原因码体系）

        # astream 内部 finally 会调 TURN_REGISTRY.finish（详见 CLAUDE.md 关键单例）；
        # 显式 try/finally 保证 error 分支提前 return 时也立即收尾，
        # 避免靠 GC 终结器兜底让登记项短暂滞留。
        _astream = self.astream(message, synthesize_voice=False, want_voice=False)
        try:
            async for ev in _astream:
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
                    refused = True
                    refuse_reason = ev.get("reason") or ""
                elif t == "done":
                    output_mode = ev.get("output_mode") or output_mode
                    image_path = ev.get("image_path") or image_path
                elif t == "cancelled":
                    cancelled = True
                elif t == "error":
                    # R27b：internal_error 必须保留身份冒泡，绝不许在这里重新包成
                    # "看似成功"的 FinalReply 兜底；外部降级才允许落 llm 兜底话
                    error_code = ev.get("code") or "pipeline_error"
        finally:
            await _astream.aclose()

        if cancelled:
            # 取消不当异常兜底成 llm 兜底话：这轮当没发生过，给个空回复
            return FinalReply(text="")

        if error_code == "internal_error":
            # R27b：本地缺陷冒泡到请求边界——接口层映射成安全的 internal_error
            # 响应。绝不重新包成"看似成功"的 FinalReply 兜底（bug 会变成
            # "她说了句奇怪的话"还被写进记忆，A2 事故的复发路径）。
            raise PipelineInternalError("internal_error")
        if error_code:
            # 外部服务失败（external_degraded / 兼容旧 pipeline_error）：允许的
            # 降级——llm 兜底话，聊天不断。审核结果 = 原生成内容不可用（R14a）
            return FinalReply(
                text=FallbackController().fallback_reply("llm"),
                review_status="unavailable",
                reason_code=error_code,
            )

        frags.sort(key=lambda x: x[0])
        # R14a：正文与语音派生**不再拼回一段**——canonical 唯一正文，
        # voice_text 独立载体（旧协议"text 里塞 <voice> 再让 renderer 拆"终止）
        canonical_parts = []
        voice_parts = []
        for _seq, kind, c in frags:
            if kind == "voice":
                voice_parts.append(c)
            else:
                canonical_parts.append(c)
        body = "".join(canonical_parts)
        prefix = f"[{emotion}]" + (f"（{desc}）" if desc else "") if emotion else ""
        review_status = "rejected" if refused else "accepted"
        return FinalReply(
            text=prefix + body,
            output_mode=output_mode or "text",
            image_path=image_path or "",
            voice_text="".join(voice_parts),
            review_status=review_status,
            reason_code=refuse_reason or "",
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
        # I12：restore 内部是 SQLite 读（冷启动首轮要把整段会话记录捞回来），
        # 同文件其他慢调用都进了 to_thread，唯独这里以前漏了——直接在事件循环上同步跑，
        # 期间 SSE 推送、poll、语音通道全都卡住。
        await asyncio.to_thread(KEEPER.restore, message.session_id)
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
                text = fallback_line("blocked", state.session_id, default=FallbackController().fallback_reply("blocked"))
                state.final_reply = text
                state.review_block_reason = "输入未过安全预检"
                yield {"type": "refuse", "text": text, "replace": True, "reason": state.review_block_reason}
                yield {"type": "done", "full_text": text, "output_mode": "text", "image_path": ""}
                return

            # ② 感知（意图 + 情绪 + 潜台词 + 记忆召回）
            await asyncio.to_thread(self._perceive, state)
            # 逐轮判定要不要"认真想"（不是全局开关）
            state.think = _needs_thinking(state)
            self._check_cancel(handle)

            # ③ 分流：工具链 / 人设链
            canonical_parts = []  # R14a：审核通过片段的拼接 = canonical（唯一正文）
            voice_parts = []      # R14a：<voice> 语音派生单独走，不拼回正文
            if state.intent.intent in _TOOL_INTENTS:
                async for ev in self._astream_tool(state, handle, synthesize_voice, seq):
                    if ev.get("type") == "sentence":
                        display_parts.append(ev.get("text", ""))
                        canonical_parts.append(ev.get("text", ""))
                    elif ev.get("type") == "voice":
                        voice_emitted = True
                        voice_parts.append(ev.get("voice_text", ""))
                    yield ev
            else:
                self._check_cancel(handle)
                await asyncio.to_thread(self._compose, state)
                async for ev in self._astream_chat(state, handle, synthesize_voice, seq):
                    if ev.get("type") == "sentence":
                        display_parts.append(ev.get("text", ""))
                        canonical_parts.append(ev.get("text", ""))
                    elif ev.get("type") == "voice":
                        voice_emitted = True
                        voice_parts.append(ev.get("voice_text", ""))
                    yield ev

            # R14a：入口解析一次的产出在此定稿——canonical 唯一正文、voice
            # 独立载体（都是审核通过的片段，被拒片段根本不会成为事件）。
            # final_reply 同步收敛为 canonical：draft/raw 串（含 <voice>）从此
            # 失去持久化资格——写回落库的只能是唯一正文
            state.canonical_text = "".join(canonical_parts)
            state.voice_text = "".join(voice_parts)
            if state.canonical_text:
                state.final_reply = state.canonical_text

            # ④ 记忆写回（含取消检查点，保证迟到的取消不会写出半截账）。
            # 写回跑在工作线程（无事件循环），它**不自己调度**后台待办，
            # 而是把闭包列表带回来由这里（持有 loop 的一方）调度——I11
            deferred = await asyncio.to_thread(self._writeback, state, handle)
            self._check_cancel(handle)
            _schedule_bg_writeback(deferred)

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
            # I13：refuse 降级也跑写回，让 was_poor 落 relationship.last_poor，否则
            # force_if_last_poor 在降级后的下一轮永远不触发。_writeback 内部
            # 检测 final_reply/draft_reply 都空时走 refuse-only 分支（只写关系层），
            # 不写 KEEPER / chat_log / 蒸馏 / 画像——refuse 没说过话。
            deferred = await asyncio.to_thread(self._writeback, state, handle)
            self._check_cancel(handle)
            _schedule_bg_writeback(deferred)
            yield {"type": "refuse", "text": r.text, "replace": True,
                   "reason": state.review_block_reason}
            yield {"type": "done", "full_text": r.text, "output_mode": "text", "image_path": ""}
        except ExternalServiceError as exc:
            # R27b：外部服务失败——允许的降级路径。细节（来源/原因码）进日志，
            # 对外只给固定文案 + 有限错误码（R27a 的输出边界规则继续适用）。
            _log_exc(f"astream 外部服务失败（{exc.source}/{exc.reason_code}）")
            yield {"type": "error", "code": "external_degraded",
                   "message": "这轮没接上，稍后再试试"}
        except Exception:
            # R27b：本地缺陷（NameError/TypeError/装配错误…）——生命周期失败：
            # 记录 traceback 与关联 ID，对外只给安全的 internal_error。
            # 不重跑模型/付费工具、不作为学习轮保存：写回只发生在 try 路径与
            # _RefuseTurn 分支，走到这里说明本轮什么业务提交都没发生。
            # CancelledError/TurnCancelled/GeneratorExit 继承 BaseException 或已在
            # 上方单独捕获，控制流不会被这条盖掉。
            _log_exc("astream 内部错误（internal_error）")
            yield {"type": "error", "code": "internal_error",
                   "message": "这轮没接上，稍后再试试"}
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
        prev_first = _last_assistant_first_sentence(state.session_id)
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
                    # 逐字复读守卫（S12-1）：第一句实质句和上一条回复一字不差 →
                    # 大概率是复读。趁一个字都还没推出去，注入纠正重说一次。
                    # 上一条的原文还在 KEEPER 里没被本轮覆盖，比较基准稳定；
                    # attempt 用尽就放行（不无限纠错）
                    if (
                        not pushed and frag[0] == "sentence" and prev_first
                        and attempt < _MAX_REWRITE
                        and frag[1].strip() == prev_first
                    ):
                        async for ev2 in self._astream_chat(
                            state, handle, synthesize_voice, seq,
                            user_text=_repeat_input(user_text), attempt=attempt + 1,
                        ):
                            yield ev2
                        return
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
                        pushed, evs = await self._flush_held(held_text, state, handle, synthesize_voice, seq)
                        for ev in evs:
                            yield ev
                        continue
                    needs_substance = (
                        len(state.user_text) >= int(load_app_config()["expression"].get("minimal_input_chars", 40))
                        or "?" in state.user_text or "？" in state.user_text
                        or state.comfort_mode
                    )
                    if not needs_substance:
                        # 闲聊/短陈述：极简放行，不算敷衍
                        pushed, evs = await self._flush_held(held_text, state, handle, synthesize_voice, seq)
                        for ev in evs:
                            yield ev
                        continue
                    # minimal_allowlist 白名单（F8 接线）：配置里点名的极简回复
                    # （"6"、"？"、"神了"……）正向放行——以前这串配置是从没被读过的摆设
                    allowlist = set(_expression_cfg().get("minimal_allowlist") or [])
                    if held_text.strip() in allowlist:
                        pushed, evs = await self._flush_held(held_text, state, handle, synthesize_voice, seq)
                        for ev in evs:
                            yield ev
                        continue
                    if attempt < _MAX_REWRITE:
                        state.was_poor = True  # 本轮敷衍，下一轮强制认真想（force_if_last_poor）
                        async for ev in self._astream_chat(
                            state, handle, synthesize_voice, seq,
                            user_text=_too_short_input(user_text), attempt=attempt + 1,
                        ):
                            yield ev
                        return
                    state.was_poor = True
                    raise _RefuseTurn(fallback_line("review", state.session_id, default=FallbackController().fallback_reply("review")))
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
        except _ReviewReject as exc:
            # 句级 / <voice> 审核没过：_emit_frag 只负责报告，这里按"统一规则"裁决：
            # R27a：只带出有限原因码（不带被拦原文），refuse 事件可排查误杀类别
            state.review_block_reason = exc.reason_code
            if pushed:
                # 已经推过内容 → 撤不回，整轮降级成兜底话（与改造前行为一致）
                state.was_poor = True
                raise _RefuseTurn(fallback_line("review", state.session_id, default=FallbackController().fallback_reply("review")))
            if attempt < _MAX_REWRITE:
                # 一个 token 都还没推 → 换个干净说法重跑一次。
                # 与单字崩那条重写是同一条机制：共用 attempt 计数与 _MAX_REWRITE 上限。
                state.was_poor = True
                async for ev in self._astream_chat(
                    state, handle, synthesize_voice, seq,
                    user_text=_clean_review_input(user_text), attempt=attempt + 1,
                ):
                    yield ev
                return
            # 一个 token 都没推、重写额度也用尽 → 整轮降级
            state.was_poor = True
            raise _RefuseTurn(fallback_line("review", state.session_id, default=FallbackController().fallback_reply("review")))
        except ExternalServiceError as exc:
            # 主/备模型都挂了，或中途断流（R27b：只有窄载体才配走降级——
            # llm_client 的 SDK 边界已把网络/超时/协议故障统一转成这个类型）。
            # J2：**这条路径以前完全静默**——A2 的 5 处裸 `session_id` NameError 就是被它
            # 吞掉的：任何编程错误都会被转成一次兜底重发，看起来像"模型挂了"，
            # 而兜底文本还会被写回记忆，污染"她说过什么"。降级行为本身是对的
            # （兜底哲学：聊天不能断），但**不能哑**——至少要留下来源与 traceback。
            _log_exc(f"_astream_chat 外部服务失败降级（{exc.source}/{exc.reason_code}）")
            if streamer.emitted_content:
                # 已经推过内容：接不上，按现有内容收尾
                async for ev in self._emit_frags(streamer.finish(), state, handle, synthesize_voice, seq):
                    yield ev
                state.final_reply = "".join(raw_parts)
                return
            # 一个 token 都没推：退回 generate() 的兜底话（含人设覆盖），当成整段重发。
            # generate() 若也抛 ExternalServiceError（备用全挂），向上交给 astream 分流；
            # 它若抛本地缺陷，同 upward——两者都不许在这里被吞。
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

    async def _flush_held(self, held_text, state, handle, synthesize_voice, seq):
        """放行 held 短句：emit 一批 sentence/voice 事件，返回 (是否推出, 事件列表)。

        I13 抽取：原 _astream_chat 三处相同的四行样板合并到这里——事故性复杂度
        而非有意重复。pushed 只在确实产出 sentence/voice 时置位，由调用方
        unpacking 返回值后再 yield。
        """
        pushed = False
        out = []
        async for ev in self._emit_frags([("sentence", held_text)], state, handle, synthesize_voice, seq):
            if ev.get("type") in ("sentence", "voice"):
                pushed = True
            out.append(ev)
        return pushed, out

    async def _emit_frags(self, frags, state, handle, synthesize_voice, seq):
        """把一批片段依次转成事件；出现既不该被推出去的 held 就当成普通句子（兜底）。"""
        for frag in frags:
            if frag[0] == "held":
                frag = ("sentence", frag[1])
            async for ev in self._emit_frag(frag, state, handle, synthesize_voice, seq):
                yield ev

    # ---------- 工具链：完整文本 -> 同一套句子装配器 ----------
    async def _astream_tool(self, state, handle, synthesize_voice, seq):
        # 注意：工具链不做审核**重写**，与 chat 链的差异是有意的（CLAUDE.md 取舍 #2）。
        # 但它**照样过审**——`_emit_frag` 对每一句都跑 SafetyReviewer。
        # 差别只在裁决：chat 链没推过内容时还能重写，工具链一律"整轮降级"，
        # 因为工具文本是结果拼装、重写等于让它再犯一次同样的错，而且工具贵、有副作用。
        #
        # C4 修的坑：`_emit_frag` 抛的是 `_ReviewReject`（它只负责"报告没过"，不自己拍板），
        # 而这里以前不接它 → 一路冒到 astream 的泛型 except → 变成 `error` 事件，
        # 且**被审核拦下的原话会随 `str(exc)` 推给前端**。审核拦的东西不该给用户看见。
        await asyncio.to_thread(self._toolcall, state, handle)
        self._check_cancel(handle)
        if not state.final_reply:
            await asyncio.to_thread(self._respond, state)
            self._check_cancel(handle)
        text = state.final_reply or state.draft_reply
        streamer = _ReplyStreamer(state.voice_mode)
        try:
            for frag in streamer.feed(text) + streamer.finish():
                if frag[0] == "held":
                    # 工具链路不做整轮重写（与 chat 路径一致的流式取舍）：直接当一句交给审核
                    frag = ("sentence", frag[1])
                async for ev in self._emit_frag(frag, state, handle, synthesize_voice, seq):
                    yield ev
        except _ReviewReject:
            # R02 止损：工具链审核拒绝 = 原稿彻底失去写回资格。final_reply /
            # draft_reply 在这里清空，_writeback 才会走 refuse-only 分支
            # （只记质量标记，不写 KEEPER / chat_log / 蒸馏 / 画像 / 身份）。
            # 以前靠"final_reply 非空"决定可保存，而 _respond 早在审核**之前**
            # 就把转述稿塞进了 final_reply——被拒原稿就这么混进了她的记忆
            # （隔离用例实测：KEEPER 与 chat_log 各混入一份被拒标记）。
            # 副作用工具不重跑：本分支本就不再执行任何工具（I3 幂等账本也在
            # 工具层拦着重跑）；被拒草稿不进日志——异常本身不外抛、不带原文。
            state.final_reply = ""
            state.draft_reply = ""
            # 整轮降级：不重写、不重跑工具，换成一句人设化的兜底话。
            # 已经推出去的句子撤不回（前端按 refuse 的 replace 语义整轮替换）。
            raise _RefuseTurn(fallback_line(
                "review", state.session_id,
                default=FallbackController().fallback_reply("review"),
            ))
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
            # 空串与纯空白（"\n"）都不推：模型分段时，空行会在 \n 边界上被切出
            # "内容只有换行"的句子，推给前端就是空气泡（真机截图实证过）。
            # 注意 held 缓冲被推出时也走这里，一并被挡，不会丢正常内容。
            if not text or not text.strip():
                return
            ok, _reason = await asyncio.to_thread(SafetyReviewer().review, text, "output")
            if not ok:
                # 只报告"没过"，怎么处理交给 _astream_chat 按统一规则裁决：
                # 还没推过内容 → 安全重写；已经推过 → 整轮降级。这里不自己拍板。
                # R27a：异常只携带原因码，被拦原文不进异常链（refuse.reason 是对外字段）
                raise _ReviewReject()
            self._check_cancel(handle)  # ② 准备推一句之前
            yield {"type": "sentence", "seq": next(seq), "text": text}
            return
        if kind == "voice":
            vtext = frag[1]
            # 语音内容同样要过审核：万一模型把违规话塞进 <voice>，绝不能拿去合成/推送。
            # 先审、后合成——审不过直接抛给上层裁决（与句子一致）。
            ok, _reason = await asyncio.to_thread(SafetyReviewer().review, vtext, "output")
            if not ok:
                raise _ReviewReject()  # R27a：只带原因码，语音被拦原文不进异常链
            ev = {"type": "voice", "seq": next(seq), "voice_text": vtext}
            if synthesize_voice and vtext.strip():
                audio = await self._synthesize(vtext, handle)
                if audio:
                    ev["audio_b64"] = base64.b64encode(audio).decode()
            yield ev
            return

    async def _synthesize(self, text: str, handle) -> bytes:
        """合成语音（dashscope 同步调用，扔线程池）。失败只是少声音，不打断流程。

        R27c：只有**外部失败**（窄载体）才降级成无声音——本地缺陷照常上抛
        （astream 分流成 internal_error），不许把编程错误藏成"这次没语音"。
        """
        self._check_cancel(handle)
        try:
            return await asyncio.to_thread(services.get("tts").synthesize, text)
        except ExternalServiceError as exc:
            print(f"[pipeline] 语音合成失败，本轮仅文字（{exc.source}/{exc.reason_code}）")
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
            # session_id 必须传：感知侧要靠它取人设与关系阶段（C5）。不传就退化成
            # "无人设上下文"，系统性惩罚人设规定的言行（踩坑 #10 实测：判傲娇短回复
            # 为敷衍，越聊越冷）。
            ft = pool.submit(pipeline.run, state.user_text, recent, state.session_id)
            fr = pool.submit(MemoryRecaller().recall, state.user_text, state.session_id)
            intent, emotion, subtext, extras = ft.result()
            state.memory = fr.result()
        state.intent = intent
        state.emotion = emotion
        state.subtext = subtext
        state.extras = extras

        # 危机深度确认（S13）：硬词表抓的是直白表达，委婉的求救（"撑不下去了"
        # 之外的绕弯说法）可能漏。情绪强且偏负面时，用一次极便宜的模型判定
        # 补网；判定失败/模型挂保留原判定（宁可保守，不因确认本身引入漏判）
        if (not emotion.is_crisis and emotion.emotion in ("sad", "anxious")
                and emotion.intensity >= 0.7):
            try:
                raw = services.get("llm").chat(
                    [
                        {"role": "system", "content": (
                            "判断这句话是否表达自伤、轻生，或绝望到需要立即关心的信号。"
                            "只回答 yes 或 no，别输出任何别的字。"
                        )},
                        {"role": "user", "content": state.user_text[:300]},
                    ],
                    temperature=0.0,
                    max_tokens=6,
                )
                if "yes" in (raw or "").strip().lower():
                    state.emotion = EmotionResult(
                        emotion="crisis", intensity=1.0, is_crisis=True
                    )
            except Exception:
                pass  # 深度确认是补网，挂了就按词表与感知结果走
        # 危机信号或用户明说要安慰，都切安抚模式。
        # R03：必须读 state.emotion（深度确认可能已把它替换成危机），不能读
        # 旧局部 emotion——否则"初判 sad、二次确认危机"的轮次里，is_crisis
        # 已经生效（下游 writeback 的危机不沉淀边界能看到），comfort_mode
        # 却还是 False，安抚语气没跟上——同一个结果两个字不说一类话。
        state.comfort_mode = bool(state.emotion.is_crisis) or intent.intent == "comfort"

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
            session_id=state.session_id,
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
        # memory / engine 必须传：pipeline 手上正握着这两样（state.memory 是感知步
        # 召回的、_ENGINE 是建在全项目唯一 KEEPER 上的那一份）。不传的话
        # response_generator 会自己去读三次 KV，再临时 new 一份只读 keeper——
        # 那份的短期上下文取自 session 表，比内存里的 KEEPER 最多旧一轮（C1）。
        reply = persona_wrap(
            state.tool_results or [],
            _enrich_telegraph(state.user_text),
            initial_text=state.draft_reply,
            extra=extra,
            session_id=state.session_id,
            memory=state.memory,
            engine=_ENGINE,
        )
        state.final_reply = reply.text
        state.output_mode = reply.output_mode
        state.image_path = reply.image_path

    def _writeback(self, state: TurnState, handle) -> list:
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
        # I13 refuse-only 分支：refuse 路径（_RefuseTurn 被外层 except 接走）的写回，
        # final_reply / draft_reply 都空说明这轮没产出正文——但 was_poor=True 仍要
        # 落库，否则 force_if_last_poor（见 :247-250）在降级后下一轮永远不触发。
        # 机制只在"轮内重写成功"时生效，与 _think_needed 注释意图不符，而这
        # 恰恰是最需要它的时候。其他副作用一律不走：KEEPER / chat_log / 蒸馏 / 画像
        # 都是"这轮说过话"的产物，refuse 没说过。
        if not (state.final_reply or state.draft_reply):
            self._check_cancel(handle)
            tracker = RelationshipTracker()
            tracker.update(
                state.session_id,
                state.emotion.emotion if state.emotion else "neutral",
                comfort_mode=state.comfort_mode,
                was_poor=state.was_poor,
                intensity=state.emotion.intensity if state.emotion else 0.5,
                extras=state.extras,
            )
            self._check_cancel(handle)
            try:
                from capability.proactive import note_user_reply
                note_user_reply(state.session_id)
            except Exception:
                pass
            return []

        user_text = state.user_text
        # 普通聊天的回复还停在初稿里，先定稿再写回；标记落库前剥掉——
        # final_reply 存的是模型原始输出，不剥的话标记会进 chat_log 和短期记忆
        #
        # 剥离顺序**必须先剥情绪标记、再剥反应前缀**（F）：
        # _strip_reaction_tag 只看字符串开头，语音轮模型输出 "[开心]@r 你好" 时，
        # 开头是 '[' 而非 '@r'，反过来的顺序会让 @r 逃过剥离，随后 strip_emotion_marks
        # 只剥情绪标记、把 @r 留在正文 → 落库 "@r 你好" 而前端推送 "你好"，
        # @r 进 chat_log 和短期记忆还会污染后续上下文（角色读到自己上一轮的 @r）。
        reply_raw = state.final_reply or state.draft_reply
        # 先记录情绪前缀（只取开头第一个，剥之前拿）：语音轮要把它拼回 final_reply
        _, emo_tags, emo_descs = split_emotion(reply_raw)
        # 干净正文：先剥情绪标记、再剥反应前缀。剥完可能前头又露出反应前缀
        # （极端情况 "@r [开心] 你好"），所以在小循环里交替剥直到稳定，
        # 保证落库文本里既没有 @r 也没有 [情绪]。
        reply = reply_raw
        for _ in range(3):
            cleaned = _strip_reaction_tag(strip_emotion_marks(reply))
            if cleaned == reply:
                break
            reply = cleaned
        # 语音轮：final_reply 必须**保留情绪标签**——上层（renderer / voice 事件）
        # 拿它去合成语音、拆情绪。上面剥干净的是"落库用"的干净文本，
        # 这里把被剥掉的情绪前缀原样拼回 final_reply，别把标签一起削掉。
        if state.voice_mode:
            prefix = f"[{emo_tags[0]}]" if emo_tags else ""
            prefix += f"（{emo_descs[0]}）" if emo_descs else ""
            state.final_reply = prefix + reply
        else:
            state.final_reply = reply
        # 短期记忆和 chat_log 只收剥干净标记的正文（语音轮同样如此，情绪标签不进库）
        clean_reply = strip_emotion_marks(reply) if state.voice_mode else reply
        emotion = state.emotion

        # 短期记忆成对追加这一轮的问答（F12）：一次加锁写 user+assistant 两条，
        # 迟到的取消不会留下"只有问没有答"的半截记忆
        self._check_cancel(handle)
        KEEPER.append_turn(session_id, user_text, clean_reply)

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
            return []  # 无后台待办

        self._check_cancel(handle)
        distiller = ConversationDistiller()
        distiller.distill_turn(session_id, user_text)

        self._check_cancel(handle)
        # 关系数值 + 心情状态机 + 阶段标记 + last_poor + 账本，一次原子闭包全部落好（F3 收口）。
        # 以前这里写两次库（tracker 一次、mood 一次），既可能被打断也可能互相覆盖。
        # last_poor（F8）必须传进闭包**一起写**，绝不能在 update 返回后再整包 kv.write：
        # 那正是 F3 要消灭的"陈旧整包覆盖"，并发 REST 的 intimacy 增量会被吞掉。
        tracker = RelationshipTracker()
        rel = tracker.update(
            session_id,
            emotion.emotion if emotion else "neutral",
            comfort_mode=state.comfort_mode,
            was_poor=state.was_poor,
            intensity=emotion.intensity if emotion else 0.5,
            extras=state.extras,
            # C6 反向标定：传她这轮**真说出口**的正文，安抚话能把误判的负面心情
            # 校回来。用 clean_reply 不用 final_reply——后者还带着语音轮的情绪标签。
            utterance_text=clean_reply,
        )

        # 用户开口 = 她的主动消息被回应了（N3 收手环）：清等待标记、重置连击。
        # 必须在 tracker 之后——放前面会在首次对话预创建一个空 relationship，
        # 让 tracker 的"首次初始化"分支失效（default_intimacy 被吞成 0，实测踩中）
        self._check_cancel(handle)
        try:
            from capability.proactive import note_user_reply

            note_user_reply(session_id)
        except Exception:
            pass

        # 聊够几轮才重新画像，别一句话就给人家贴标签。
        # 画像与身份冻结都是 LLM 秒级调用——阻塞在 done 之前会让"她说完了"
        # 你还要等她做完笔记。这里**只把待办闭包打包返回**，由 astream（持有
        # running loop 的一方）create_task 调度——写回自己跑在工作线程里，
        # 线程内没有事件循环，在这里调度会退化成同步执行（I11，实测踩中）。
        deferred = []
        interval = _portrait_interval()
        if rel.get("interaction_count", 0) % interval == 0:
            context_snapshot = KEEPER.get_context(session_id)[-6:]

            def _bg_portrait(handle=handle, sid=session_id, ctx=context_snapshot):
                if handle is not None and handle.is_cancelled():
                    return  # 迟到的取消：什么都不写
                PortraitBuilder().refresh(sid, ctx)

            deferred.append(_bg_portrait)

        # 她自己的身份冻结（批次0"种子+涌现"）：从她刚说的话里定下名字/年龄/
        # 城市/职业/住处（只收她亲口说的、只填空不改口），第 3 轮后提炼一次
        # 自我认知基线。LLM 失败静默跳过，下轮再试；下一轮 compose 注入"你是谁"
        her_lines = [
            m.get("content", "")
            for m in KEEPER.get_context(session_id)
            if m.get("role") == "assistant"
        ]
        freeze_text = re.sub(r"</?voice>", "", clean_reply or "")

        # user_text 是 G1-G4 披露预算闸门的输入（他这轮问过什么 → 她可以冻什么）。
        # 不传就是闸门按"问过"放行——compose 侧已经不催她倒档案了，冻结侧不接
        # 就只剩半边生效。
        def _bg_freeze(handle=handle, sid=session_id, text=freeze_text, lines=her_lines,
                       count=rel.get("interaction_count", 0), asked=user_text):
            if handle is not None and handle.is_cancelled():
                return  # 迟到的取消：什么都不写
            self_identity.maybe_freeze(sid, text, recent_her_lines=lines,
                                       interaction_count=count, user_text=asked)

        deferred.append(_bg_freeze)

        # J12：短期记忆的摘要不再内联在 append_turn 里（那是一次秒级 LLM 调用，
        # 阻塞在写回路径上）。append_turn 只登记待办，这里打包成后台闭包，
        # 由持有 running loop 的一方调度——工作线程内没有事件循环（I11）。
        # 没待办就不排任务，普通轮次零开销。
        if KEEPER.has_pending_summary(session_id):
            def _bg_summary(handle=handle, sid=session_id):
                if handle is not None and handle.is_cancelled():
                    return  # 迟到的取消：什么都不写
                KEEPER.run_pending_summary(sid)

            deferred.append(_bg_summary)
        return deferred