"""调度层 - LangGraph 状态与节点

DialogueState 是一轮对话在节点之间流转的状态袋子，
每个节点只干一件事：调能力层一个模块，把结果写回状态，不带业务逻辑。
图怎么连（条件边）见 dialogue_orchestrator.py 的 build_graph。
"""

from typing import Optional, TypedDict

from shared.types import EmotionResult, IntentResult, MemoryBundle, PromptPackage

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
from capability.response_generator import generate, persona_wrap
from capability.toolcall import ToolCallOrchestrator

from orchestration.managers import FallbackController, InfoGapCoordinator
from shared.singletons import services
from tools.speech import strip_emotion_marks

# 短期记忆的管家全项目就这一份：compose 要拿上下文、writeback 要写记录，
# 两处用的必须是同一个对象，不然短期记忆各记各的就对不上了
KEEPER = SessionMemoryKeeper()

# 人设引擎也共用一份，绑定上面那个管家
_ENGINE = PersonaEngine(KEEPER)

# 后置审核最多重写几次，超了就给兜底话，别死循环
_MAX_REWRITE = 2

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


class DialogueState(TypedDict, total=False):
    """单轮对话的流转状态（每个节点读取并补充字段）。"""

    # 输入
    user_text: str                        # 用户消息文本
    session_id: str                       # 会话 ID
    voice_mode: bool                      # 是否语音轮（语音轮回复要带情绪标注）
    user_image: str = ""                  # 用户随消息上传的图片地址（多模态模型看）

    # 各节点产出
    safety_passed: bool                   # 安全预检是否通过
    intent: Optional[IntentResult]        # 意图识别结果
    emotion: Optional[EmotionResult]      # 情绪感知结果
    comfort_mode: bool                    # 是否进入安抚模式
    memory: Optional[MemoryBundle]        # 记忆召回包
    prompt: Optional[PromptPackage]       # 人设提示词包
    draft_reply: str                      # 回复初稿（可能是模型直答，也可能是工具初答）
    tool_results: list                    # 工具管线跑出来的结果列表
    rewrite_count: int                    # 后置审核后已重写的次数
    review_passed: bool                   # 后置审核是否通过
    final_reply: str                      # 最终回复
    output_mode: str                      # text / image
    image_path: str                       # 回复带图时的图片路径
    error: str                            # 异常信息（触发兜底时填写）


def safety_precheck_node(state: DialogueState) -> DialogueState:
    """节点：安全预检。用户的话先过一遍黑名单。"""
    reviewer = SafetyReviewer()
    ok, _reason = reviewer.review(state["user_text"], mode="input")
    state["safety_passed"] = ok
    return state


def perceive_node(state: DialogueState) -> DialogueState:
    """节点：感知 + 记忆召回并行。

    意图/情绪（一次 LLM 调用）和记忆召回（两次向量检索）互不依赖，
    串行等于把两条网络延迟加起来，回复慢一大截——扔线程池里同时跑，
    总耗时取两者较慢的那个。代价是工具类意图也会多付两次向量检索，
    但工具轮本来就要跑好几秒外部接口，这点开销不心疼。
    """
    from concurrent.futures import ThreadPoolExecutor

    pipeline = PerceptionPipeline()
    # 最近几轮对话给模型当参考，"还要"这种话没上下文根本判不准
    recent = KEEPER.get_context(state["session_id"])[-4:]
    with ThreadPoolExecutor(max_workers=2) as pool:
        ft = pool.submit(pipeline.run, state["user_text"], recent)
        fr = pool.submit(MemoryRecaller().recall, state["user_text"], state["session_id"])
        intent, emotion = ft.result()
        state["memory"] = fr.result()
    state["intent"] = intent
    state["emotion"] = emotion
    # 危机信号或用户明说要安慰，都切安抚模式
    state["comfort_mode"] = bool(emotion.is_crisis) or intent.intent == "comfort"
    return state


def compose_node(state: DialogueState) -> DialogueState:
    """节点：人设组装。安抚模式换 comfort 语气，正常轮先掷一次"小动作骰子"。"""
    mode = "comfort" if state.get("comfort_mode") else "chat"
    memory = state.get("memory") or MemoryBundle()
    rel = memory.relationship or {}
    quirk = QuirkDirector().roll(
        memory,
        stage=rel.get("stage", "初识"),
        mood=rel.get("mood", "平常"),
        comfort_mode=state.get("comfort_mode", False),
    )
    emotion = state.get("emotion")
    state["prompt"] = _ENGINE.compose(
        mode,
        memory,
        state["session_id"],
        emotion_label=emotion.emotion if emotion else "",
        quirk=quirk,
    )
    return state


def toolcall_node(state: DialogueState) -> DialogueState:
    """节点：工具管线。天气缺城市时先问人，别瞎查默认城市。"""
    intent = state["intent"]
    # 工具链路不走 recall 节点，档案直接从库里读，城市才兜底得上
    profile = {}
    try:
        profile = services.get("kv_store").read("profile", state["session_id"]) or {}
    except Exception:
        pass
    fallback_city = profile.get("city", "")

    # 天气意图但用户没提城市、档案里也没有：直接追问，不瞎查
    if intent.intent == "weather" and not fallback_city and not _has_city(state["user_text"]):
        state["final_reply"] = InfoGapCoordinator().ask("city", state["session_id"])
        return state

    out = ToolCallOrchestrator().run(
        state["user_text"], intent.intent, fallback_city, session_id=state["session_id"]
    )
    state["tool_results"] = out["results"]
    # 模型自己先给的答案存成初稿，respond 节点再拿去润成人设口吻
    state["draft_reply"] = out["final_text"]
    return state


# 纯 emoji / 无实质文字输入：小模型没话接，容易借着人设蹦单字。
# 这里只对"发给 LLM"的输入补一句语义，库里的原始 user 消息不受影响。
_TELEGRAPH_RE = None


def _enrich_telegraph(raw: str) -> str:
    import re
    global _TELEGRAPH_RE
    if _TELEGRAPH_RE is None:
        _TELEGRAPH_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
    t = (raw or "").strip()
    if not t:
        return "（对方安静地陪着你去）用一句自然的完整话回应他。"
    if not _TELEGRAPH_RE.search(t):
        return (
            f"{t}（对方只发来这个表情，这是一句很生动的情绪。好好接住他："
            "用一句完整的俏皮或关心的话回他，把话说完整，绝不只回单个字。）"
        )
    return t


def respond_node(state: DialogueState) -> DialogueState:
    """节点：把工具结果转述成人话。模型初答只是参考，要重说一遍。"""
    extra = _VOICE_EMOTION_RULE if state.get("voice_mode") else ""
    reply = persona_wrap(
        state.get("tool_results") or [],
        _enrich_telegraph(state["user_text"]),
        initial_text=state.get("draft_reply", ""),
        extra=extra,
    )
    state["final_reply"] = reply.text
    state["output_mode"] = reply.output_mode
    state["image_path"] = reply.image_path
    return state


def generate_node(state: DialogueState) -> DialogueState:
    """节点：生成回复初稿。人设提示词 + 记忆 + 用户的话（可带图），直接调模型。"""
    extra = _VOICE_EMOTION_RULE if state.get("voice_mode") else ""
    state["draft_reply"] = generate(
        state["prompt"],
        _enrich_telegraph(state["user_text"]),
        extra=extra,
        user_image=state.get("user_image", ""),
    )
    return state


def rewrite_node(state: DialogueState) -> DialogueState:
    """节点：重写。上一版没过审，换个说法再来，重写次数加一。

    工具链路没跑过 compose（没有 prompt 包），重写得走重新转述那条路，
    直接 generate 会因为拿不到人设提示词把整图炸掉。
    """
    if state.get("review_block_reason") == "too_short":
        fixed_input = (
            f"{state['user_text']}\n"
            "（系统提醒：你刚才只回了一个字，太敷衍了。请重新说一句完整的话，"
            "20~50 字，把此刻的情绪和想说的话好好说清楚，"
            "话都要说完整，不许用单个字符号打发人。）"
        )
    else:
        fixed_input = (
            f"{state['user_text']}\n"
            "（系统提醒：刚才的回复没通过检查，请换个干净的说法重新回答，"
            "别提任何违规内容，也别提被拦下这件事。）"
        )
    extra = _VOICE_EMOTION_RULE if state.get("voice_mode") else ""
    if state.get("prompt") is not None:
        state["draft_reply"] = generate(state["prompt"], fixed_input, extra=extra)
    else:
        # 工具链路：带修正提示重转述一遍工具结果
        extra = extra + "\n" if extra else ""
        reply = persona_wrap(
            state.get("tool_results") or [],
            fixed_input,
            initial_text=state.get("draft_reply", ""),
            extra=f"{extra}上一版回复没通过内容检查，换一种干净的说法重新转述，别提被拦下这件事。",
        )
        state["draft_reply"] = reply.text
        state["output_mode"] = reply.output_mode
        state["image_path"] = reply.image_path
    state["rewrite_count"] = state.get("rewrite_count", 0) + 1
    return state


def _is_onechar_reply(reply: str) -> bool:
    """单字崩检测：整句去掉空格标点后只剩 1 个汉字/字母/数字，判为敷衍。
    被夸/被闹时小模型爱用"哼/啧/哈/你/哦"单字打发，这里硬拦。"""
    import re
    cnt = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", reply or ""))
    return cnt <= 1


def safety_review_node(state: DialogueState) -> DialogueState:
    """节点：后置审核。回复出口前查一遍黑名单和破人设话术，外加单字崩拦截。"""
    reviewer = SafetyReviewer()
    reply = state.get("draft_reply", "")
    ok, reason = reviewer.review(reply, mode="output")
    if ok and _is_onechar_reply(reply):
        ok, reason = False, "too_short"
    state["review_passed"] = ok
    state["review_block_reason"] = reason
    return state


def refuse_node(state: DialogueState) -> DialogueState:
    """节点：婉拒。安全不过关或审核超限，直接给固定话术收场。"""
    controller = FallbackController()
    if not state.get("safety_passed", True):
        state["final_reply"] = controller.fallback_reply("blocked")
    else:
        state["final_reply"] = controller.fallback_reply("review")
    state["output_mode"] = "text"
    state["image_path"] = ""
    return state


def writeback_node(state: DialogueState) -> DialogueState:
    """节点：记忆写回。短期记忆、长期沉淀、亲密度、画像，一样别落。"""
    session_id = state["session_id"]
    user_text = state["user_text"]
    # 普通聊天的回复还停在初稿里，先定稿再写回
    reply = state.get("final_reply") or state.get("draft_reply", "")
    state["final_reply"] = reply
    # 语音轮：短期记忆和聊天记录只收剥掉情绪标记的正文；
    # final_reply 保持原样（带标签），上层拿它去合成语音才有情绪可拆
    clean_reply = strip_emotion_marks(reply) if state.get("voice_mode") else reply
    emotion = state.get("emotion")

    # 短期记忆追加这一轮的问答
    KEEPER.append(session_id, "user", user_text)
    if clean_reply:
        KEEPER.append(session_id, "assistant", clean_reply)

    # 聊天记录落库（chat_log 表）：日记生成、重启恢复都靠这份数据
    # 危机轮也要记——对话发生过就该留痕，只是不涨亲密度不沉淀记忆
    intent = state.get("intent")
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
            "mode": "voice" if state.get("voice_mode") else "text",
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
                "mode": state.get("output_mode", "text"),
            },
        )

    # 危机或婉拒的轮次不沉淀、不涨亲密度，这些轮次不算正常互动
    if emotion and emotion.is_crisis:
        return state

    distiller = ConversationDistiller()
    distiller.distill_turn(session_id, user_text)

    tracker = RelationshipTracker()
    rel = tracker.update(session_id, emotion.emotion if emotion else "neutral")

    # 心情状态机：数值之外，角色此刻的情绪也往前走一格，下轮 compose 要用
    rel["mood"] = MoodEngine().update(
        rel, emotion.emotion if emotion else "neutral", state.get("comfort_mode", False)
    )
    try:
        kv.write("relationship", session_id, rel)
    except Exception:
        pass

    # 聊够几轮才重新画像，别一句话就给人家贴标签
    interval = _portrait_interval()
    if rel.get("interaction_count", 0) % interval == 0:
        PortraitBuilder().refresh(session_id, KEEPER.get_context(session_id)[-6:])
    return state


def _has_city(text: str) -> bool:
    """粗查用户话里有没有城市字样，有"天气"前面接地名就算。"""
    import re

    return bool(re.search(r"[\u4e00-\u9fa5]{2,8}(市)?的?天气", text or ""))


def _portrait_interval() -> int:
    """几轮聊天更新一次画像，从配置里读，读不到就 5 轮。"""
    try:
        from config.settings import load_app_config

        return int(load_app_config()["memory"].get("session_update_interval", 5)) or 5
    except Exception:
        return 5
