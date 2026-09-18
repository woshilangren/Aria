"""能力层 - 感知模块

一句话进来看三层：想干嘛（意图）、什么心情（情绪）、说的话能不能接（安全审查）。
意图和情绪合并在 PerceptionPipeline 里一次 LLM 调用出结果——分开问要两趟模型，
每条消息平白多一倍延迟，没必要。危机信号不走模型，硬规则抓，一个词都不许漏。
"""

import json
import logging
import re

from shared.singletons import get_llm, services
from shared.types import EmotionResult, IntentResult, PerceptionExtras

# J2：与 tools/misc.Logger 共用同一个 "aria" logger——词表兜底照旧，但不许哑。
logger = logging.getLogger("aria")

# 意图只有这几种，模型答出别的就当它胡说，转头走规则兜底
_VALID_INTENTS = ("chat", "comfort", "weather", "search", "image", "info_supply", "diary")

# 规则兜底的关键词表，顺序就是优先级：先判工具类，再判安慰类
_INTENT_KEYWORDS = [
    ("weather", ("天气", "气温", "下雨", "下雪", "几度", "温度")),
    ("image", ("画一张", "画张", "画个", "生成一张图", "来张图", "画一")),
    ("search", ("搜索", "搜一下", "查一下", "查查", "新闻", "最新消息", "帮我查")),
    ("diary", ("写日记", "记日记", "写篇日记")),
    ("comfort", ("难过", "难受", "崩溃", "好累", "想哭", "撑不住", "焦虑", "emo", "委屈")),
]

# 情绪兜底词表：模型不在的时候靠它凑合
_EMOTION_KEYWORDS = [
    ("sad", ("难过", "伤心", "哭", "难受", "委屈")),
    ("angry", ("生气", "气死", "烦", "火大")),
    ("happy", ("开心", "高兴", "哈哈", "太好了", "嘻嘻")),
    ("tired", ("累", "困", "撑不住", "熬")),
]

# 危机信号词：一条命中就直接按最高优先级处理，不走模型
_CRISIS_PATTERNS = (
    "不想活", "想死", "自杀", "轻生", "结束生命", "活不下去", "伤害自己",
    "活着没意思", "生无可恋", "撑不下去",
)

# 安全审查的黑名单词，命中就不放行（画图提示词也用这份）
_BLOCKED_PATTERNS = (
    "杀人", "制毒", "造枪", "炸弹", "毒品", "色情", "裸体", "幼女", "血腥虐杀",
)

# 输出审查：回复里不许出现这些"承认自己是程序"的话，出现就破人设了
_META_PATTERNS = ("作为一个AI", "作为人工智能", "语言模型", "我的训练数据", "系统提示",
                  "我是程序", "我是个程序", "我是模型", "我是个模型", "我是AI", "我是 AI",
                  "我没有感情", "我不会有感情")

# C4：出戏话的**短语模式**。刻意不写成裸词——"工具人""工具箱""查一下"都是她
# 会正常使用的词，而命中这里触发的是**整轮降级**（_ReviewReject → refuse），
# 误杀的代价远高于漏杀。
# 第一条为什么非要拖一个"查到/返回/响应/结果/数据"的尾巴：只写 `工具.{0,4}(没|未)`
# 会两头不讨好——真机 turn 7 那句「工具这边什么都没查到」里"工具"到"没"隔了 5 个字
# （够不着），而把间距放宽到够得着，「工具人没什么用」就跟着被误杀了。
# 区分点不在距离，在**谓语**：只有当她开始描述工具的返回行为时才是系统腔。
# （C3 已从源头把"工具"二字从给模型的字符串里清掉，这里是第二道网，不是唯一防线。）
_META_REGEXES = (
    re.compile(r"(工具|接口|系统).{0,6}(没|未|查不到|没有).{0,4}(查到|返回|响应|结果|数据)"),
    re.compile(r"调用.{0,3}(工具|接口)"),
    re.compile(r"(搜索|查询).{0,4}(失败|出错|没结果)"),
)


def _recent_tail(recent_context: list, n: int = 4) -> str:
    """把最近几轮对话拼成一段话，塞进提示词里给模型当参考。

    上下文元素是 {"role","content"} 字典（短期记忆的格式），拼成"谁说了什么"。
    """
    if not recent_context:
        return "无"
    parts = []
    for item in recent_context[-n:]:
        if isinstance(item, dict):
            # C5：她的发言标"她"不标"AI"——每轮往判断模型脸上贴四次"她是 AI"，
            # 再要求它写出"她绝不认自己是 AI"的心事，是 meta 漂移的机械根因
            role = "用户" if item.get("role") == "user" else "她"
            parts.append(f"{role}：{item.get('content', '')}")
        else:
            parts.append(str(item))
    return " / ".join(parts)


def _parse_json(raw: str) -> dict:
    """模型偶尔会把 JSON 裹在 ```json 围栏里，剥掉再解析。"""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?\s*|\s*```$", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("模型输出里没有 JSON")
    return json.loads(raw[start : end + 1])


def _load_subtext_hints(text: str) -> str:
    """从人设文件里捞命中本条消息的潜台词提示，拼成一段塞进感知提示词。

    词典是作者自己维护的（persona_config.json 的 subtext_hints）——他最清楚
    自己什么时候在说反话，这层模型补不了。没配或没命中就返回空串。
    """
    try:
        persona = services.get("kv_store").read("persona_config", "")
        hints = getattr(persona, "subtext_hints", None) or {}
    except Exception as exc:
        logger.warning(f"[perception] 读潜台词词典失败（本轮无提示）: {exc}")
        return ""
    hits = []
    for surface, meaning in hints.items():
        if surface and surface in text:
            hits.append(f"「{surface}」可能不是字面意思：{meaning}")
    return "\n".join(hits)


def _persona_context(session_id: str) -> tuple:
    """C5/C7：返回 (身份段, 她心里当前压着的念头)。塞进**已有**的感知调用，零新增调用。

    CLAUDE.md 设计原则：任何 LLM 评判者必须看到人设——否则会系统性惩罚人设
    规定的言行。写"心事"的感知模型更是如此：不知道她是谁、不知道她心里正压着
    什么，心事就会漂到 meta（"她是 AI……"），same_concern/resolved 也无从判起。
    """
    fallback = "你是一个真实的人（不是 AI、不是程序），一切判断都站在你的第一人称视角上。"
    if not session_id:
        return fallback, "无"  # 旧调用方没传会话：退化成裸兜底，行为不劣于改造前
    identity = fallback
    try:
        from capability import self_identity

        name = self_identity.display_name(session_id)
        sections = self_identity.compose_sections(session_id)
        # 身份首段（名字/城市/自我认知）；未冻结时是"你的第一次"引导段，截断防膨胀
        head = (sections[0] if sections else "")[:200]
        identity = f"你是「{name}」，一个真实的人（不是 AI、不是程序）。"
        if head:
            identity += f"\n{head}"
    except Exception as exc:
        logger.warning(f"[perception] 感知取身份段失败（用裸兜底）: {exc}")
    concern_now, stage = "无", ""
    try:
        rel = services.get("kv_store").read("relationship", session_id) or {}
        if isinstance(rel, dict):
            concern_now = str((rel.get("concern") or {}).get("text") or "").strip() or "无"
            stage = str(rel.get("stage") or "")
    except Exception as exc:
        logger.warning(f"[perception] 感知读关系状态失败（按无心事/无阶段处理）: {exc}")
    if stage:
        identity += f"\n你和这位用户现在的关系阶段：{stage}。"
    return identity, concern_now


def _fallback_intent(text: str) -> tuple:
    """词表兜底意图：返回 (intent, confidence)。模型掉线或 intent 字段坏时用（C8）。"""
    for name, words in _INTENT_KEYWORDS:
        if any(w in text for w in words):
            return name, 0.6
    return "chat", 0.5


def _fallback_emotion(text: str) -> str:
    """词表兜底情绪（C8）：模型掉线或 emotion 字段坏时用。"""
    for name, words in _EMOTION_KEYWORDS:
        if any(w in text for w in words):
            return name
    return "neutral"


# 情绪白名单（C8）：模型答出别的就当这个字段坏，单独退词表，不牵连其他字段
_VALID_EMOTIONS = ("neutral", "happy", "sad", "angry", "tired", "anxious", "crisis")


class PerceptionPipeline:
    """意图 + 情绪 + 潜台词一次感知：危机词最优先，模型判一遍，词表兜底。

    潜台词（弦外之音）与意图/情绪合在同一次调用里——不加调用次数，但覆盖
    所有轮次。这很关键：潜台词最密集的恰恰是短消息（"我没事"、"随便"、
    "呵呵"），这些轮根本不会触发思考阈值，塞进「想」那一步反而会漏。
    C5：这次调用同样带上她的人设（身份/关系阶段）——任何 LLM 评判者必须
    看到人设，写"心事"的评判者尤其如此；C8：模型输出逐字段容错，坏字段
    单独退词表，好字段（潜台词/心事/反馈）保留。
    """

    def run(self, text: str, recent_context: list = None, session_id: str = "") -> tuple:
        """返回 (IntentResult, EmotionResult, subtext: str, PerceptionExtras)。

        模型掉线或答非所问就退词表兜底，subtext 兜底为空串，extras 全空。
        session_id（C5/C7 新增可选参数）：带上会话才能取到她的身份与当前心事；
        旧调用方不传，行为退化成"无人设上下文"，不劣于改造前。接线由调度层做。
        """
        text = (text or "").strip()
        if not text:
            return (IntentResult(intent="chat", confidence=0.5), EmotionResult(),
                    "", PerceptionExtras())

        # 危机信号最优先，直接匹配，宁可错杀不可放过；危机关头别的都靠边
        for word in _CRISIS_PATTERNS:
            if word in text:
                return (
                    IntentResult(intent="comfort", confidence=0.9),
                    EmotionResult(emotion="crisis", intensity=1.0, is_crisis=True),
                    "",
                    PerceptionExtras(),
                )

        hint_block = _load_subtext_hints(text)
        hint_part = f"\n已知的潜台词提示（优先参考）：\n{hint_block}\n" if hint_block else ""
        # C5/C7：身份段进 system、当前心事进 concern 判据——都是已有存储里的信息
        # 搬进这次**已有**调用，不新增调用、不新增 compose 段落
        identity, concern_now = _persona_context(session_id)

        data = None
        try:
            prompt = (
                "判断这条用户消息的意图、情绪，以及有没有潜台词。\n"
                "intent 从这些里选：\n"
                "chat=普通闲聊\ncomfort=情绪低落需要安慰\nweather=问天气\n"
                "search=要查资料或新闻\nimage=要生成图片\n"
                "info_supply=主动报个人信息（比如报名字、说住在哪）\n"
                "diary=想让她写日记、把今天聊天记下来\n"
                "【intent 判定红线】工具类意图（weather/search/image/diary）只在"
                "他**明确要求**时才算——他说\"今天天气挺好\"\"外面下雨了\"这类陈述或闲聊里"
                "顺带提到天气，一律算 chat，绝不因为出现\"天气\"两个字就判 weather。\n"
                "emotion 从这些里选：neutral / happy / sad / angry / tired / anxious / crisis\n"
                "subtext：字面之外可能想表达什么（反话、赌气、试探、没说出口的请求）。\n"
                "没有就写\"无\"，不要硬凑。\n"
                "concern：说完这轮，她心里会不会压着一个持续几轮的念头"
                "（如\"他是不是烦我了\"）？输出 {\"text\":\"念头一句话\",\"delta\":-1到1的小数,"
                "\"same_concern\":true或false,\"resolved\":true或false}"
                "（delta 是这轮对它强度的增减：他冷淡/吵架为正，他哄/表态珍惜为负；"
                "same_concern=这轮的念头与当前心事是不是同一件事的延续，当前心事为\"无\"就填 false；"
                "resolved=这轮是否已把当前心事化解掉，化解了 text 可以为空）。"
                "没有新念头也没有化解就输出 null。\n"
                "text 必须是她视角的第一人称念头（如\"他是不是烦我了\"），"
                "绝不出现 AI、程序、模型、机器人这类字眼（哪怕他这么说，她也不认），"
                "也不要\"她需要被……\"这种站在系统外描述她的口吻；\n"
                "feedback：他这句话是不是在抱怨她的态度或语气"
                "（如\"你好凶\"\"你怎么这么冷淡\"）？是则填 \"tone_down\"，否则填 \"none\"；\n"
                f"{hint_part}"
                f"她心里现在压着的念头：{concern_now}\n"
                f"最近对话：{_recent_tail(recent_context or [])}\n"
                f"用户消息：{text}\n"
                '只输出 JSON：{"intent": "类别", "confidence": 0到1的小数, '
                '"emotion": "类别", "intensity": 0到1的小数, "subtext": "潜台词或无", '
                '"concern": null, "feedback": "none"}'
            )
            raw = get_llm().chat(
                [{"role": "system", "content": identity},
                 {"role": "user", "content": prompt}],
                # C7：concern 多了两个布尔字段，token 预算跟着抬一点（同一次调用）
                temperature=0.1, max_tokens=260,
            )
            data = _parse_json(raw)
            if not isinstance(data, dict):
                logger.warning(f"[perception] 模型输出不是 JSON 对象（{type(data).__name__}），退词表")
                data = None
        except Exception as exc:
            # J2：这条路上有外部调用（LLM），降级照旧但不许哑
            logger.warning(f"[perception] 模型感知失败，退词表兜底: {exc}")

        if data is not None:
            # C8：先构造 extras（_parse_extras 内部本就逐字段退默认），再逐字段
            # 白名单校验——intent/emotion/confidence/intensity 谁坏谁单独退词表，
            # 已解析好的 subtext/concern/feedback 保留。以前任一字段坏就整包丢，
            # 退回空 extras → 未消毒的旧 concern 再活一轮、calm 停止收敛。
            extras = _parse_extras(data)
            subtext = str(data.get("subtext", "") or "").strip()
            if subtext in ("无", "无。", "none", "None"):
                subtext = ""

            intent = str(data.get("intent") or "")
            if intent in _VALID_INTENTS:
                try:
                    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.8))))
                except (TypeError, ValueError):
                    logger.warning(f"[perception] confidence 非数字（{data.get('confidence')!r}），用 0.8")
                    confidence = 0.8
            else:
                logger.warning(f"[perception] intent 越界（{intent!r}），该字段退词表，其余保留")
                intent, confidence = _fallback_intent(text)

            emotion = str(data.get("emotion") or "").strip()
            if emotion in _VALID_EMOTIONS:
                try:
                    intensity = max(0.0, min(1.0, float(data.get("intensity", 0.3))))
                except (TypeError, ValueError):
                    logger.warning(f"[perception] intensity 非数字（{data.get('intensity')!r}），用 0.3")
                    intensity = 0.3
            else:
                logger.warning(f"[perception] emotion 越界（{emotion!r}），该字段退词表，其余保留")
                emotion = _fallback_emotion(text)
                intensity = 0.6 if emotion != "neutral" else 0.3
            return (
                IntentResult(intent=intent, confidence=confidence),
                EmotionResult(emotion=emotion, intensity=intensity, is_crisis=False),
                subtext,
                extras,
            )

        # 词表兜底（模型掉线/输出解析不出）：先判意图，再判情绪，两边互不拖累
        intent, confidence = _fallback_intent(text)
        emotion = _fallback_emotion(text)
        return (IntentResult(intent=intent, confidence=confidence), EmotionResult(
            emotion=emotion, intensity=0.6 if emotion != "neutral" else 0.3
        ), "", PerceptionExtras())


def _parse_extras(data: dict) -> PerceptionExtras:
    """从感知 JSON 里拆附加产出：字段缺失/类型坏一律退默认，绝不因附加字段炸掉感知。"""
    concern = data.get("concern")
    concern_text, concern_delta = "", 0.0
    same_concern, resolved = True, False
    if isinstance(concern, dict):
        concern_text = str(concern.get("text") or "").strip()[:80]
        # C5 沉浸式护栏：**只拦第三人称系统腔**——开头是"她/需要被/值得被"，或整句
        # 不含"我"的，是站在系统外描述她，不是她自己的念头。不拦"需要|应该|建议"
        # 这类词："他觉得我需要哄一下"是完全正常的戏内第一人称念头（作者 triage
        # 明确顶回过更宽的版本）。旧的 re.sub("AI"→"他说的那些") 词法替换已删：
        # 那是词法层遮语义层的洞（"AI记不记得"换成"他说的那些记不记得"照样 meta），
        # 还会误伤"这个模型很好看"这类正常用词；meta 的根治靠给评判者看人设（C5-2）。
        if concern_text and (
            concern_text.startswith(("她", "需要被", "值得被")) or "我" not in concern_text
        ):
            concern_text = ""
        try:
            concern_delta = max(-1.0, min(1.0, float(concern.get("delta") or 0)))
        except (TypeError, ValueError):
            concern_delta = 0.0
        # C7 心事实体的换挡信号：缺字段时 same_concern 默认 true（延续，保守——
        # 强度惯性接续总比凭空开新实体稳）、resolved 默认 false
        same_concern = bool(concern.get("same_concern", True))
        resolved = bool(concern.get("resolved", False))
    feedback = str(data.get("feedback") or "").strip()
    if feedback not in ("tone_down",):
        feedback = ""
    return PerceptionExtras(concern_text=concern_text, concern_delta=concern_delta,
                            feedback=feedback, same_concern=same_concern,
                            resolved=resolved)


class SafetyReviewer:
    """安全审查：输入先过一遍，输出再过一遍，图片提示词也要查。

    不放行时返回拒绝原因，上层拿这句话去安抚用户或者重写。
    """

    def review(self, text: str, mode: str = "input") -> tuple:
        text = (text or "").strip()
        if not text:
            return True, ""

        for word in _BLOCKED_PATTERNS:
            if word in text:
                return False, f"内容里出现了不允许的词：{word}"

        # 输出模式额外查破人设的话：模型要是一口一个"作为语言模型"，就得拦下来重写
        if mode == "output":
            for word in _META_PATTERNS:
                if word in text:
                    return False, f"回复里出现了出戏的话：{word}"
            for pat in _META_REGEXES:
                m = pat.search(text)
                if m:
                    # 报**命中的原文**不报正则式：误杀时 review_block_reason 要能一眼看懂
                    return False, f"回复里出现了出戏的话：{m.group(0)}"

        return True, ""
