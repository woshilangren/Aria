"""能力层 - 感知模块

一句话进来看三层：想干嘛（意图）、什么心情（情绪）、说的话能不能接（安全审查）。
意图和情绪合并在 PerceptionPipeline 里一次 LLM 调用出结果——分开问要两趟模型，
每条消息平白多一倍延迟，没必要。危机信号不走模型，硬规则抓，一个词都不许漏。
"""

import json
import re

from shared.singletons import get_llm
from shared.types import EmotionResult, IntentResult

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
_META_PATTERNS = ("作为一个AI", "作为人工智能", "语言模型", "我的训练数据", "系统提示")


def _recent_tail(recent_context: list, n: int = 4) -> str:
    """把最近几轮对话拼成一段话，塞进提示词里给模型当参考。

    上下文元素是 {"role","content"} 字典（短期记忆的格式），拼成"谁说了什么"。
    """
    if not recent_context:
        return "无"
    parts = []
    for item in recent_context[-n:]:
        if isinstance(item, dict):
            role = "用户" if item.get("role") == "user" else "AI"
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


class PerceptionPipeline:
    """意图 + 情绪一次感知：危机词最优先，模型判一遍，词表兜底。"""

    def run(self, text: str, recent_context: list = None) -> tuple:
        """返回 (IntentResult, EmotionResult)。模型掉线或答非所问就退词表兜底。"""
        text = (text or "").strip()
        if not text:
            return IntentResult(intent="chat", confidence=0.5), EmotionResult()

        # 危机信号最优先，直接匹配，宁可错杀不可放过；危机关头别的都靠边
        for word in _CRISIS_PATTERNS:
            if word in text:
                return (
                    IntentResult(intent="comfort", confidence=0.9),
                    EmotionResult(emotion="crisis", intensity=1.0, is_crisis=True),
                )

        try:
            prompt = (
                "判断这条用户消息的意图和情绪。\n"
                "intent 从这些里选：\n"
                "chat=普通闲聊\ncomfort=情绪低落需要安慰\nweather=问天气\n"
                "search=要查资料或新闻\nimage=要生成图片\n"
                "info_supply=主动报个人信息（比如报名字、说住在哪）\n"
                "diary=想让AI写日记、把今天聊天记下来\n"
                "emotion 从这些里选：neutral / happy / sad / angry / tired / anxious / crisis\n"
                f"最近对话：{_recent_tail(recent_context or [])}\n"
                f"用户消息：{text}\n"
                '只输出 JSON：{"intent": "类别", "confidence": 0到1的小数, '
                '"emotion": "类别", "intensity": 0到1的小数}'
            )
            raw = get_llm().chat(
                [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=96
            )
            data = _parse_json(raw)
            intent = str(data.get("intent", ""))
            if intent not in _VALID_INTENTS:
                raise ValueError(f"意图不在白名单里：{intent}")
            intensity = float(data.get("intensity", 0.3))
            return (
                IntentResult(intent=intent, confidence=float(data.get("confidence", 0.8))),
                EmotionResult(
                    emotion=str(data.get("emotion", "neutral")),
                    intensity=max(0.0, min(1.0, intensity)),
                    is_crisis=False,
                ),
            )
        except Exception:
            pass  # 模型掉线或答非所问都别慌，下面有词表兜底

        # 词表兜底：先判意图，再判情绪，两边互不拖累
        intent, confidence = "chat", 0.5
        for name, words in _INTENT_KEYWORDS:
            if any(w in text for w in words):
                intent, confidence = name, 0.6
                break
        emotion = "neutral"
        for name, words in _EMOTION_KEYWORDS:
            if any(w in text for w in words):
                emotion = name
                break
        return IntentResult(intent=intent, confidence=confidence), EmotionResult(
            emotion=emotion, intensity=0.6 if emotion != "neutral" else 0.3
        )


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

        return True, ""
