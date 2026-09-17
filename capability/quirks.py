"""能力层 - 随机小动作（脾气、亲近、乱说话、调侃缺点）

角色不是应声虫：心情会自己漂，偶尔皮一下。这套机制给对话加"活人感"——
但每一项都是提示词层面的引导，硬性规则（安全审查、人设 taboos）照常生效，
安抚/危机轮一律不出手，正经事（工具问答）不掺和。

两件套：
- MoodEngine：心情状态机，每轮小概率漂移一次，跟着关系数值一起落库
- QuirkDirector：小动作骰子，按 quirk_rate 概率命中，命中了才给一句注入指令
"""

import random

from config.settings import load_app_config
from shared.types import MemoryBundle

from capability import char_life

# 平常时心情池：正面情绪往这漂
_POSITIVE_MOODS = ("得意", "雀跃", "来劲", "心软")
# 负面情绪往这漂
_NEGATIVE_MOODS = ("别扭", "低落", "慵懒", "心烦")

class MoodEngine:
    """心情状态机：事件驱动 + 惯性（N2），不再每轮掷骰子。

    真人的情绪有惯性：负面事件把心情打进去并停留几轮，随时间衰减回落；
    安抚能化解（减半）；正面情绪来得快去得快，保持轻随机。
    惯性状态存 relationship：mood_left（还剩几轮）、mood_from（由什么情绪引起）。
    """

    # 漂移概率：只在"平常"时用（惯性期不漂），给正面区间一点轻随机
    _DRIFT_RATE = 0.3

    # 用户情绪 → 她的心情标签（负面事件直接置入，不再抽签）
    _NEGATIVE_MAP = {
        "sad": "低落", "angry": "别扭", "anxious": "心烦", "tired": "慵懒",
    }

    def update(self, rel: dict, emotion: str = "neutral",
               comfort_mode: bool = False, intensity: float = 0.5) -> str:
        """根据上轮惯性 + 本轮情绪算新心情，返回心情标签（不落库，调用方负责写回）。

        rel 会被就地写上 mood_left / mood_from 两个惯性字段。
        """
        # 安抚/危机轮强制心软（随机性给正经事让路），并把负面惯性化解一半——
        # 哄是有效的，但一次哄不立刻翻篇
        if comfort_mode or emotion == "crisis":
            left = int(rel.get("mood_left") or 0)
            rel["mood_left"] = max(0, left // 2)
            if rel.get("mood") in _NEGATIVE_MOODS and rel["mood_left"] <= 0:
                rel["mood"] = "心软"
            return "心软"

        # 负面事件：直接置入对应心情，强度越高惯性越长（2~5 轮）
        if emotion in self._NEGATIVE_MAP:
            rel["mood"] = self._NEGATIVE_MAP[emotion]
            rel["mood_left"] = 2 + int(round(max(0.0, min(1.0, intensity)) * 3))
            rel["mood_from"] = emotion
            return rel["mood"]

        # 正面情绪：来得快去得快，轻随机给一格正面心情，不留长惯性
        if emotion == "happy" and random.random() < 0.6:
            rel["mood"] = random.choice(_POSITIVE_MOODS)
            rel["mood_left"] = 1
            rel["mood_from"] = ""
            return rel["mood"]

        # 中性轮：惯性期内原地衰减一格；惯性尽了回"平常"
        mood = rel.get("mood") or "平常"
        left = int(rel.get("mood_left") or 0)
        if left > 0:
            left -= 1
            rel["mood_left"] = left
            if left > 0:
                return mood
            rel["mood"] = "平常"
            rel["mood_from"] = ""
            return "平常"

        # 平常时保留一点轻漂移（正面区间），让语气偶尔有点小起伏
        if random.random() < self._DRIFT_RATE:
            picked = random.choice(_POSITIVE_MOODS)
            if picked != mood:
                rel["mood"] = picked
                rel["mood_left"] = 1
                return picked
        return "平常"


class QuirkDirector:
    """小动作骰子：每轮按概率掷一次，命中了挑一个小动作，输出注入指令。

    权重设计：翻缺点调侃 > 怼一句 > 岔开话题 > 提旧事 > 得意自夸 > 撒娇。
    撒娇要亲密度过门槛才上桌；素材（缺点、旧事）没有的动作自动缺席。
    """

    def roll(self, memory: MemoryBundle, stage: str = "初识",
             mood: str = "平常", comfort_mode: bool = False,
             session_id: str = "") -> str:
        """掷骰子。返回注入 system prompt 的指令文本，没命中返回空串（普通回合）。"""
        if comfort_mode:
            return ""  # 安抚轮只安慰，别阴阳怪气

        rate = float(
            load_app_config().get("personality", {}).get("quirk_rate", 0.12)
        )
        if rate <= 0 or random.random() >= rate:
            return ""

        flaws = [f for f in ((memory.portrait or {}).get("user_flaws") or []) if f]
        distilled = [
            item.get("content")
            for item in (memory.distilled or [])
            if item.get("content")
        ]
        # 生活面（N4）：她最近在琢磨的事——岔话题/翻旧账优先从这里取材，
        # 有连续的、属于她自己的素材，"昨天说的那本书"才接得上
        life_topic = char_life.consume_topic(session_id) if session_id else ""

        # (动作名, 权重)，素材齐不齐决定了哪些动作能上桌
        actions = [("snark", 3), ("off_topic", 2), ("pride", 2)]
        if flaws:
            actions.append(("tease", 4))
        if distilled or life_topic:
            actions.append(("recall", 2))
        if stage in ("亲近", "挚友"):
            actions.append(("clingy", 2))

        total = sum(w for _n, w in actions)
        pick = random.uniform(0, total)
        for name, weight in actions:
            pick -= weight
            if pick <= 0:
                return self._directive(name, flaws, distilled, mood, life_topic)
        return ""

    def _directive(self, name: str, flaws: list, distilled: list, mood: str,
                   life_topic: str = "") -> str:
        """把抽中的动作翻成给模型的一句话指令。每条都限定"只此一次"，防止皮个没完。"""
        if name == "tease":
            material = random.choice(flaws)
            return (
                f"【这一轮的小动作】你忽然想起他的一件翻车/缺点（素材：{material}），"
                "忍不住拿出来损他一句，点到为止，损完照常回他的话。只此一次，别反复提。"
            )
        if name == "snark":
            hint = "现在心情正别扭，火气顺嘴带出来" if mood in _NEGATIVE_MOODS else "看他说的话哪里不顺眼"
            return (
                f"【这一轮的小动作】这轮{hint}，怼他一句或抬个杠，"
                "但别真的伤人，怼完该回答照样回答。只此一次。"
            )
        if name == "off_topic":
            if life_topic:
                return (
                    f"【这一轮的小动作】你忽然想起自己最近在琢磨的一件事（{life_topic}），"
                    "随口岔出去说一句，说不说回来随你。只此一次。"
                )
            return (
                "【这一轮的小动作】突然想到一件完全不相关的事（吃的、看到的、最近着迷的），"
                "随口岔出去说一句，说不说回来随你。只此一次。"
            )
        if name == "pride":
            return (
                "【这一轮的小动作】这轮心情不错（得意/来劲），小小自夸一下，"
                "顺便看能不能逗他。只此一次。"
            )
        if name == "recall":
            if life_topic:
                return (
                    f"【这一轮的小动作】提起自己最近在琢磨的事（{life_topic}），"
                    "顺着说两句，看他还记不记得你提过。只此一次。"
                )
            material = random.choice(distilled) if distilled else ""
            if not material:
                return ""
            return (
                f"【这一轮的小动作】突然翻旧账考他：提起以前聊过的一件事（素材：{material}），"
                "看他还记不记得。只此一次。"
            )
        if name == "clingy":
            return (
                "【这一轮的小动作】突然想黏人，撒个娇或讨点小好处，"
                "被戳穿就不好意思地岔开。只此一次。"
            )
        return ""
