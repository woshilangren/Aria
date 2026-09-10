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

# 平常时心情池：正面情绪往这漂
_POSITIVE_MOODS = ("得意", "雀跃", "来劲", "心软")
# 负面情绪往这漂
_NEGATIVE_MOODS = ("别扭", "低落", "慵懒", "心烦")

# 撒娇解锁的亲密度门槛：刚认识就撒娇就太油腻了
_CLINGY_INTIMACY = 20


class MoodEngine:
    """心情状态机：大多数时候维持原样，偶尔漂一格。

    心情只影响"怎么说话"，不影响"说什么"——别让随机性盖过正经回答。
    """

    # 漂移概率：一轮里约三成机会换个心情，太频繁就神经质了
    _DRIFT_RATE = 0.3

    def update(self, rel: dict, emotion: str = "neutral", comfort_mode: bool = False) -> str:
        """根据上轮心情 + 本轮情绪算新心情，返回心情标签（不落库，调用方负责写回）。"""
        # 安抚和危机轮强制心软，随机性给正经事让路
        if comfort_mode or emotion == "crisis":
            return "心软"

        current = rel.get("mood") or "平常"
        if random.random() >= self._DRIFT_RATE:
            return current

        if emotion in ("sad", "angry", "tired", "anxious"):
            pool = _NEGATIVE_MOODS
        elif emotion in ("happy",):
            pool = _POSITIVE_MOODS
        else:
            pool = _POSITIVE_MOODS + _NEGATIVE_MOODS
        picked = random.choice(pool)
        # 转一圈又转回原样的没意思，等于白漂，直接给"平常"表示翻篇了
        return "平常" if picked == current else picked


class QuirkDirector:
    """小动作骰子：每轮按概率掷一次，命中了挑一个小动作，输出注入指令。

    权重设计：翻缺点调侃 > 怼一句 > 岔开话题 > 提旧事 > 得意自夸 > 撒娇。
    撒娇要亲密度过门槛才上桌；素材（缺点、旧事）没有的动作自动缺席。
    """

    def roll(self, memory: MemoryBundle, stage: str = "初识",
             mood: str = "平常", comfort_mode: bool = False) -> str:
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

        # (动作名, 权重)，素材齐不齐决定了哪些动作能上桌
        actions = [("snark", 3), ("off_topic", 2), ("pride", 2)]
        if flaws:
            actions.append(("tease", 4))
        if distilled:
            actions.append(("recall", 2))
        if stage in ("亲近", "挚友"):
            actions.append(("clingy", 2))

        total = sum(w for _n, w in actions)
        pick = random.uniform(0, total)
        for name, weight in actions:
            pick -= weight
            if pick <= 0:
                return self._directive(name, flaws, distilled, mood)
        return ""

    def _directive(self, name: str, flaws: list, distilled: list, mood: str) -> str:
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
            material = random.choice(distilled)
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
