"""能力层 - 她的日子与生活面（C2 第一刀 + N4）

**刻意只做 C2 的第一刀**（项目作者拍板）：不做 blocks/时间线/当前指针——
那是完整版生活模拟器，一致性黑洞，暂缓（见 方案.md 待定 1）。这一刀只有两样：

- **当天 shape**（今天大概是什么日子：忙碌工作日/懒散周末/…）：离线生成、
  刻意压低分辨率，只供 compose 染色语气，**不是播报素材**——真人收到消息
  先回你，不是先报告行踪；
- **生活面条目**（她最近在琢磨的事，N4）：给 quirk 岔话题、recall 翻旧账和
  主动开口（N3）当素材——"昨天说的那本书"三天后还被提起，连续感的地基。

一致性纪律：shape 每天生成一次、当天冻结；条目取用计数 used_count>=2 退休
（防车轱辘话）。生成失败退按星期几的默认 shape，无感降级。
"""

from datetime import datetime

from shared.singletons import services
from tools.misc import ClockTool, parse_llm_json

_SHAPES = ("忙碌工作日", "懒散周末", "朋友日", "加班日", "平常的一天")

_WEEKDAY_DEFAULT = {
    0: "忙碌工作日", 1: "忙碌工作日", 2: "平常的一天", 3: "平常的一天",
    4: "平常的一天", 5: "懒散周末", 6: "懒散周末",
}


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def get(session_id: str) -> dict:
    """读她今天的日子：{day, shape, items:[{topic, detail, used_count}]}，坏数据退空。"""
    try:
        rec = (services.get("kv_store").read("self", session_id) or {}).get("life") or {}
        return rec if isinstance(rec, dict) else {}
    except Exception:
        return {}


def shape_line(session_id: str) -> str:
    """compose 注入用：今天的 shape 染色行。过期/缺失返回空串（不注入）。"""
    rec = get(session_id)
    if rec.get("day") != _today() or not rec.get("shape"):
        return ""
    return (
        f"【你今天】今天大概是个「{rec['shape']}」。让语气带上这个状态，"
        "但别主动播报你在干嘛——他问起时顺口说就行。"
    )


def consume_topic(session_id: str) -> str:
    """取一条生活面素材给 quirk/主动开口用：取用计数+1，用满两次的退休。

    没有可用素材返回空串（调用方回退到老行为）。
    """
    try:
        picked = {"topic": ""}

        def _take(rec: dict) -> dict:
            rec = rec or {}
            life = rec.get("life") or {}
            items = [it for it in (life.get("items") or []) if it.get("detail")]
            usable = [it for it in items if int(it.get("used_count") or 0) < 2]
            if not usable:
                return rec
            import random

            it = random.choice(usable)
            it["used_count"] = int(it.get("used_count") or 0) + 1
            picked["topic"] = it["detail"]
            rec["life"] = life
            return rec

        services.get("kv_store").update("self", session_id, _take)
        return picked["topic"]
    except Exception:
        return ""


def ensure(session_id: str) -> None:
    """巡检线程每天调一次：当天 shape 没生成就生成，条目不足就补一条。

    全程静默失败：shape 退按星期几的默认值，条目留空——聊天与主动开口照常。
    """
    try:
        rec = get(session_id)
        if rec.get("day") == _today() and rec.get("shape") and len(rec.get("items") or []) >= 2:
            return  # 今天已备齐
        if not rec.get("shape") or rec.get("day") != _today():
            shape = _generate_shape(session_id)
            _set_shape(session_id, shape)
        if len((get(session_id)).get("items") or []) < 2:
            topic = _generate_topic(session_id)
            if topic:
                _add_item(session_id, topic)
    except Exception:
        pass  # 生活面是增强项


def _set_shape(session_id: str, shape: str) -> None:
    def _apply(rec: dict) -> dict:
        rec = rec or {}
        life = rec.get("life") or {}
        life["day"] = _today()
        life["shape"] = shape
        rec["life"] = life
        return rec

    services.get("kv_store").update("self", session_id, _apply)


def _add_item(session_id: str, detail: str) -> None:
    def _apply(rec: dict) -> dict:
        rec = rec or {}
        life = rec.get("life") or {}
        items = life.get("items") or []
        if any(it.get("detail") == detail for it in items):
            return rec  # 去重
        items.append({"topic": detail[:12], "detail": detail, "used_count": 0})
        life["items"] = items[-5:]  # 上限 5 条，旧的挤掉
        rec["life"] = life
        return rec

    services.get("kv_store").update("self", session_id, _apply)


def _generate_shape(session_id: str) -> str:
    """今天的 shape：让 LLM 按人设和星期几选一个，刻意压低分辨率。"""
    weekday = "一二三四五六日"[datetime.now().weekday()]
    try:
        raw = services.get("llm").chat(
            [
                {"role": "system", "content": (
                    f"今天是周{weekday}。她是一个正在过自己生活的年轻女孩（人设：见下）。"
                    "从这些里选一个作为她今天的总基调："
                    f"{'/'.join(_SHAPES)}。只输出这个名字本身，别加任何字。"
                )},
                {"role": "user", "content": "今天她大概是什么日子？"},
            ],
            temperature=0.7,
            max_tokens=12,
        )
        shape = (raw or "").strip()[:10]
        if shape in _SHAPES:
            return shape
    except Exception:
        pass
    return _WEEKDAY_DEFAULT[datetime.now().weekday()]


def _generate_topic(session_id: str) -> str:
    """一条"她最近在琢磨的事"：必须具体（书名/菜名/小目标），30 字内。"""
    try:
        persona = services.get("kv_store").read("persona_config", "")
        seed = (persona.background_story or "")[:200] if persona else ""
        raw = services.get("llm").chat(
            [
                {"role": "system", "content": (
                    "以她的口吻想一件她最近在琢磨的小事（在看的书/想吃的馆子/"
                    "一个小目标/一个观察），必须具体、有实体词，30 字以内，"
                    "第一人称，只输出这一句话。"
                    + (f"\n她的大致背景：{seed}" if seed else "")
                )},
                {"role": "user", "content": "想一件。"},
            ],
            temperature=0.9,
            max_tokens=60,
        )
        detail = (raw or "").strip().strip('"「」')
        return detail[:40] if detail else ""
    except Exception:
        return ""
