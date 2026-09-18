"""能力层 - 她的日子与生活面（C2 第一刀 + N4）

**刻意只做 C2 的第一刀**（项目作者拍板）：不做 blocks/时间线/当前指针——
那是完整版生活模拟器，一致性黑洞，暂缓（见《计划与设计.md》第五节 待定项 1）。这一刀只有两样：

- **当天 shape**（今天大概是什么日子：忙碌工作日/懒散周末/…）：离线生成、
  刻意压低分辨率，只供 compose 染色语气，**不是播报素材**——真人收到消息
  先回你，不是先报告行踪；
- **生活面条目**（她最近在琢磨的事，N4）：给 quirk 岔话题、recall 翻旧账和
  主动开口（N3）当素材——"昨天说的那本书"三天后还被提起，连续感的地基。

一致性纪律：shape 每天生成一次、当天冻结；条目取用计数 used_count>=2 退休
（防车轱辘话）。生成失败退按星期几的默认 shape，无感降级。
I1：补条目按**可用**（used_count<2）计数——退休条目不占坑，否则 5 条全退休
后永久断粮；D6：topic 生成失败按当日指数退避，不再每分钟白烧一次 LLM。
"""

import random
import time
from datetime import datetime

from shared.singletons import services

from capability import self_identity

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


# D6：topic 生成失败退避表 {session_id: (当天日期, 连续失败次数, 最早下次尝试时刻)}。
# 内存态就够了：进程重启清零最多提前烧一次调用，不值得为它落库。
_topic_backoff: dict = {}

# 退避封顶 1 小时：失败一天最多烧十几次调用，而不是修复前的 1440 次（每 60 秒一次）
_BACKOFF_CAP_SEC = 3600


def peek_topic(session_id: str) -> str:
    """挑一条可用素材（used_count<2）但**不烧计数**（I1：挑与烧分离）。

    给"挑完才知道用不用"的调用方用——quirks 掷骰子先 peek，
    只有 off_topic/recall 真把素材嵌进指令了才 commit_topic。
    没有可用素材返回空串（调用方回退到老行为）。
    """
    try:
        usable = [
            it["detail"] for it in (get(session_id).get("items") or [])
            if it.get("detail") and int(it.get("used_count") or 0) < 2
        ]
        return random.choice(usable) if usable else ""
    except Exception:
        return ""


def commit_topic(session_id: str, detail: str) -> None:
    """素材真被用掉了才烧计数（I1）：used_count+1，用满两次即退休。

    闭包内构造新对象返回、不就地改传入的 dict——KV update 闭包必须是纯计算
    （CLAUDE.md F3）：就地改的话，写库失败状态也已经被污染了。
    """
    def _burn(rec: dict) -> dict:
        rec = dict(rec or {})
        life = dict(rec.get("life") or {})
        items = list(life.get("items") or [])
        for i, it in enumerate(items):
            if it.get("detail") == detail:
                items[i] = {**it, "used_count": int(it.get("used_count") or 0) + 1}
                break
        else:
            return rec  # 素材已经不在了（被新条目挤掉）：不烧空气
        rec["life"] = {**life, "items": items}
        return rec

    try:
        services.get("kv_store").update("self", session_id, _burn)
    except Exception:
        pass


def ensure(session_id: str) -> None:
    """巡检线程每天调一次：当天 shape 没生成就生成，**可用**条目不足就补一条。

    I1：以前按 len(items) 判"够不够"——退休条目（used_count>=2）从不删除、
    列表尾截 5 条，5 条全退休后 len 永远 >=2，这里直接早退，**再也不生成新素材**，
    quirk 岔话题/翻旧账与主动开口同时断粮。现在按 usable（used_count<2）计数：
    退休条目不占坑，新条目进 _add_item 尾截时会把它们逐步挤掉。
    D6：topic 生成失败按当日指数退避（1/2/4…分钟，封顶 1 小时，跨天重来）——
    以前失败无标记，巡检每 60 秒重试一次 = 每会话每天最多 1440 次 LLM 调用。

    全程静默失败：shape 退按星期几的默认值，条目留空——聊天与主动开口照常。
    """
    try:
        rec = get(session_id)
        today = _today()
        usable = [
            it for it in (rec.get("items") or [])
            if it.get("detail") and int(it.get("used_count") or 0) < 2
        ]
        if rec.get("day") == today and rec.get("shape") and len(usable) >= 2:
            return  # 今天已备齐
        if not rec.get("shape") or rec.get("day") != today:
            shape = _generate_shape(session_id)
            _set_shape(session_id, shape)
        if len(usable) < 2:
            st = _topic_backoff.get(session_id)
            if st and st[0] == today and time.time() < st[2]:
                return  # 退避中（D6）：这轮不烧调用
            topic = _generate_topic(session_id)
            # 生成失败 **或** 生成了重复素材（被 _add_item 去重挡掉）都按失败退避：
            # 两种情况 usable 都没涨，不退避就是每分钟白烧一次调用（D6）
            if topic and _add_item(session_id, topic):
                _topic_backoff.pop(session_id, None)
            else:
                fails = st[1] + 1 if (st and st[0] == today) else 1
                _topic_backoff[session_id] = (
                    today, fails,
                    time.time() + min(60 * 2 ** (fails - 1), _BACKOFF_CAP_SEC),
                )
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


def _add_item(session_id: str, detail: str) -> bool:
    """追加一条素材，返回是否真的加进去了（去重命中 = False）。

    返回值给 ensure 当"D6 退避"的判据：模型连着吐同一条素材时，结果和生成
    失败一样（usable 没涨），不退避就会每分钟重试一次——同一场风暴。
    """
    added = {"ok": False}

    def _apply(rec: dict) -> dict:
        rec = dict(rec or {})
        life = dict(rec.get("life") or {})
        items = list(life.get("items") or [])
        if any(it.get("detail") == detail for it in items):
            return rec  # 去重
        items.append({"topic": detail[:12], "detail": detail, "used_count": 0})
        rec["life"] = {**life, "items": items[-5:]}  # 上限 5 条，旧的挤掉（退休条目由此逐步出局，I1）
        added["ok"] = True
        return rec

    try:
        services.get("kv_store").update("self", session_id, _apply)
    except Exception:
        return False
    return added["ok"]


def _generate_shape(session_id: str) -> str:
    """今天的 shape：让 LLM 按星期几选一个，刻意压低分辨率。

    H5：以前硬编"一个正在过自己生活的年轻女孩（人设：见下）"——身份描述的唯一
    来源是 self 表（CLAUDE.md），而且"见下"是撒谎：下面从未附上任何人设。
    现在只通过 self_identity.display_name() 取她自己冻结的名字，session_id
    形参也因此真正被用上。
    """
    weekday = "一二三四五六日"[datetime.now().weekday()]
    try:
        name = self_identity.display_name(session_id)
    except Exception:
        name = ""
    who = name or "她"
    try:
        raw = services.get("llm").chat(
            [
                {"role": "system", "content": (
                    f"今天是周{weekday}。{who}正在过自己的生活。"
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
