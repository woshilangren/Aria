"""能力层 - 闲置主动关怀

伴侣不能只会被动接话：用户聊完就走、闲了半个多小时，该自己把今天
的对话整理成日记收好。这个模块就是后台巡检员，每隔一会儿看一眼
"谁闲太久了还没写日记"，该写就写，写完不声张。
"""

import random
import threading
from datetime import datetime, timedelta

from config.settings import load_app_config
from shared.singletons import services

from tools.misc import ClockTool

from capability import char_life, self_identity


class ProactiveSpeaker:
    """主动开口（N3 + S6 精化）：她也会先说话——但宁缺毋滥。

    触发四层闸（全部可配，proactive.enabled 默认关）：
    1. **时段窗口**：不查墙上时钟，从 chat_log 学"这个人活跃在几点"（近 14 天
       按小时直方图）。作息越乱 → 窗口越宽 → 置信越低 → 越不敢冷 ping。
       数据不足 14 天 → 只走搭便车（跳过）。
    2. **关系门控**：亲密度 < 20（初识）不主动——主动是处出来的。
    3. **节流**：每天最多 daily_max 条；连续 ignored_limit 次没被回 → 当天闭嘴
       （收手环：主动是让人关掉应用最快的方式）。
    4. **内容有据**：素材取自生活面（char_life）或关系氛围线——"刚下楼买了杯
       咖啡巨难喝"和"想起你上次说的那件事"，绝不说"在吗"。

    交付通道：生成的消息写入 chat_log（intent='proactive'），前端轮询
    /api/proactive/poll 拉取——落库保证刷新后历史一致（说过的不会消失）。
    """

    _IGNORE_LIMIT = 3

    def __init__(self):
        self._lock = threading.Lock()
        self._last_sent_day = ""
        self._sent_today = 0

    # ---- 闸门 ----
    def _active_window(self, session_id: str) -> bool:
        """当前小时是否落在"这个人大概率醒着"的窗口里。数据不足返回 None（不冷 ping）。"""
        try:
            rows = services.get("kv_store").last_chat_per_session()
            _ = rows  # 占位：小时直方图要从 chat_log 拿，见 _hour_histogram
        except Exception:
            return None
        hist = self._hour_histogram(session_id)
        if hist is None or sum(hist.values()) < 20:
            return None  # 作息数据不足（<20 条），保守：不冷 ping
        hours = sorted(hist.keys())
        # 窗口 = 活跃小时的并集；当前小时命中率越高越敢开口。
        # 作息乱（活跃小时多）→ 命中率天然被摊薄 → 越不敢开口，这正是想要的
        hit_rate = (hist.get(datetime.now().hour, 0) / max(1, sum(hist.values()))) * len(hours)
        return hit_rate >= 0.4 or datetime.now().hour in hours

    def _hour_histogram(self, session_id: str) -> "dict | None":
        """近 14 天 chat_log 的小时分布 {hour: count}；拿不到返回 None。

        P0-2 修复（codex 2026-09-17 审查指出）：以前 `services.get("kv_store") or KVStoreTool()`
        在 kv_store 未注册时静默 new 第二份 sqlite 连接——绕开 CLAUDE.md "KEEPER 全项目
        一份"的不变式（虽然字面是 KEEPER，但 kv_store 同样应是单例）。
        现在的策略：拿不到就 return None，让上层 `_active_window` 走保守分支（数据不足不冷 ping），
        而不是悄悄开个新连接。`KVStoreTool` 仍由 `bootstrap.py` 装配，缺了就该响在 startup，
        不是这里二开。
        """
        from shared.singletons import services

        kv = services.get("kv_store")
        if kv is None:
            return None
        try:
            return kv.chat_log_hour_distribution(session_id, days=14)
        except Exception:
            return None

    def _gap_days(self, session_id: str) -> float:
        try:
            rel = services.get("kv_store").read("relationship", session_id) or {}
            last = rel.get("last_interaction") or ""
            return max(0.0, (datetime.now() - datetime.fromisoformat(last)).total_seconds() / 86400)
        except Exception:
            return 0.0

    def _gate(self, session_id: str) -> bool:
        cfg = load_app_config().get("proactive", {})
        if not cfg.get("enabled", False):
            return False
        if self._sent_today >= int(cfg.get("daily_max", 2) or 2):
            return False
        try:
            rel = services.get("kv_store").read("relationship", session_id) or {}
        except Exception:
            return False
        if rel.get("consecutive_ignored", 0) >= self._IGNORE_LIMIT:
            return False  # 收手环：连着没被理，今天闭嘴
        if rel.get("proactive_awaiting"):
            return False  # 上一条主动消息还没被回应：等，别连击
        intimacy = rel.get("intimacy", 0) or 0
        if intimacy < int(cfg.get("min_intimacy", 20)):
            return False  # 初识不主动
        gap_h = self._gap_days(session_id) * 24
        window = self._active_window(session_id)
        if gap_h >= 2 and window is False:
            return False  # 真隔开了时间还不在他的活跃窗口：不吵
        return True

    # ---- 生成 ----
    def _compose(self, session_id: str, kind: str) -> str:
        from capability.memory import recent_arc

        life = ""
        try:
            items = (char_life.get(session_id).get("items") or [])
            usable = [it["detail"] for it in items if int(it.get("used_count") or 0) < 2]
            if usable:
                life = random.choice(usable)
        except Exception:
            life = ""
        arc = ""
        try:
            arc = recent_arc(session_id)
        except Exception:
            arc = ""
        name = self_identity.display_name(session_id)
        prompt = (
            f"你是{name}，一个正在过自己生活的年轻女孩。现在你要**主动**给他发一条消息（他在等不等于在线）。\n"
            f"时机：{kind}。\n"
            + (f"你最近的日子：{arc}\n" if arc else "")
            + (f"你最近在琢磨的事：{life}\n" if life else "")
            + "要求：一句话，30 字以内，口语，具体、有据（用上面给你的素材，别硬凑也别说教）；"
            "绝不问'在吗''在干嘛'这种空话；不许提任何机器、系统、设定的字眼。只输出这一句话。"
        )
        try:
            text = (services.get("llm").chat(
                [{"role": "user", "content": prompt}], temperature=0.9, max_tokens=60
            ) or "").strip()
        except Exception:
            return ""  # 生成失败就不发——宁可沉默，不像人就别硬发
        return text[:60]

    def _record_sent(self, session_id: str) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            if self._last_sent_day != today:
                self._last_sent_day = today
                self._sent_today = 0
            self._sent_today += 1

    def tick(self, session_id: str) -> str:
        """巡检线程每分钟调一次：全部闸门过了才生成并发一条。返回发出的文本（否则空）。"""
        try:
            if not self._gate(session_id):
                return ""
            now = datetime.now()
            gap_h = self._gap_days(session_id) * 24
            if gap_h < 2:
                return ""  # 刚聊完没多久，轮不到主动开口
            if 6 <= now.hour < 11:
                kind = "隔了一晚的早上，自然地打招呼（像想起他了一样）"
            elif 23 <= now.hour or now.hour < 2:
                kind = "深夜，随口一句自己的状态或想到的事（绝不催他睡觉、不问他在不在）"
            elif gap_h >= 1:
                kind = "隔了半天以上，想起他或想起上次聊的事，自然地接上话头"
            else:
                return ""
            text = self._compose(session_id, kind)
            if not text:
                return ""
            self._persist(session_id, text)
            self._record_sent(session_id)
            return text
        except Exception:
            return ""

    def _persist(self, session_id: str, text: str) -> None:
        """主动消息落 chat_log（intent='proactive'）：刷新后历史一致。
        同时置"等待回应"标记——用户回复时由 pipeline 清掉，被晾 6 小时则计一次忽视。"""
        try:
            services.get("kv_store").write(
                "session", session_id,
                {"role": "assistant", "text": text, "intent": "proactive",
                 "emotion": "", "mode": "proactive"},
            )
            services.get("kv_store").update(
                "relationship", session_id,
                lambda rel: {**rel, "proactive_awaiting": True,
                             "proactive_at": ClockTool().now()},
            )
        except Exception:
            pass


# 主动消息的"已回"判定：用户回复后重置收手环（在 pipeline 写回时调）
def note_user_reply(session_id: str) -> None:
    """用户开口说话 = 她的主动被回应了：清等待标记、重置连续未回计数。

    关系数据还不存在时是 no-op——绝不预创建空 relationship（那会让
    tracker 的"首次初始化"分支失效，default_intimacy 被吞成 0）。
    """
    try:
        def _reset(rel: dict) -> dict:
            if not rel:
                return rel
            return {**rel, "consecutive_ignored": 0, "proactive_awaiting": False}

        services.get("kv_store").update("relationship", session_id, _reset)
    except Exception:
        pass


def note_proactive_ignored(session_id: str) -> None:
    """她又主动了一条但没被理（下次用户说话前由巡检累计）：连击+1。"""
    try:
        services.get("kv_store").update(
            "relationship", session_id,
            lambda rel: {**rel,
                         "consecutive_ignored": int(rel.get("consecutive_ignored") or 0) + 1},
        )
    except Exception:
        pass


class IdleDiaryWatcher:
    """闲置太久自动写日记的后台看门人。

    起一条守护线程，每分钟醒一次点名：哪个会话的最后一条聊天
    超过 idle_minutes 还没给它写过当天的日记，就补一篇；
    顺带照料她的生活面（char_life）与主动开口（ProactiveSpeaker）。
    """

    # 巡检节奏：一分钟看一眼足够了，日记不赶时间
    _CHECK_INTERVAL = 60

    def __init__(self):
        self._done = set()      # 已经写过的 (session_id, 当天日期)，防止反复重写
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._speaker = ProactiveSpeaker()

    def start(self) -> None:
        """把巡检线程跑起来，重复调用没副作用。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="idle-diary-watcher"
        )
        self._thread.start()

    def stop(self) -> None:
        """叫停巡检（进程退出时守护线程本来也会跟着走，这接口主要给测试用）。"""
        self._stop.set()

    def _loop(self) -> None:
        # Event.wait 兼当 sleep 和停机开关：stop() 一喊立马醒过来退出
        while not self._stop.wait(self._CHECK_INTERVAL):
            try:
                self._sweep()
            except Exception:
                pass  # 巡检挂一轮无所谓，下一轮接着来

    def _sweep(self) -> None:
        """点名一遍所有会话，闲够久的补写当天日记；顺带照料她的生活面（C2 第一刀）。"""
        idle_minutes = load_app_config()["proactive"].get("idle_minutes", 30)
        threshold = datetime.now() - timedelta(minutes=idle_minutes)

        for entry in services.get("kv_store").last_chat_per_session():
            last_time = (entry.get("last_time") or "")[:19]
            if not last_time:
                continue
            try:
                last_dt = datetime.fromisoformat(last_time)
            except ValueError:
                continue  # 时间戳格式不对就跳过，别为一条脏数据惊动整个巡检

            session_id = entry["session_id"]
            # 她的日子（C2 第一刀/N4）：每个有点动静的会话，每天生成一次 shape
            # 与生活面素材——独立 try，日记失败不影响它，反之亦然
            try:
                char_life.ensure(session_id)
            except Exception:
                pass
            # 主动开口（N3）：全部闸门在 tick 里过，没过就安静返回
            try:
                self._speaker.tick(session_id)
            except Exception:
                pass

            if last_dt >= threshold:
                continue  # 还没闲够，下轮再看

            # 日记跟着最后聊天的日期走：闲超时往往跨了零点，日期不能想当然用今天
            date = last_time[:10]
            if (session_id, date) in self._done:
                continue  # 这天的日记已经补过了，别反复让模型白写
            # skip_if_exists：进程重启 _done 清零也不怕，在册的日记直接跳过
            diary = services.get("diary_writer").write_diary(session_id, date, skip_if_exists=True)
            if diary:
                self._done.add((session_id, date))
