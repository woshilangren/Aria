"""能力层 - 闲置主动关怀

伴侣不能只会被动接话：用户聊完就走、闲了半个多小时，该自己把今天
的对话整理成日记收好。这个模块就是后台巡检员，每隔一会儿看一眼
"谁闲太久了还没写日记"，该写就写，写完不声张。
"""

import random
import threading
import time
from datetime import datetime, timedelta

import logging

logger = logging.getLogger("aria")

from config.settings import load_app_config
from shared.singletons import services
from shared.timeutils import safe_delta_seconds

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
       （收手环：主动是让人关掉应用最快的方式）。⚠ I7 如实标注：收手环的计数
       入口 note_proactive_ignored 目前全仓无调用点，环还没转起来——见其注释。
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
    def _active_window(self, session_id: str) -> "bool | None":
        """当前小时是否落在"这个人大概率醒着"的窗口里。数据不足返回 None（不冷 ping）。

        I8：返回类型以前标 `-> bool` 却会返回 None——类型谎言，和 _gate 把 None
        当放行是同一个坑，一起修。
        """
        hist = self._hour_histogram(session_id)
        if hist is None or sum(hist.values()) < 20:
            return None  # 作息数据不足（<20 条），保守：不冷 ping
        hours = sorted(hist.keys())
        # 窗口 = 活跃小时的并集；当前小时命中率越高越敢开口。
        # 作息乱（活跃小时多）→ 命中率天然被摊薄 → 越不敢开口，这正是想要的。
        # hit_rate 语义：相对"均匀活跃"的比值（均匀分布在活跃小时上 = 1.0）。
        # I8：以前是 `hit_rate >= 0.4 or hour in hours`——那个 or 让该小时 14 天内
        # 有过哪怕一条记录就放行，命中率门槛形同虚设，与上面注释的意图正好相反。
        hit_rate = (hist.get(datetime.now().hour, 0) / max(1, sum(hist.values()))) * len(hours)
        return hit_rate >= 0.4

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
            # R09a：统一解析（Z / naive 本地语义 / 坏值降级都收口在 timeutils）
            gap = safe_delta_seconds(datetime.now(), last)
            return max(0.0, (gap or 0.0) / 86400)
        except Exception:
            return 0.0

    def _gate(self, session_id: str) -> bool:
        cfg = load_app_config().get("proactive", {})
        if not cfg.get("enabled", False):
            return False
        # I6：跨天滚动放在闸门最前面——以前 _sent_today 只在 _record_sent（发送
        # 成功后）里清零：昨天满额 → 今天 _gate 永远 False → 不重启进程永不复位。
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            if self._last_sent_day != today:
                self._last_sent_day = today
                self._sent_today = 0
            sent = self._sent_today
        if sent >= int(cfg.get("daily_max", 2) or 2):
            return False
        try:
            rel = services.get("kv_store").read("relationship", session_id) or {}
        except Exception:
            return False
        if rel.get("consecutive_ignored", 0) >= self._IGNORE_LIMIT:
            return False  # 收手环：连着没被理，今天闭嘴（⚠ 计数入口当前无调用点，见 note_proactive_ignored）
        if rel.get("proactive_awaiting"):
            return False  # 上一条主动消息还没被回应：等，别连击
        intimacy = rel.get("intimacy", 0) or 0
        if intimacy < int(cfg.get("min_intimacy", 20)):
            return False  # 初识不主动
        gap_h = self._gap_days(session_id) * 24
        window = self._active_window(session_id)
        # I8：None（作息数据不足）也按拦截——_active_window 的注释一直写着
        # "数据不足不冷 ping"，以前 `window is False` 却把 None 放行了，正好相反。
        if gap_h >= 2 and window is not True:
            return False  # 真隔开了时间还不在他的活跃窗口（或压根不知道他几点醒）：不吵
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
        # H5：身份描述不硬编（"年轻女孩"之类）——她的身份唯一来源是 self 表，
        # 这里只给名字；"在过自己的生活"由素材段（arc/life）自己带出来
        prompt = (
            f"你是{name}。现在你要**主动**给他发一条消息（他在等不等于在线）。\n"
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
        """巡检线程每分钟调一次：全部闸门过了才生成并发一条。返回发出的文本（否则空）。

        ⚠ 现状与原则的冲突（I8，本批不改行为、留给作者定）：下面的时段分支按
        **墙上时钟小时桶**触发，由 _sweep 每 60 秒调一次——这就是定时触发，
        与 CLAUDE.md"主动性宁缺毋滥：事件触发 + 用户在场触发，绝不定时冷 ping"
        冲突，目前只是被 proactive.enabled 默认关掩盖着。改成真事件驱动
        （如检测到用户在场信号才开口）是产品决策，见 计划与设计.md I8。
        """
        try:
            if not self._gate(session_id):
                return ""
            now = datetime.now()
            gap_h = self._gap_days(session_id) * 24
            if gap_h < 2:
                return ""  # 刚聊完没多久，轮不到主动开口
            # I8：以前这里还有 `elif gap_h >= 1: ... else: return ""`——上面已保证
            # gap_h >= 2，else 永不可达，删掉（注释不许描述不存在的路径）
            if 6 <= now.hour < 11:
                kind = "隔了一晚的早上，自然地打招呼（像想起他了一样）"
            elif 23 <= now.hour or now.hour < 2:
                kind = "深夜，随口一句自己的状态或想到的事（绝不催他睡觉、不问他在不在）"
            else:
                kind = "隔了半天以上，想起他或想起上次聊的事，自然地接上话头"
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

        同时置 proactive_awaiting"等待回应"标记 + proactive_at 时间戳。
        收手环（I7）在 `note_user_reply` 里闭合：用户回复时同一个原子闭包判
        "是不是被晾超了 6 小时"，超了记一次 consecutive_ignored、不清零连击，
        _gate 靠那个计数压后续主动频率。这里只负责把标记和时间戳落准。
        """
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


# 她主动开口后，多久没被理就算"晾了一次"（设计口径 6 小时，见 计划与设计.md I7）
_IGNORE_AFTER_SEC = 6 * 3600


def _bump_ignored(rel: dict) -> dict:
    """收手环 +1（纯计算，不改入参）：她又主动了一条但一直没被理。"""
    return {**rel, "consecutive_ignored": int(rel.get("consecutive_ignored") or 0) + 1}


def _awaiting_too_long(rel: dict) -> bool:
    """她是不是已经在等回应、而且等超了阈值。

    时间戳读不出来/格式不对一律按"没超"处理——收手环是用来**减少**她主动的，
    判错的方向应该是少记一次忽视，不是多记一次让她闭嘴。
    """
    if not rel.get("proactive_awaiting"):
        return False
    at = rel.get("proactive_at") or ""
    # R09a：统一解析；解析失败（没记录过/坏值）按"没在收手期"处理
    elapsed = safe_delta_seconds(datetime.now(), at)
    if elapsed is None:
        return False
    return elapsed >= _IGNORE_AFTER_SEC


# 主动消息的"已回"判定：用户回复后重置收手环（在 pipeline 写回时调）
def note_user_reply(session_id: str) -> None:
    """用户开口说话 = 她的主动被回应了：清等待标记、重置连续未回计数。

    I7 的收手环在这里闭合：**同一个原子闭包**里先判"她是不是被晾超了 6 小时"，
    超了就记一次忽视、且这一次不清零连击；没超才照常清零。
    之所以不拆成"先调 note_proactive_ignored 再调 note_user_reply"——那是三次
    库操作（读一次、写两次），中间有竞态窗口，还多一条"顺序不能反，先清就读不到
    awaiting 了"的隐形纪律（这项目已经栽过 note_user_reply 顺序一次）。
    一个闭包里做完，顺序问题根本不存在。

    关系数据还不存在时是 no-op——绝不预创建空 relationship（那会让
    tracker 的"首次初始化"分支失效，default_intimacy 被吞成 0）。
    """
    try:
        def _reset(rel: dict) -> dict:
            if not rel:
                return rel
            if _awaiting_too_long(rel):
                # 被晾够了才回：这一次算忽视，连击不清零（_gate 靠它压主动频率）
                return {**_bump_ignored(rel), "proactive_awaiting": False}
            return {**rel, "consecutive_ignored": 0, "proactive_awaiting": False}

        services.get("kv_store").update("relationship", session_id, _reset)
    except Exception:
        pass


def note_proactive_ignored(session_id: str) -> None:
    """收手环 +1：她又主动了一条但一直没被理。

    I7 现状：**pipeline 侧的收手环已经闭合了**（见 note_user_reply，用户回复时
    在同一个原子闭包内判超时并计数），所以这个函数不再是"唯一入口"，而是留给
    另一条候选路径——巡检侧定时扫 proactive_at 超时未回。那条要不要做属于产品
    决策（与 I8 冷 ping 定性同一批），**目前没有调用点**。
    在作者拍板之前别删：它和 note_user_reply 共用 _bump_ignored，接上就是零成本。
    """
    try:
        def _count(rel: dict) -> dict:
            if not rel:
                return rel  # 同 note_user_reply：绝不预创建空 relationship（会吞 default_intimacy）
            return _bump_ignored(rel)

        services.get("kv_store").update("relationship", session_id, _count)
    except Exception:
        pass


class IdleDiaryWatcher:
    """闲置太久自动写日记的后台看门人。

    起一条守护线程，每分钟醒一次点名：哪个会话的最后一条聊天
    超过 idle_minutes 还没给它写过当天的日记，就补一篇；
    顺带照料她的生活面（char_life）与主动开口（ProactiveSpeaker），
    以及记忆候选池的 14 天过期清理（E1，一天一次）。
    """

    # 巡检节奏：一分钟看一眼足够了，日记不赶时间
    _CHECK_INTERVAL = 60

    # 日记失败退避封顶 1 小时（I2-4）：以前 _save 失败 → _done 不写入 →
    # 每 60 秒重试一轮、每轮白烧 LLM 调用；现在 1/2/4…分钟指数退避
    _DIARY_BACKOFF_CAP_SEC = 3600

    # R19c：补偿任务消费参数。每轮巡检最多领 3 条、租约 5 分钟、退避从 1 分钟
    # 起指数翻倍封顶 1 小时、预算 5 次——耗尽标 exhausted（保留载荷），格式坏
    # 直接 blocked，都不自动删除，不无限烧 API。
    _PW_BATCH = 3
    _PW_LEASE_SECONDS = 300
    _PW_MAX_ATTEMPTS = 5
    _PW_BACKOFF_BASE = 60
    _PW_BACKOFF_CAP = 3600

    def __init__(self):
        # I2-1：_done 从"写没写过 (session_id, 日期)"的集合改成
        # {(session_id, 日期): 已覆盖到的最后聊天时间}——幂等键加上"是否已覆盖
        # 当天新内容"，不然上午写过日记后，下午的对话永远进不了日记（F13）。
        # 仍是**实例级**去重（CLAUDE.md 取舍 12：双 lifespan 双实例会重复写，
        # 靠 main.py 的一次装配守卫兜底）。
        self._done: dict = {}
        # I2-4：{(session_id, 日期): (连续失败次数, 最早下次重试时刻)}
        self._diary_fails: dict = {}
        # E1：候选池过期清理的"今天做过没"标记
        self._last_prune_day = ""
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
        # K5：.get 兜住段缺失——load_app_config() 有内置默认值，但 config.json
        # 被写坏走降级时整个 "proactive" 段可能不存在，直接索引会 KeyError 停摆巡检
        idle_minutes = load_app_config().get("proactive", {}).get("idle_minutes", 30)
        kv = services.get("kv_store")

        self._prune_candidates_once(kv)

        # R19c：补偿任务消费复用本巡检调度点（不另起线程）——每轮只领一小批
        # 到期任务，绝不一次冲刷所有历史；任务在数据库锁外执行。
        try:
            self._drain_pending_writes(kv)
        except Exception as exc:
            logger.warning(f"[proactive] 补偿任务消费失败（下轮再试）: {exc}")

        for entry in kv.last_chat_per_session():
            last_time = entry.get("last_time") or ""
            if not last_time:
                continue
            # R09a：统一解析；**不得 [:19] 截断**——那会把 "+08:00" 偏移切掉，
            # 同一时刻两种写法算出两个答案。比较改用受保护的秒差
            idle_seconds = safe_delta_seconds(datetime.now(), last_time)
            if idle_seconds is None:
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

            if idle_seconds < idle_minutes * 60:
                continue  # 还没闲够，下轮再看

            self._write_idle_diary(session_id, last_time)

    def _drain_pending_writes(self, kv) -> None:
        """消费 pending_writes 里的到期补偿任务（R19c）。

        只处理已到期且未持租约的行，每轮一小批；成功标 done，失败按次数
        指数退避、耗尽标 exhausted，坏载荷/未知种类标 blocked——全都不自动
        删除业务载荷。执行在巡检线程里、数据库锁外（租约已由领取事务落下）。
        """
        claimed = kv.lease_pending(limit=self._PW_BATCH,
                                   lease_seconds=self._PW_LEASE_SECONDS)
        for task in claimed:
            self._apply_pending_task(kv, task)

    def _apply_pending_task(self, kv, task: dict) -> None:
        task_id = task["task_id"]
        kind = task["kind"]
        payload = task.get("payload")
        if payload is None:
            kv.block_pending(task_id, "payload 不可解析（永久格式错误）")
            return
        try:
            if kind == "vector_memory":
                from data.schemas import MemoryItem

                item = MemoryItem(**payload)
                if services.get("vector_store").upsert_memory(item) is True:
                    kv.finish_pending(task_id)
                    return
                raise RuntimeError("upsert 未成功（返回非 True）")
            if kind == "session_summarize":
                # 延迟 import 破环：proactive 不在顶层依赖 orchestration
                from orchestration.pipeline import KEEPER

                msgs = payload.get("msgs") or []
                if KEEPER._summarize(payload.get("session_id", ""), msgs):
                    kv.finish_pending(task_id)
                    return
                raise RuntimeError("summarize 未成功")
            # 未知种类：不烧 API，保留载荷待人工
            kv.block_pending(task_id, f"未知任务种类：{kind}")
            return
        except TypeError as exc:
            # 载荷结构与当前代码不匹配属于永久错误，重试无意义
            kv.block_pending(task_id, f"载荷重构失败（永久格式错误）: {exc}")
            return
        except Exception as exc:
            backoff = min(
                self._PW_BACKOFF_CAP,
                self._PW_BACKOFF_BASE * (2 ** max(0, task.get("attempts", 1) - 1)),
            )
            state = kv.fail_pending(task_id, str(exc),
                                    backoff_seconds=backoff,
                                    max_attempts=self._PW_MAX_ATTEMPTS)
            logger.warning(f"[proactive] 补偿任务失败（{task_id} → {state}，"
                           f"退避 {backoff}s）: {exc}")

    def _prune_candidates_once(self, kv) -> None:
        """E1：memory_candidates 的 14 天过期清理，挂巡检、一天只做一次。

        为什么必须接上：prune_candidates 此前全仓零调用者——CLAUDE.md 承诺的
        "候选池 14 天过期"从未运行过一次，池子无界增长，几个月前的一条幻觉
        只要重现一次就能凑满晋升计数。清理条数进日志，让"闸真的在跑"有据可查。

        分层纪律（K7）：不直摸 data 层，走 kv_store 门面（`KVStoreTool.prune_candidates`）。
        getattr 探测保留着不是为了等门面——门面已经有了——而是为了让"门面被谁
        改名/删掉"退化成"今天不清理"，而不是巡检线程当场炸掉。真要退回
        `from data.sqlite_store import get_db` 摸 _lock/_conn 才是错的。
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_prune_day == today:
            return
        self._last_prune_day = today  # 先记账再干活：失败也不每分钟重试，明天再来
        prune = getattr(kv, "prune_candidates", None)
        if not callable(prune):
            return  # 门面被改名/删了：今天不清理，也不许绕过门面直摸 data 层
        try:
            removed = int(prune(14) or 0)
            print(f"[prune] memory_candidates 过期清理（14 天未晋升）：{removed} 条")
        except Exception:
            pass  # 清理失败不拦巡检，下一天再试

    def _write_idle_diary(self, session_id: str, last_time: str) -> None:
        """闲够久的会话补写日记（I2-1 覆盖式幂等 + I2-4 失败退避）。

        幂等判据不再是"这个日期写过没"，而是"日记是否已覆盖到当天最后一条
        对话"：_done 记 (session, 日期) -> 已覆盖的 last_time 快照当快路径；
        快照对不上（下午又聊了 / 进程重启）就问 DiaryWriter.covers()，
        没覆盖才让模型重写（同 id 覆盖同一篇）。
        失败退避（I2-4）：write_diary 空返回且确实没覆盖 = 写失败，
        1/2/4…分钟指数退避（封顶 1 小时），不再每 60 秒白烧一轮 LLM。
        """
        writer = services.get("diary_writer")
        if writer is None:
            return  # bootstrap 没装成（diary_writer 初始化失败会 mark_error），别硬调
        # 日记跟着最后聊天的日期走：闲置触发往往跨了零点，日期不能想当然用今天
        date = last_time[:10]
        key = (session_id, date)
        if self._done.get(key) == last_time:
            return  # 上次点名到现在没有新聊天记录，在册日记必然还覆盖着
        try:
            covered = writer.covers(session_id, date)
        except Exception:
            covered = False
        if covered:
            self._done[key] = last_time
            self._diary_fails.pop(key, None)
            return
        fails, next_try = self._diary_fails.get(key, (0, 0.0))
        if fails and time.time() < next_try:
            return  # 退避中：这轮不烧调用
        diary = writer.write_diary(session_id, date, skip_if_exists=True)
        if diary:
            self._done[key] = last_time
            self._diary_fails.pop(key, None)
        else:
            fails += 1
            self._diary_fails[key] = (
                fails,
                time.time() + min(60 * 2 ** (fails - 1), self._DIARY_BACKOFF_CAP_SEC),
            )
