"""能力层 - 记忆模块

记忆分四种，各管各的：
- SessionMemoryKeeper：短期记忆，最近几轮原文放内存，太长就让模型压成摘要
- MemoryRecaller：开聊前把档案、画像、关系、长期记忆一次捞齐
- ProfileUpdater / PortraitBuilder：档案和画像的唯二写入口
- ConversationDistiller：从聊天里抓硬事实，收工时写当日总结
- RelationshipTracker：亲密度好感度这些数值的增减
"""

import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime

from config.settings import load_app_config
from capability.quirks import MoodEngine
from data.schemas import MemoryItem
from shared.singletons import services
from shared.timeutils import safe_delta_seconds
from shared.types import MemoryBundle
from tools.misc import ClockTool, parse_llm_json

# J2：复用 tools/misc.Logger 底下的同一个 "aria" logger——兜底哲学不变（降级照旧、
# 聊天照常），但降级不许哑：凡吞异常处都留一条带异常信息的 warning。那边补上
# FileHandler 后这些告警会自动落盘，这里不用（也不该）自己建 handler。
logger = logging.getLogger("aria")


def _enqueue_retry(kind: str, payload: dict) -> None:
    """J12"静默遗忘需要补偿队列"的挂点：记忆写入类失败在这里落待重试标记。

    队列表归 data 层（迁移机制在别的批次落地），所以只探测门面有没有
    enqueue_retry(kind, payload_json)：有就入队等巡检消费；没有就退化成一条
    带内容的告警日志——宁可吵，不可哑（本项目已发生三次"吞异常=功能静默死亡"）。
    """
    try:
        from data.sqlite_store import get_db

        enqueue = getattr(get_db(), "enqueue_retry", None)
        if callable(enqueue):
            enqueue(kind, json.dumps(payload, ensure_ascii=False)[:2000])
            return
    except Exception as exc:
        logger.warning(f"[memory] 补偿队列写入失败（kind={kind}）: {exc}")
        return
    logger.warning(f"[memory] 补偿队列未落地，待重试留痕于此（kind={kind}）: "
                   f"{json.dumps(payload, ensure_ascii=False)[:200]}")


def remember_note(session_id: str, content: str, kind: str = "event",
                  importance: int = 3, feeling: str = "", appraisal: str = "",
                  valence: float = 0.0, arousal: float = 0.3) -> None:
    """往长期记忆写一条。全项目**只有这里**负责构造 MemoryItem + 落向量库 + 失败补偿。

    抽成模块级公开函数而不是留在 ConversationDistiller 里：G5 要在
    capability/self_identity.py 里记"她改过口"，那条不该复制一份 MemoryItem 构造
    和 J12 的补偿逻辑（复制一份 = 以后改一处忘一处）。
    ⚠ 同层调用方注意环：memory → quirks → char_life → self_identity，所以
    self_identity 里必须**函数内延迟 import** 本函数（memory.py 反向调
    self_identity.display_name 用的也是同一招）。
    """
    item = MemoryItem(
        memory_id=uuid.uuid4().hex,
        session_id=session_id,
        kind=kind,
        content=content,
        importance=importance,
        timestamp=ClockTool().now(),
        feeling=feeling,
        appraisal=appraisal,
        valence=valence,
        arousal=arousal,
        peak_moment=ClockTool().now() if feeling else "",
    )
    try:
        ok = services.get("vector_store").upsert_memory(item)
    except Exception as exc:
        # J12：向量库挂了聊天照常，但这正是"静默遗忘"——调用方那边状态可能
        # 已经改了、长期记忆却没落，两边不一致且以前无任何痕迹。现在落一条
        # 待重试标记进补偿队列（队列没落地时至少日志留痕），不许无声蒸发。
        logger.warning(f"[memory] 长期记忆写入向量库失败（kind={kind}）: {exc}")
        _enqueue_retry("vector_memory", dict(getattr(item, "__dict__", {}) or {}))
        return
    # R04：只有明确 True 才算成功。upsert_memory 的 False（集合未装配、内部
    # 写失败）以前被无声丢弃——异常走了补偿挂点、False 却蒸发，正是
    # "False 返回未消费"的缺口。现在 False 与异常走同一个可观测失败入口：
    # 一次失败最多登记一次（upsert_memory 内部只告警不登记，登记权在本层）。
    # 现阶段只接现有告警/补偿挂点，持久队列与重试消费是 R19 的活。
    if ok is not True:
        logger.warning(f"[memory] 长期记忆写入向量库未成功（kind={kind}，"
                       f"返回 {ok!r}），落待重试标记")
        _enqueue_retry("vector_memory", dict(getattr(item, "__dict__", {}) or {}))


class SessionMemoryKeeper:
    """短期记忆管家：一个会话一份上下文，原文放内存里。

    线程安全：append/get_context/restore/reset 会被不同线程（主图节点跑在
    asyncio/线程池里）并发调用，对三张字典的读写统一在同一把锁下完成；
    LLM 摘要这种慢调用一律放到锁外做，避免把会话串行化。
    另带会话上限 + LRU 淘汰，避免多会话长跑时内存只增不减。
    """

    def __init__(self):
        cfg = load_app_config()["memory"]
        self._max_turns = cfg["max_turns_short_term"]
        self._max_sessions = int(cfg.get("max_sessions", 8) or 8)
        self._contexts = {}   # session_id -> [{"role","content"}, ...]
        self._summaries = {}  # session_id -> 被挤出去的旧对话浓缩成的摘要
        self._touched = {}    # session_id -> 最后被碰的单调时刻，LRU 淘汰用
        # J12：被挤出上下文、等着摘要的旧消息。以前 _summarize 内联在 append 里，
        # 溢出那一轮用户白等一次 LLM 往返；失败还把最老一半对话直接丢掉（静默遗忘）。
        # 现在 append 只登记待办，慢调用由调用方在合适时机跑 run_pending_summary。
        self._pending = {}    # session_id -> [{"role","content"}, ...]
        self._pending_cap = max(200, int(self._max_turns) * 10)  # 持续失败时的内存保险丝
        self._lock = threading.Lock()

    def _evict_locked(self) -> None:
        """会话数超上限时淘汰最久未碰的那个，四张字典一起清。要求已持锁。"""
        while len(self._contexts) > self._max_sessions:
            victim = min(self._contexts, key=lambda k: self._touched.get(k, 0.0))
            self._contexts.pop(victim, None)
            self._summaries.pop(victim, None)
            self._touched.pop(victim, None)
            self._pending.pop(victim, None)

    def append(self, session_id: str, role: str, text: str) -> None:
        if not text:
            return
        old_part = None
        with self._lock:
            context = self._contexts.setdefault(session_id, [])
            context.append({"role": role, "content": text})
            self._touched[session_id] = time.monotonic()
            # 超长就把最老的一半摘出来；锁内只做截断，摘要慢调用不在这条路上（J12）
            if len(context) > self._max_turns:
                cut = len(context) // 2
                old_part = context[:cut]
                self._contexts[session_id] = context[cut:]
            self._evict_locked()
        if old_part:
            self._queue_pending(session_id, old_part)

    def append_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """成对追加一轮问答（F12）：一次加锁写进 user + assistant 两条。

        以前 user/assistant 分两次加锁，迟到的取消落在两次之间会留下
        "只有问没有答"的半截记忆——违反"这轮当没发生过"的不变式。
        两条文本都为空就什么都不写。
        """
        if not user_text and not assistant_text:
            return
        old_part = None
        with self._lock:
            context = self._contexts.setdefault(session_id, [])
            if user_text:
                context.append({"role": "user", "content": user_text})
            if assistant_text:
                context.append({"role": "assistant", "content": assistant_text})
            self._touched[session_id] = time.monotonic()
            if len(context) > self._max_turns:
                cut = len(context) // 2
                old_part = context[:cut]
                self._contexts[session_id] = context[cut:]
            self._evict_locked()
        if old_part:
            self._queue_pending(session_id, old_part)

    def _queue_pending(self, session_id: str, old_part: list) -> None:
        """把被挤出的旧对话登记成"需要摘要"的待办（J12），不在写回路径上烧 LLM。

        待办积压超上限时丢最老的并大声告警：LLM 长期挂着时内存不能只增不减，
        丢掉的份额靠补偿队列的待重试标记留痕，不做无声蒸发。
        """
        with self._lock:
            queued = self._pending.setdefault(session_id, [])
            queued.extend(old_part)
            if len(queued) > self._pending_cap:
                dropped = queued[:-self._pending_cap]
                del queued[:-self._pending_cap]
                logger.warning(f"[memory] 会话 {session_id} 的待摘要积压超上限，"
                               f"丢弃最老 {len(dropped)} 条（摘要长期失败？）")

    def has_pending_summary(self, session_id: str) -> bool:
        """J12 接口：这个会话有没有等着摘要的旧对话（调用方决定何时消化）。"""
        with self._lock:
            return bool(self._pending.get(session_id))

    def run_pending_summary(self, session_id: str) -> bool:
        """J12 接口：消化"需要摘要"的待办。返回是否全部成功。

        由调用方在合适的时机跑（如写回完成后的后台任务）——摘要的慢 LLM 调用
        从此不再阻塞当轮回复。失败时待办**塞回队列**等下次重试（不丢 old_part），
        同时落补偿队列标记；持续失败才会触到 _queue_pending 的上限保险丝。
        """
        with self._lock:
            msgs = self._pending.pop(session_id, None)
        if not msgs:
            return True
        if self._summarize(session_id, msgs):
            return True
        with self._lock:
            # 塞回时排在期间新积压的前面，保持时间顺序
            rest = self._pending.get(session_id, [])
            self._pending[session_id] = (msgs + rest)[:self._pending_cap]
        return False

    def _summarize(self, session_id: str, old_part: list) -> bool:
        """把被挤出去的旧对话压成摘要，成功 True / 失败 False。慢调用在锁外做。

        持锁做模型调用会把所有会话串行化，比不加锁还糟，所以这里先放开锁。
        摘要是"追加式"的：每次淘汰往旧摘要后面接一段，长会话里会无限膨胀——
        超过 400 字就把旧摘要+新段落一起再压缩一次，摘要始终是"一段话"而不是
        一本流水账（token 成本和上下文质量都受益）。
        """
        old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_part)
        with self._lock:
            base = self._summaries.get(session_id, "")
        needs_recompress = len(base) > 400
        try:
            if needs_recompress:
                got = services.get("llm").chat(
                    [
                        {"role": "system", "content": "把新旧两段摘要融合成一段话，保留关键信息（名字、约定、聊过的事），150 字以内。"},
                        {"role": "user", "content": f"【旧摘要】\n{base}\n\n【新段落】\n{old_text}"},
                    ],
                    temperature=0.3,
                    max_tokens=250,
                )
            else:
                got = services.get("llm").chat(
                    [
                        {"role": "system", "content": "把这段对话浓缩成一段话，保留关键信息（名字、约定、聊过的事），100 字以内。"},
                        {"role": "user", "content": old_text},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )
        except Exception as exc:
            # J12：摘要失败不再"丢掉最老一半、当没发生过"——待办由调用方塞回重试，
            # 这里补一条持久化的待重试标记（静默遗忘需要补偿队列）
            logger.warning(f"[memory] 会话 {session_id} 摘要失败（{len(old_part)} 条待重试）: {exc}")
            _enqueue_retry("session_summarize", {
                "session_id": session_id, "n_msgs": len(old_part),
                "head": old_text[:200],
            })
            return False
        with self._lock:
            if needs_recompress:
                # 重压缩是把"旧摘要+新段落"融成一段：直接替换，不能再追加
                # （追加会让旧摘要的内容出现两份）
                self._summaries[session_id] = got.strip()
            else:
                # 写回时以锁内最新值为准（期间别的线程可能已写入新摘要），避免丢更新
                summary = self._summaries.get(session_id, "")
                self._summaries[session_id] = f"{summary}\n{got}".strip() if summary else got
        return True

    def get_context(self, session_id: str) -> list:
        """喂给模型的上下文：摘要放最前面（如果有），后面跟最近几轮原文。

        返回副本（连内部字典元素也各复制一份），调用方在外面切片/遍历/改写都不影响内部。
        """
        with self._lock:
            self._touched[session_id] = time.monotonic()
            summary = self._summaries.get(session_id, "")
            session_msgs = [dict(m) for m in self._contexts.get(session_id, [])]
        messages = []
        if summary:
            messages.append({"role": "system", "content": f"【之前聊过的内容，摘要】\n{summary}"})
        messages.extend(session_msgs)
        return messages

    def restore(self, session_id: str) -> None:
        """进程重启后把上次的聊天记录捞回来，不然每次重启都失忆。

        读库是磁盘动作，放到锁外；只在读写字典时持锁（与 append 同一原则）。
        """
        with self._lock:
            if self._contexts.get(session_id):
                return  # 内存里已经有了就不用折腾
        try:
            records = services.get("kv_store").read("session", session_id) or []
        except Exception as exc:
            logger.warning(f"[memory] 会话 {session_id} 短期记忆恢复失败（按空上下文继续）: {exc}")
            return
        restored = [
            {"role": r.get("role", "user"), "content": r.get("text", "")}
            for r in records[-self._max_turns:]
            if r.get("text")
        ]
        with self._lock:
            if self._contexts.get(session_id):
                return  # 期间别的线程已经恢复了，别覆盖
            self._contexts[session_id] = restored
            self._touched[session_id] = time.monotonic()
            self._evict_locked()

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._contexts.pop(session_id, None)
            self._summaries.pop(session_id, None)
            self._touched.pop(session_id, None)
            self._pending.pop(session_id, None)  # J12：待办跟着会话一起清，别摘要到上一段人生


class MemoryRecaller:
    """把一个会话能想起来的东西全捞出来，打包成 MemoryBundle 一次带走。"""

    def recall(self, query: str, session_id: str) -> MemoryBundle:
        bundle = MemoryBundle()
        try:
            kv = services.get("kv_store")
            bundle.profile = kv.read("profile", session_id) or {}
            bundle.portrait = kv.read("portrait", session_id) or {}
            bundle.relationship = kv.read("relationship", session_id) or {}
        except Exception as exc:
            logger.warning(f"[memory] 档案/画像/关系读取失败（空着继续，不拦聊天）: {exc}")

        try:
            top_k = load_app_config()["memory"]["recall_top_k"]
            bundle.distilled = services.get("vector_store").search_memory(query or session_id, top_k=top_k)
        except Exception as exc:
            logger.warning(f"[memory] 长期记忆召回失败（按无记忆继续）: {exc}")
            bundle.distilled = []

        # 日记单独搜一遍：日记是角色的回忆，跟硬事实记忆不是一个味，都捞点才聊得起来
        try:
            bundle.diaries = services.get("vector_store").search_diary(query or session_id, top_k=3)
        except Exception as exc:
            logger.warning(f"[memory] 日记召回失败（按无日记继续）: {exc}")
            bundle.diaries = []
        return bundle


class ProfileUpdater:
    """用户档案的增改，只认这些字段，别的写了也白写。"""

    _FIELDS = (
        "nickname", "gender", "age", "city", "occupation", "birthday",
        "health_notes", "likes", "dislikes", "notable_facts",
    )

    def set_field(self, session_id: str, field_name: str, value) -> bool:
        if field_name not in self._FIELDS:
            return False
        try:
            kv = services.get("kv_store")

            def _merge(profile: dict) -> dict:
                if isinstance(value, list):
                    # 列表类字段做去重合并，别越攒越重复
                    profile[field_name] = list(dict.fromkeys((profile.get(field_name) or []) + value))
                else:
                    profile[field_name] = value
                return profile

            # 走原子闭包（F3）：REST 端点与聊天写回线程并发改档案时不再互相覆盖
            kv.update("profile", session_id, _merge)
            return True
        except Exception as exc:
            logger.warning(f"[memory] 档案字段 {field_name} 写入失败: {exc}")
            return False

    def get_field(self, session_id: str, field_name: str):
        try:
            return (services.get("kv_store").read("profile", session_id) or {}).get(field_name)
        except Exception as exc:
            logger.warning(f"[memory] 档案字段 {field_name} 读取失败（按没有处理）: {exc}")
            return None


class MemoryGatekeeper:
    """记忆防污染的晋升与仲裁管线（S4）：先怀疑、再验证、慢接受。

    LLM 抽取的"事实"绝不直接入档：
    - 第 1 道：置信度门槛（复用 memory.profile_confidence_threshold）；
    - 第 2 道：晋升门槛——同内容出现 >=2 次才晋升（单次可能是玩笑/口误/幻觉）；
    - 第 3 道：矛盾仲裁——新旧值冲突时不自动覆盖，让 LLM 判一次"真变更还是
      口误/玩笑"，仲裁失败保留候选下次重试（绝不把幻觉焊死在档案里）；
    - 配套：入池超过 14 天没晋升的候选过期放弃（巡检线程周期清理）。
    正则抽取的硬事实（确定性高）不在此列，仍走 apply_fact 直通。
    列表型字段（likes/dislikes/notable_facts/health_notes）同样入闸（E3）：
    逐项走候选池，晋升时包成 [项] 交给 apply_fact 做去重合并——单次 LLM
    输出直通列表字段曾是幻觉焊死最宽的一扇门。
    """

    _PROMOTION_HITS = 2

    # 列表合并语义的字段：与 ProfileUpdater.set_field 的 isinstance(value, list)
    # 分支同一批。晋升时必须以 [content] 传值，否则字符串会整包替换掉已有列表。
    _LIST_MERGE_FIELDS = frozenset({"likes", "dislikes", "notable_facts", "health_notes"})

    def process(self, session_id: str, field: str, content: str,
                quote: str = "", confidence: float = 0.0) -> None:
        from data.sqlite_store import get_db

        threshold = float(load_app_config()["memory"].get("profile_confidence_threshold", 0.6))
        try:
            confidence = float(confidence or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        if field not in ProfileUpdater._FIELDS or confidence < threshold:
            return
        content = str(content or "").strip()[:300]
        if not content:
            return
        try:
            hits, cand_id = get_db().candidate_hit(session_id, field, content, quote)
        except Exception as exc:
            logger.warning(f"[memory] 候选池写入失败，本轮放弃（field={field}）: {exc}")
            return  # 候选池挂了，本轮放弃（和记忆写入同一兜底哲学）
        if hits < self._PROMOTION_HITS:
            return  # 证据不足，继续攒

        # 达到晋升门槛：仲裁新旧值
        try:
            old = (services.get("kv_store").read("profile", session_id) or {}).get(field)
        except Exception as exc:
            logger.warning(f"[memory] 仲裁前读旧值失败（按无旧值处理，field={field}）: {exc}")
            old = None
        is_list_field = field in self._LIST_MERGE_FIELDS
        if is_list_field and isinstance(old, list) and content in [str(x) for x in old]:
            # 列表字段已有这一项：无变更可仲裁也无东西可合并，直接标晋升，
            # 别让它每攒够 2 次就白烧一遍仲裁 LLM
            self._mark_promoted(cand_id)
            return
        # 仲裁员看到的旧值统一成人话：列表摊平成顿号串，否则它拿 "['咖啡']" 比 "咖啡"
        old_for_arb = "、".join(str(x) for x in old) if isinstance(old, list) else old
        verdict = self._arbitrate(field, old_for_arb, content, quote, session_id=session_id)
        if verdict == "keep":
            # E2：注释曾经声称"候选标晋升"而代码什么都没做——被否决的值每次重现都
            # 重跑一次仲裁 LLM，还可能哪天翻成 update 把口误焊进档案。现在真的标。
            self._mark_promoted(cand_id)
            return  # 仲裁认定口误/玩笑：保留旧值，候选已标晋升
        if verdict is None:
            return  # 仲裁失败（LLM 挂/输出坏）：不标晋升，下次出现重试
        distiller = ConversationDistiller()  # E5：只有 update 路径才用得上，别在白跑的路径上构造
        value = [content] if is_list_field else content
        if distiller.apply_fact(session_id, field, value):
            self._mark_promoted(cand_id)

    @staticmethod
    def _mark_promoted(cand_id: int) -> None:
        """标晋升。失败只告警：候选会再攒一轮 hits 重仲裁一次，是浪费不是错误。"""
        try:
            from data.sqlite_store import get_db

            get_db().mark_candidate_promoted(cand_id)
        except Exception as exc:
            logger.warning(f"[memory] 候选 {cand_id} 标晋升失败（可能重复仲裁一次）: {exc}")

    @staticmethod
    def _persona_line(session_id: str) -> str:
        """仲裁员的人设行（E5）。CLAUDE.md 设计原则：任何 LLM 评判者必须看到人设，
        否则会系统性惩罚人设规定的言行。取不到就空串（仲裁照跑，不拦）。"""
        try:
            from capability import self_identity

            name = self_identity.display_name(session_id)
            line = f"你是「{name}」，一个真实的人（沉浸式人设，绝不认自己是 AI）。"
            try:
                rel = services.get("kv_store").read("relationship", session_id) or {}
                stage = str(rel.get("stage") or "") if isinstance(rel, dict) else ""
            except Exception as exc:
                logger.warning(f"[memory] 仲裁人设行读关系阶段失败（略过 stage）: {exc}")
                stage = ""
            if stage:
                line += f"你和这位用户的关系阶段是「{stage}」。"
            return line
        except Exception as exc:
            logger.warning(f"[memory] 仲裁人设行获取失败（用裸提示词继续）: {exc}")
            return ""

    def _arbitrate(self, field: str, old, new: str, quote: str,
                   session_id: str = "") -> "str | None":
        """矛盾仲裁：返回 "update" / "keep" / None（失败）。无冲突直接 update。

        E5：判"真变更还是口误/玩笑"必须给他自己的原话（quote）——以前调用链把
        quote 丢了，仲裁员对着"他的原话：（空）"硬判，等于掷骰子。
        """
        if not old or str(old).strip() == new.strip():
            return "update"
        try:
            raw = services.get("llm").chat(
                [
                    {"role": "system", "content": (
                        f"{self._persona_line(session_id)}"
                        "你是档案仲裁员。档案里已有一个旧值，对话里出现了新信息。"
                        "判断新信息是【真变更】（他生活变了，应更新）还是【临时状态/口误/玩笑】"
                        "（不应更新）。判的是**用户**的档案，只依据他的原话和上下文，"
                        "你自己的设定不参与判断。只输出 JSON：{\"action\":\"update\"或\"keep\","
                        "\"reason\":\"一句话理由\"}。"
                    )},
                    {"role": "user", "content": f"字段：{field}\n旧值：{old}\n新值：{new}\n他的原话：{quote}"},
                ],
                temperature=0.1,
                max_tokens=120,
            )
            data = parse_llm_json(raw)
            action = str(data.get("action") or "")
            if action not in ("update", "keep"):
                return None
            return action
        except Exception as exc:
            logger.warning(f"[memory] 仲裁失败（field={field}，保留候选下次重试）: {exc}")
            return None


def _dialogue_fp(t) -> str:
    """对话消息的"已计数"指纹（E6 去重用）：role + 内容前 80 字足以区分两条消息。

    不是密码学场景，不需要哈希；存原文片段反而方便人工排查画像里数过什么。
    """
    if isinstance(t, dict):
        return f"{t.get('role', '')}|{str(t.get('content', ''))[:80]}"
    return str(t)[:80]


class PortraitBuilder:
    """软画像：从最近聊天里提炼标签、兴趣、核心需求，顺带 LLM 抽硬事实和缺点。

    每 5 轮才跑一次（调用方控制），模型调用开销平摊在这几轮里；
    抽取结果带置信度，低于门槛的不入库——这是记忆防污染的第一道闸。
    失败就静默，别打扰聊天。
    E6：调用方给的窗口可能重叠（每 5 轮取最近 N 条），每条消息带"已计数"
    指纹存在画像记录里，只有**新消息**才喂模型——同一句话被数两次不算
    "两次独立证据"，否则闸门的 "≥2 次晋升" 会被重叠窗口架空。
    """

    def refresh(self, session_id: str, recent_dialogues: list) -> None:
        if not recent_dialogues:
            return
        try:
            kv = services.get("kv_store")
            # E6：先滤掉已计数过的消息（读画像是外部调用，坏了按"没数过"降级，
            # 最坏退回旧行为=窗口重叠，不拦刷新本身）
            try:
                counted = set((kv.read("portrait", session_id) or {}).get("counted_fp") or [])
            except Exception as exc:
                logger.warning(f"[memory] 画像读已计数指纹失败（本次按全新窗口处理）: {exc}")
                counted = set()
            fresh = [t for t in recent_dialogues if _dialogue_fp(t) not in counted]
            if not fresh:
                return  # 窗口整个数过了：没有新证据，不烧 LLM 也不给幻觉重复计数
            # 上下文元素是 {"role","content"} 字典，翻成人话再喂模型。
            # C5：她的发言标"她"不标"AI"——每轮往判断模型脸上贴"她是 AI"，
            # 再要求它写出"她绝不认自己是 AI"的产出，是心事漂向 meta 的机械根因
            chats = "\n".join(
                (
                    f"{'用户' if t.get('role') == 'user' else '她'}：{t.get('content', '')}"
                    if isinstance(t, dict)
                    else str(t)
                )
                for t in fresh[-10:]
            )
            raw = services.get("llm").chat(
                [
                    {"role": "system", "content": (
                        "从聊天记录里总结用户画像，只输出 JSON：\n"
                        '{"portrait_tags": ["标签"], "core_needs": "一句话", "interests": ["兴趣"], '
                        '"hard_facts": [{"field": "字段名", "value": "值", "confidence": 0到1的小数, '
                        '"quote": "用户说出这件事的原话片段"}], '
                        '"flaws": ["用户的缺点、翻车、出糗事"]}\n'
                        "标签不超过 5 个；hard_facts 的 field 只能取 "
                        "nickname/gender/age/city/occupation/birthday/likes/dislikes/notable_facts 之一，"
                        "只收用户明确说出口的事实，拿不准就给低 confidence；"
                        "quote 必须是记录里真实出现过的原话（仲裁要用，编不出来就不给这条 fact）；"
                        "flaws 最多 3 条，看不出就给空列表。"
                    )},
                    {"role": "user", "content": chats},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            data = parse_llm_json(raw)
            mem_cfg = load_app_config()["memory"]
            limit = mem_cfg["portrait_tag_limit"]
            threshold = float(mem_cfg.get("profile_confidence_threshold", 0.6))

            gate = MemoryGatekeeper()  # E5：无状态小对象，用到才建
            distiller = None
            # 硬事实入库（S4/E3）：标量和列表**都**走"候选→晋升→仲裁"闸门，绝不直接入档。
            # 列表字段逐项入池（apply_fact 的合并语义在晋升后由闸门以 [项] 传值保留）——
            # 以前列表单次 LLM 输出直通入档+入向量库，是幻觉焊死最宽的一扇门
            for fact in data.get("hard_facts") or []:
                if not isinstance(fact, dict):
                    continue
                try:
                    confidence = float(fact.get("confidence", 0))
                except (TypeError, ValueError):
                    continue
                field = str(fact.get("field", "")).strip()
                value = fact.get("value")
                if not field or value in (None, ""):
                    continue
                quote = str(fact.get("quote") or "")  # E5：原话一路带到仲裁员面前
                if isinstance(value, list):
                    for item in value:
                        item = str(item or "").strip()
                        if item:
                            gate.process(session_id, field, item, quote=quote, confidence=confidence)
                else:
                    gate.process(session_id, field, str(value), quote=quote, confidence=confidence)

            # 画像整包读改写走原子闭包（F3）：锁内只做纯合并计算，
            # 期间别的写方（语音写回/巡检）落的字段不会被这次覆盖掉
            new_flaws: list = []  # 闭包里算出的"这次真正新出现的缺点"，出锁后喂向量库
            fresh_fps = [_dialogue_fp(t) for t in fresh]  # E6：本次计数的指纹，闭包纯计算用

            def _apply(old: dict) -> dict:
                old = old or {}
                pre_flaws = set(old.get("user_flaws") or [])
                for flaw in data.get("flaws") or []:
                    flaw = str(flaw).strip()
                    if flaw and flaw not in pre_flaws:
                        new_flaws.append(flaw)
                kept_tags, tag_updated = self._decay_tags(old, data.get("portrait_tags"), limit)
                # 指纹滚动保留最近 60 条（约 30 轮）：只需盖住相邻窗口的重叠量，
                # 全量保留会让画像记录只增不减
                kept_fp = list(dict.fromkeys(list(old.get("counted_fp") or []) + fresh_fps))[-60:]
                return {
                    "portrait_tags": kept_tags,
                    "core_needs": data.get("core_needs") or old.get("core_needs", ""),
                    "interests": self._merge(old.get("interests"), data.get("interests"), limit),
                    "relationship_assessment": old.get("relationship_assessment", ""),
                    "user_flaws": self._merge_flaws(old.get("user_flaws"), data.get("flaws")),
                    "tag_updated": tag_updated,
                    "counted_fp": kept_fp,
                    "updated_at": ClockTool().now(),
                }

            kv.update("portrait", session_id, _apply)

            # 缺点/翻车记录：画像存一份（上面闭包里），向量库存一份（语义召回）
            for flaw in new_flaws[:3]:
                if distiller is None:
                    distiller = ConversationDistiller()  # E5：没有新缺点就一个都不用建
                distiller.remember_flaw(session_id, flaw)
        except Exception as exc:
            logger.warning(f"[memory] 画像刷新失败（下轮再试，指纹未计数会自动重试）: {exc}")

    @staticmethod
    def _merge(old_list, new_list, limit: int) -> list:
        merged = list(dict.fromkeys(list(old_list or []) + list(new_list or [])))
        return merged[:limit]

    @staticmethod
    def _merge_flaws(old_list, new_list, limit: int = 10) -> list:
        """缺点记录新的在前，旧的往后挤，超上限淘汰——人也记不住无限多的破事。"""
        merged = list(dict.fromkeys(list(new_list or []) + list(old_list or [])))
        return [str(f).strip() for f in merged if str(f).strip()][:limit]

    @staticmethod
    def _decay_tags(old: dict, new_tags, limit: int) -> tuple:
        """合并新旧标签并做过期清理，返回 (保留的标签, 每个标签最后出现时间)。

        tag_updated 记每个标签最后一次被提到的时刻，超过 persona_decay_days
        没再出现就清掉——过时的印象不该一直贴在人身上。
        """
        now = ClockTool().now()
        tag_updated = dict(old.get("tag_updated") or {})
        for t in new_tags or []:
            tag_updated[str(t)] = now
        merged = list(dict.fromkeys(list(old.get("portrait_tags") or []) + list(new_tags or [])))
        decay_days = float(load_app_config()["memory"].get("persona_decay_days", 14) or 14)
        kept = []
        for tag in merged:
            ts = tag_updated.get(tag, "")
            if ts:
                # R09b：统一解析（Z / naive 本地语义 / 坏值降级收口在 timeutils）
                age_days = safe_delta_seconds(now, ts)
                if age_days is not None and age_days / 86400 > decay_days:
                    tag_updated.pop(tag, None)
                    continue  # 太久没再出现的印象，过期清掉
                # 时间戳坏了就当没记过，标签保留（旧行为不变）
            kept.append(tag)
        return kept[:limit], tag_updated


class ConversationDistiller:
    """长期记忆沉淀：正则抓硬事实（比让模型抽又稳又省），重要的事写进向量库。"""

    # 硬事实抓取规则：字段名 -> 正则
    # 正则通道是**直通入档**的（distill_turn -> apply_fact，无置信度无仲裁），
    # 误报即永久档案——所以这里的取舍一律"宁漏勿误"：漏掉的还有 LLM 通道兜着，
    # 焊错的没人救。E4 修过三个实测误报：
    # - nickname："我叫了个车" 曾收进 "了个车"——叫 后排除动词补语（做/了/着/过/个），
    #   捕获限 4 字（中文昵称几乎不超 4 字，贪到 10 字全是把谓语宾语当名字）；
    # - age："我奶奶今年80岁" 曾收进用户年龄 80——我 与数字之间出现亲属称谓就不算
    #   用户自己的年龄（亲属词表宁多勿少，同"宁漏勿误"）。
    _FACT_PATTERNS = [
        ("nickname", re.compile(
            r"我(?:叫(?!做|了|着|过|个)|的名字是|叫做)([\u4e00-\u9fa5A-Za-z0-9]{1,4})"
        )),
        ("city", re.compile(r"我(?:住在|住|生活)在?([\u4e00-\u9fa5]{2,8}?)(?:市|区|省)")),
        ("age", re.compile(
            r"我(?![\u4e00-\u9fa5，, ]{0,10}(?:奶奶|爷爷|姥姥|姥爷|外婆|外公|儿子|女儿|孙子|孙女|"
            r"老公|老婆|丈夫|妻子|爸|妈|叔|伯|舅|姨|姑|姐|哥|弟|妹|侄|甥))"
            r"[\u4e00-\u9fa5，, ]{0,10}?(\d{1,2})岁"
        )),
        ("birthday", re.compile(r"(?:我的)?生日(?:是)?(\d{1,2}月\d{1,2}[号日])")),
        ("occupation", re.compile(
            r"我(?:是|在做|从事)(?:一名)?(学生|老师|程序员|医生|护士|工程师|设计师|公务员|销售|司机|工人|厨师|会计|律师)"
        )),
    ]
    _FIELD_LABELS = {
        "nickname": "叫", "city": "住在", "age": "今年",
        "birthday": "生日是", "occupation": "职业是",
    }

    # 匹配点跟前或匹配段内部出现这些字就算否定（"我不叫X""我不住在Y"），别把反话当事实收
    _NEGATION_WORDS = ("不", "没", "别", "未")

    @classmethod
    def _is_negated(cls, text: str, start: int, end: int = -1) -> bool:
        """否定守卫：匹配起点前两个字内 **或匹配段内部** 出现否定词就当没说。

        E4：以前只看起点前 2 字，而所有 pattern 都以"我+动词"锚定——"我不叫X"
        的 叫 分支根本匹配不上，"我不住在上海"走的是裸 住 分支、否定词藏在
        **span 内部**（我|不住在|上海）。起点前的窗口对这些 pattern 近乎死代码，
        现在 span 内也查。end<0 表示调用方没给 span，退回只查起点前（兼容旧调用）。
        """
        window = text[max(0, start - 2):start]
        span = text[start:end] if 0 <= start <= end else ""
        return any(n in window or n in span for n in cls._NEGATION_WORDS)

    def apply_fact(self, session_id: str, field: str, value) -> bool:
        """硬事实入库的唯一通道：值没变一条记忆都不加，变了连改口一起记。

        正则和 LLM 抽取都汇到这，防污染的两个关键动作（去重、改口更新）只写这一遍。
        """
        if field not in ProfileUpdater._FIELDS:
            return False
        try:
            old = (services.get("kv_store").read("profile", session_id) or {}).get(field)
        except Exception as exc:
            logger.warning(f"[memory] apply_fact 读旧值失败（按无旧值处理，field={field}）: {exc}")
            old = None
        if old == value:
            return False  # 老话重提，档案已对，别往向量库堆重复记忆
        if not ProfileUpdater().set_field(session_id, field, value):
            return False
        label = self._FIELD_LABELS.get(field, field)
        if old:
            self._remember(
                session_id,
                f"用户把{label}从「{old}」改成了「{value}」，以新的为准",
                "hard_fact",
                5,
            )
        else:
            self._remember(session_id, f"用户{label}{value}", "hard_fact", 5)
        return True

    def distill_turn(self, session_id: str, user_text: str) -> None:
        """每轮都跑一遍：正则抓硬事实，否定句跳过，剩下交给 apply_fact 把关。"""
        if not user_text:
            return
        for field, pattern in self._FACT_PATTERNS:
            m = pattern.search(user_text)
            # E4：span 一起交给否定守卫——否定词可能藏在匹配段内部（"我不住在上海"）
            if not m or self._is_negated(user_text, m.start(), m.end()):
                continue
            self.apply_fact(session_id, field, m.group(1))

    def distill_day(self, session_id: str, texts: list) -> None:
        """收工时把今天聊的归成一段总结存起来，明天聊起来有话说。

        S1：同一次调用顺带产出**她自己对这一天的感受**（生成时想一次就冻死）。
        JSON 解析失败退回纯文本总结（感受留空），行为不比改造前差。
        """
        if not texts:
            return
        try:
            joined = "\n".join(str(t) for t in texts[-40:])
            raw = services.get("llm").chat(
                [
                    {"role": "system", "content": (
                        "把这段聊天总结成一段话，重点记用户提到的事、约定和情绪，120 字以内。\n"
                        "只输出 JSON：{\"summary\":\"总结\",\"feeling\":\"她自己对今天的感受，"
                        "第一人称、带身体或感官细节的短语（如'嘴上嫌他烦，其实挺高兴'），30字内\","
                        "\"appraisal\":\"她为什么这么感觉，一句话\","
                        "\"valence\":-1到1的小数,\"arousal\":0到1的小数}"
                        "。解析失败会丢掉总结，所以 JSON 必须合法。"
                    )},
                    {"role": "user", "content": joined},
                ],
                temperature=0.3,
                max_tokens=320,
            )
            try:
                data = parse_llm_json(raw)
                # J2：这段路径上没有外部调用（纯解析），不许 except Exception 兜底——
                # 解析器只会抛 ValueError 家族；输出不是对象是模型答非所问，显式判掉
                if not isinstance(data, dict):
                    raise ValueError(f"日总结输出不是 JSON 对象：{type(data).__name__}")
                summary = str(data.get("summary") or "").strip()
                feeling = str(data.get("feeling") or "").strip()[:60]
                appraisal = str(data.get("appraisal") or "").strip()[:80]
                try:
                    valence = max(-1.0, min(1.0, float(data.get("valence") or 0)))
                    arousal = max(0.0, min(1.0, float(data.get("arousal") or 0.3)))
                except (TypeError, ValueError):
                    valence, arousal = 0.0, 0.3
            except ValueError:
                summary, feeling, appraisal, valence, arousal = (raw or "").strip(), "", "", 0.0, 0.3
            if summary:
                self._remember(session_id, summary, "daily_summary", 3,
                               feeling=feeling, appraisal=appraisal,
                               valence=valence, arousal=arousal)
        except Exception as exc:
            logger.warning(f"[memory] 日总结失败（今天不沉淀，聊天照常）: {exc}")

    def remember_flaw(self, session_id: str, flaw: str) -> None:
        """把用户的缺点/翻车记进长期记忆，语义召回时能捞出来当调侃素材。

        S1 启发式：缺点/翻车是负向事件（效价负、唤醒中等），不用烧 LLM。
        """
        if flaw:
            self._remember(session_id, f"他的缺点/翻车：{flaw}", "user_flaw", 4,
                           valence=-0.6, arousal=0.5)

    def _remember(self, session_id: str, content: str, kind: str, importance: int,
                  feeling: str = "", appraisal: str = "",
                  valence: float = 0.0, arousal: float = 0.3) -> None:
        """沉淀一条长期记忆。真身在模块级 remember_note（G5 也要用，见那里的注释）。"""
        remember_note(session_id, content, kind=kind, importance=importance,
                      feeling=feeling, appraisal=appraisal,
                      valence=valence, arousal=arousal)


def _arc_from_entries(entries: list, limit: int = 3, now=None) -> str:
    """账本条目 -> 关系弧线文本（S10 的纯计算部分，单测直接喂假账本）。

    空账本 / 全是空 reason → 空串（不注入，绝不硬凑）。
    趋势按账本 delta 求和：≥ +2 说"明显在升温"，≤ -2 说"有点僵"。
    now 参数留给测试注入固定时刻，生产代码不用传。

    输出形态分两种，取决于账本**是否跨天**：
    - 单日（只有一个 label，全是"今天"）：**不加日期前缀**，直接把去重后的
      reason 用顿号连起来。同一天聊三轮本就产出三条同样的 reason，逐条署名
      "今天…；今天…；今天…" 是机械重复，直接把机器味暴露给模型；单日也没有
      "哪一天"的信息量，日期标签在这里纯属噪音。
    - 跨天（≥2 个不同 label）：才加日期前缀，每个 label 只出现一次。

    同一 label 内的 reason 去重（dict.fromkeys 保序）：她同一天重复做了同一件事，
    叙事里说一次就够，说三遍是在露怯。
    """
    if not entries:
        return ""
    now = now or datetime.now()
    # label -> 该 label 下按原顺序出现过的 reason（去重）
    grouped: dict = {}
    order = []  # label 首次出现的顺序，保证输出稳定
    for entry in entries[-limit:]:
        label = ""
        # R09b：统一解析——days 语义与 timedelta.days 一致（负值=未来时间戳，
        # 交给下面的 fallback 标签，与旧行为相同）
        diff = safe_delta_seconds(now, entry.get("time", ""))
        if diff is not None:
            diff = int(diff // 86400)
            label = {0: "今天", 1: "昨天", 2: "前天"}.get(diff, f"{diff}天前")
        reason = (entry.get("reason") or "").strip()
        if not reason:
            continue
        if label not in grouped:
            grouped[label] = []
            order.append(label)
        if reason not in grouped[label]:
            grouped[label].append(reason)  # 同 label 内去重，保留首次出现的位置
    if not grouped:
        return ""

    # 跨天才加日期前缀：只有一个 label（或全是"今天"）时日期没有区分度，省掉
    cross_day = len(order) >= 2
    segments = []
    for label in order:
        joined = "、".join(grouped[label])
        segments.append(f"{label}{joined}" if (cross_day and label) else joined)
    arc = "；".join(segments)

    try:
        trend = sum(float(entry.get("delta") or 0) for entry in entries)
    except (TypeError, ValueError):
        trend = 0
    # 单日措辞用"最近这几轮"：sum 的是"最近 N 笔"，说"这几天"会把同一天的多次
    # 互动夸大成跨天趋势，语义失真；跨天才用"这几天"
    span = "这几天" if cross_day else "最近这几轮"
    if trend >= 2:
        arc += f"；整体来看，{span}关系明显在升温"
    elif trend <= -2:
        arc += f"；整体来看，{span}关系有点僵"
    return arc + "。"


def recent_arc(session_id: str, limit: int = 3) -> str:
    """关系氛围线（S10）：近几次关系账本的趋势 → 一条零成本的关系弧线。

    快照式注入（心情数字/关系数值）没有"刚发生过什么"的过程感；这里用纯模板
    把账本拼成叙事——"昨天聊得开心，关系热乎了一点；整体来看，这几天在升温"。
    任何异常都不注入（增强项，聊天照常）。LLM 零调用。
    """
    try:
        entries = services.get("kv_store").recent_ledger(session_id, n=8)
        return _arc_from_entries(entries, limit=limit)
    except Exception as exc:
        logger.warning(f"[memory] 关系氛围线聚合失败（本轮不注入）: {exc}")
        return ""  # 氛围线是增强项，任何异常都不注入


# C6 反向标定词表：**只抓显式安抚/道歉**——她的话说出口之后，反过来校正状态
# （注入了"别扭"、话却是哄人的，错的是状态不是话）。语气级的收敛（句子变短、
# 用词变软）留给 H6 的 energy 观测，两者共用观测面，别在这里扩词表。
_SOOTHE_WORDS = ("抱抱", "不生气", "别生气", "我错了", "是我不好", "对不起", "抱歉", "消消气")

# 负面心情集：与 capability/quirks._NEGATIVE_MOODS 同一批标签（那边是模块私有，
# 这里持一份副本；改动必须两边同步——persona_engine 也有一份同样的副本）。
_NEG_MOODS = ("别扭", "低落", "慵懒", "心烦")


class RelationshipTracker:
    """关系数值：每次聊完按情绪微调，数值攒着决定关系阶段。

    整个"读整包 → 加增量 → 心情走格 → 写回"收进一次原子闭包（F3）：
    文字聊天、语音写回、巡检线程、REST 端点四个并发写方，谁也不能把谁的
    增量覆盖掉。心情（MoodEngine）也在这里一起算——以前 pipeline 先写一次
    relationship、再单独写一次 mood，两次写之间既可能被打断也可能互相覆盖。
    每次数值变动进账本（affection_history），留痕可审计。
    """

    # 不同情绪带来的数值变化，正负都写明白
    _EMOTION_DELTA = {
        "happy": {"intimacy": 2, "affection": 2, "trust": 1},
        "sad": {"intimacy": 1, "affection": 3, "trust": 1},
        "crisis": {"intimacy": 0, "affection": 4, "trust": 2},
        "angry": {"intimacy": -1, "affection": -2, "trust": -1},
        "tired": {"intimacy": 1, "affection": 1, "trust": 0},
    }
    _STAGES = ((70, "挚友"), (45, "亲近"), (20, "熟悉"), (0, "初识"))

    def __init__(self):
        self._mood_engine = MoodEngine()

    def update(self, session_id: str, emotion: str = "neutral",
               comfort_mode: bool = False, was_poor: bool = False,
               intensity: float = 0.5, extras=None,
               utterance_text: str = "") -> dict:
        """按情绪微调数值，一次原子闭包落好数值 + 心情 + 心事 + 收敛 + 阶段标记。

        was_poor（F8）：本轮是否被判敷衍/违规重写，收进闭包一起写。以前 pipeline
        在闭包返回后又整包 kv.write 一次 relationship，那正是 F3 要消灭的
        "读整包 -> 内存改 -> 整包覆盖写"——并发的 REST 增量会被这次陈旧覆盖吞掉。
        放进闭包后天然只生效一轮（下一轮写回会带着新值覆盖），且不再有第二次写。
        默认 False 保证旧调用方（如语音写回）不受影响。

        intensity：本轮情绪强度（N2 惯性时长随它走）；extras（S2/N2/C7）：感知的
        附加产出（心事实体信号 same_concern/resolved + 语气反馈），没有传 None。

        utterance_text（C6 反向标定，取舍 #11）：她**本轮实际说出口的话**。
        话是规范事实、状态是派生估计——话语明显是显式安抚而注入的心情还挂在
        负面时，把 mood/mood_left 校回来（并走账本留痕），**绝不改那句话本身**。
        默认空串：旧调用方（语音写回/巡检）行为完全不变，接线由调度层做。
        全部收进同一把锁。
        """
        kv = services.get("kv_store")
        before = {"intimacy": None}  # 闭包里带出来，账本在锁外记（锁不可重入）

        def _bump(rel: dict) -> dict:
            if not rel:
                # 第一次聊天，从人设配置里拿初始亲密度（读人设文件不经库锁，无死锁风险）
                try:
                    default = kv.read("persona_config", "").default_intimacy
                except Exception as exc:
                    logger.warning(f"[memory] 读初始亲密度失败（按 0 起）: {exc}")
                    default = 0
                rel = {
                    "intimacy": default, "affection": 0, "trust": 0,
                    "interaction_count": 0, "stage": "初识", "last_interaction": "",
                }
            before["intimacy"] = rel.get("intimacy", 0)

            delta = self._EMOTION_DELTA.get(
                emotion, {"intimacy": 1, "affection": 1, "trust": 0}
            )
            rel["intimacy"] = max(0, min(100, rel.get("intimacy", 0) + delta["intimacy"]))
            rel["affection"] = max(0, min(100, rel.get("affection", 0) + delta["affection"]))
            rel["trust"] = max(0, min(100, rel.get("trust", 0) + delta["trust"]))
            rel["interaction_count"] = rel.get("interaction_count", 0) + 1
            rel["last_interaction"] = ClockTool().now()
            new_stage = self._stage_of(rel["intimacy"])
            old_stage = rel.get("stage", "初识")
            if new_stage != old_stage:
                rel["stage"] = new_stage
                if self._stage_rank(new_stage) > self._stage_rank(old_stage):
                    # 升档留个一次性标记（N8）：下一轮 compose 让她自然意识到"更熟了"，
                    # 下一轮写回时会被清掉，只提一次
                    rel["stage_upgraded"] = f"从「{old_stage}」走进了「{new_stage}」"
                else:
                    rel["stage_upgraded"] = ""
            else:
                # 不是本轮升的档：把上一轮可能残留的标记消费掉（置空）
                rel["stage_upgraded"] = ""
            # 心情状态机（N2 惯性版）：数值之外，角色此刻的情绪也往前走一格
            rel["mood"] = self._mood_engine.update(
                rel, emotion, comfort_mode=comfort_mode, intensity=intensity
            )
            # C6 反向标定：她的话说出口之后，反过来校正状态。话语命中显式安抚词、
            # 心情却还挂在负面 → 状态错了，归"平常"、清惯性。硬边界（取舍 #11）：
            # 只动 mood/mood_left/mood_from 这些数值，当轮推出去的话一个字不碰——
            # 做成"发现不一致就重写回复"= 流式下事后撒谎。翻盘走账本，可审计。
            if utterance_text and rel.get("mood") in _NEG_MOODS:
                hit = next((w for w in _SOOTHE_WORDS if w in utterance_text), "")
                if hit:
                    before["mood_recal"] = (rel.get("mood"), hit)  # 账本锁外记，与亲密度账本同一纪律
                    rel["mood"] = "平常"
                    rel["mood_left"] = 0
                    rel["mood_from"] = ""
            # 心事（S2/C7）：concern 是实体 {id,text,created_at,intensity}。
            # 旧病：强度有惯性而文本每轮整句覆盖——同一个强度值每轮挂到不同念头上，
            # 状态被劈成两个不同步的对象（"每轮重掷骰子"）。现在由感知在**同一次
            # 调用**里判 same_concern/resolved：延续=实体不变强度惯性；翻篇=关旧开新。
            concern = dict(rel.get("concern") or {})
            if extras is not None and getattr(extras, "resolved", False):
                concern = {"text": "", "intensity": 0.0}  # 他化解了这个念头：直接归零翻篇
            elif extras is not None and getattr(extras, "concern_text", ""):
                text = extras.concern_text
                try:
                    delta_c = float(extras.concern_delta or 0)
                except (TypeError, ValueError):
                    delta_c = 0.0
                if extras.same_concern and concern.get("text"):
                    # 同一个念头的延续：实体不变（id/created_at 保留），文本允许换
                    # 措辞，强度走惯性
                    concern["text"] = text
                    old_i = float(concern.get("intensity") or 0)
                    concern["intensity"] = round(max(0.0, min(1.0, old_i + delta_c)), 3)
                else:
                    # 新念头（same=false 关旧开新；或 same=true 但旧念头已衰减殆尽）。
                    # 起点 0.5+delta 而不是 0+delta：0+delta 起点在 delta<0.35 时
                    # 永远压在 persona_engine 的表达门槛之下——文本入了库、强度在
                    # 衰减、她却从不把它说出口（隐藏 bug）
                    concern = {
                        "id": uuid.uuid4().hex,
                        "text": text,
                        "created_at": ClockTool().now(),
                        "intensity": round(max(0.0, min(1.0, 0.5 + delta_c)), 3),
                    }
            if comfort_mode:
                # 哄/道歉：心事强度减半（化解），不立刻翻篇
                concern["intensity"] = round(float(concern.get("intensity") or 0) * 0.5, 3)
            ci = float(concern.get("intensity") or 0) * 0.9
            if ci <= 0.05:
                concern = {"text": "", "intensity": 0.0}  # 归零即翻篇（实体 id 一起清）
            else:
                concern["intensity"] = round(ci, 3)
                concern.setdefault("text", "")
            rel["concern"] = concern
            # 关系收敛层（N2 宪法）：他在抱怨她的态度/语气时，对这个人的温度收敛
            # 一格——只动"对他"的刻度，绝不碰性格内核。"收敛不是改变。"
            calm = float(rel.get("calm") or 0)
            if extras is not None and getattr(extras, "feedback", "") == "tone_down":
                calm = min(0.6, calm + 0.15)
            if comfort_mode:
                calm = max(0.0, calm * 0.6)
            rel["calm"] = round(calm, 3)
            # last_poor（F8）在闭包内落账：下一轮 _needs_thinking 读到就强制认真想。
            # 与数值同一把锁写，天然只生效一轮，也不会被并发的整包写覆盖
            rel["last_poor"] = bool(was_poor)
            return rel

        try:
            rel = kv.update("relationship", session_id, _bump)
        except Exception as exc:
            logger.warning(f"[memory] 关系数值原子更新失败（本轮数值不落账）: {exc}")
            return {}
        # 账本在锁外记：append_ledger 自己会拿 SQLite 锁。锁已是可重入的 RLock，
        # 闭包内再取不会自锁，但"可重入"只保证不死锁、不保证语义——嵌套写库会
        # 把外面这次原子更新拆开，所以账本仍坚持在闭包外、update 返回后再记。
        try:
            old_v = before.get("intimacy")
            if old_v is not None and rel:
                # reason 用人话写：账本不只给审计看，S10 氛围线会把它直接拼进提示词
                reason = {
                    "happy": "聊得开心，关系热乎了一点",
                    "sad": "他说了难过的事，信任多了一分",
                    "crisis": "陪他熬过了一段艰难的时刻",
                    "angry": "闹了点不愉快，热度降了一点",
                    "tired": "平平常常聊了一会儿",
                }.get(emotion or "neutral", "又聊了一轮")
                kv.log_relationship_change(session_id, old_v, rel.get("intimacy", 0), reason)
        except Exception as exc:
            logger.warning(f"[memory] 关系账本写入失败（数值本身已生效）: {exc}")
        # C6 反向标定的翻盘也走账本（delta=0 的纯审计行）：状态被"她实际说的话"
        # 校正过必须可追溯，不然哪天心情对不上注入值，查无对证
        recal = before.get("mood_recal")
        if recal and rel:
            old_mood, hit = recal
            try:
                kv.log_relationship_change(
                    session_id, rel.get("intimacy", 0), rel.get("intimacy", 0),
                    f"反向标定：她实际说了「{hit}」这类安抚话，注入的心情「{old_mood}」"
                    f"与话语不符，已归「平常」（C6：只校数值，不动说出口的话）",
                    (utterance_text or "")[:80],
                )
            except Exception as exc:
                logger.warning(f"[memory] 反向标定账本写入失败（校正本身已生效）: {exc}")
        return rel

    @staticmethod
    def _stage_rank(stage: str) -> int:
        ranks = {"初识": 0, "熟悉": 1, "亲近": 2, "挚友": 3}
        return ranks.get(stage, 0)

    def get_stage(self, session_id: str) -> str:
        try:
            rel = services.get("kv_store").read("relationship", session_id) or {}
            return rel.get("stage", "初识")
        except Exception as exc:
            logger.warning(f"[memory] 读关系阶段失败（按「初识」处理）: {exc}")
            return "初识"

    @staticmethod
    def _stage_of(intimacy: int) -> str:
        for threshold, name in RelationshipTracker._STAGES:
            if intimacy >= threshold:
                return name
        return "初识"
