"""能力层 - 记忆模块

记忆分四种，各管各的：
- SessionMemoryKeeper：短期记忆，最近几轮原文放内存，太长就让模型压成摘要
- MemoryRecaller：开聊前把档案、画像、关系、长期记忆一次捞齐
- ProfileUpdater / PortraitBuilder：档案和画像的唯二写入口
- ConversationDistiller：从聊天里抓硬事实，收工时写当日总结
- RelationshipTracker：亲密度好感度这些数值的增减
"""

import re
import threading
import time
import uuid
from datetime import datetime

from config.settings import load_app_config
from data.schemas import MemoryItem
from shared.singletons import services
from shared.types import MemoryBundle
from tools.misc import ClockTool, parse_llm_json


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
        self._lock = threading.Lock()

    def _evict_locked(self) -> None:
        """会话数超上限时淘汰最久未碰的那个，三张字典一起清。要求已持锁。"""
        while len(self._contexts) > self._max_sessions:
            victim = min(self._contexts, key=lambda k: self._touched.get(k, 0.0))
            self._contexts.pop(victim, None)
            self._summaries.pop(victim, None)
            self._touched.pop(victim, None)

    def append(self, session_id: str, role: str, text: str) -> None:
        if not text:
            return
        old_part = None
        with self._lock:
            context = self._contexts.setdefault(session_id, [])
            context.append({"role": role, "content": text})
            self._touched[session_id] = time.monotonic()
            # 超长就把最老的一半摘出来；锁内只做截断，LLM 摘要放到锁外
            if len(context) > self._max_turns:
                cut = len(context) // 2
                old_part = context[:cut]
                self._contexts[session_id] = context[cut:]
            self._evict_locked()
        if old_part:
            self._summarize(session_id, old_part)

    def _summarize(self, session_id: str, old_part: list) -> None:
        """把被挤出去的旧对话压成摘要。慢调用在锁外做，做完再持锁写回。

        持锁做模型调用会把所有会话串行化，比不加锁还糟，所以这里先放开锁。
        """
        old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_part)
        try:
            got = services.get("llm").chat(
                [
                    {"role": "system", "content": "把这段对话浓缩成一段话，保留关键信息（名字、约定、聊过的事），100 字以内。"},
                    {"role": "user", "content": old_text},
                ],
                temperature=0.3,
                max_tokens=200,
            )
        except Exception:
            return  # 摘要失败就丢掉最老的一半，聊天不能停
        with self._lock:
            # 写回时以锁内最新值为准（期间别的线程可能已写入新摘要），避免丢更新
            summary = self._summaries.get(session_id, "")
            self._summaries[session_id] = f"{summary}\n{got}".strip() if summary else got

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
        except Exception:
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


class MemoryRecaller:
    """把一个会话能想起来的东西全捞出来，打包成 MemoryBundle 一次带走。"""

    def recall(self, query: str, session_id: str) -> MemoryBundle:
        bundle = MemoryBundle()
        try:
            kv = services.get("kv_store")
            bundle.profile = kv.read("profile", session_id) or {}
            bundle.portrait = kv.read("portrait", session_id) or {}
            bundle.relationship = kv.read("relationship", session_id) or {}
        except Exception:
            pass  # 存储挂了不该拦着聊天，空着继续

        try:
            top_k = load_app_config()["memory"]["recall_top_k"]
            bundle.distilled = services.get("vector_store").search_memory(query or session_id, top_k=top_k)
        except Exception:
            bundle.distilled = []

        # 日记单独搜一遍：日记是角色的回忆，跟硬事实记忆不是一个味，都捞点才聊得起来
        try:
            bundle.diaries = services.get("vector_store").search_diary(query or session_id, top_k=3)
        except Exception:
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
            profile = kv.read("profile", session_id) or {}
            if isinstance(value, list):
                # 列表类字段做去重合并，别越攒越重复
                profile[field_name] = list(dict.fromkeys((profile.get(field_name) or []) + value))
            else:
                profile[field_name] = value
            kv.write("profile", session_id, profile)
            return True
        except Exception:
            return False

    def get_field(self, session_id: str, field_name: str):
        try:
            return (services.get("kv_store").read("profile", session_id) or {}).get(field_name)
        except Exception:
            return None


class PortraitBuilder:
    """软画像：从最近聊天里提炼标签、兴趣、核心需求，顺带 LLM 抽硬事实和缺点。

    每 5 轮才跑一次（调用方控制），模型调用开销平摊在这几轮里；
    抽取结果带置信度，低于门槛的不入库——这是记忆防污染的第一道闸。
    失败就静默，别打扰聊天。
    """

    def refresh(self, session_id: str, recent_dialogues: list) -> None:
        if not recent_dialogues:
            return
        try:
            kv = services.get("kv_store")
            old = kv.read("portrait", session_id) or {}
            # 上下文元素是 {"role","content"} 字典，翻成人话再喂模型
            chats = "\n".join(
                (
                    f"{'用户' if t.get('role') == 'user' else 'AI'}：{t.get('content', '')}"
                    if isinstance(t, dict)
                    else str(t)
                )
                for t in recent_dialogues[-10:]
            )
            raw = services.get("llm").chat(
                [
                    {"role": "system", "content": (
                        "从聊天记录里总结用户画像，只输出 JSON：\n"
                        '{"portrait_tags": ["标签"], "core_needs": "一句话", "interests": ["兴趣"], '
                        '"hard_facts": [{"field": "字段名", "value": "值", "confidence": 0到1的小数}], '
                        '"flaws": ["用户的缺点、翻车、出糗事"]}\n'
                        "标签不超过 5 个；hard_facts 的 field 只能取 "
                        "nickname/gender/age/city/occupation/birthday/likes/dislikes/notable_facts 之一，"
                        "只收用户明确说出口的事实，拿不准就给低 confidence；"
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

            distiller = ConversationDistiller()
            # 硬事实走蒸馏器的入库通道：改口连着旧值一起更新，没变不重复记
            for fact in data.get("hard_facts") or []:
                if not isinstance(fact, dict):
                    continue
                try:
                    confidence = float(fact.get("confidence", 0))
                except (TypeError, ValueError):
                    continue
                field = str(fact.get("field", "")).strip()
                value = fact.get("value")
                if confidence >= threshold and field and value not in (None, ""):
                    distiller.apply_fact(session_id, field, value)

            # 缺点/翻车记录：画像存一份（给提示词当调侃素材），向量库存一份（语义召回）
            old_flaws = set(old.get("user_flaws") or [])
            for flaw in data.get("flaws") or []:
                flaw = str(flaw).strip()
                if flaw and flaw not in old_flaws:
                    distiller.remember_flaw(session_id, flaw)

            kept_tags, tag_updated = self._decay_tags(old, data.get("portrait_tags"), limit)
            portrait = {
                "portrait_tags": kept_tags,
                "core_needs": data.get("core_needs") or old.get("core_needs", ""),
                "interests": self._merge(old.get("interests"), data.get("interests"), limit),
                "relationship_assessment": old.get("relationship_assessment", ""),
                "user_flaws": self._merge_flaws(old.get("user_flaws"), data.get("flaws")),
                "tag_updated": tag_updated,
                "updated_at": ClockTool().now(),
            }
            kv.write("portrait", session_id, portrait)
        except Exception:
            pass  # 画像更新失败无所谓，下轮再试

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
        try:
            now_dt = datetime.fromisoformat(now)
        except ValueError:
            now_dt = None
        kept = []
        for tag in merged:
            ts = tag_updated.get(tag, "")
            if now_dt and ts:
                try:
                    age_days = (now_dt - datetime.fromisoformat(ts)).total_seconds() / 86400
                    if age_days > decay_days:
                        tag_updated.pop(tag, None)
                        continue  # 太久没再出现的印象，过期清掉
                except ValueError:
                    pass  # 时间戳坏了就当没记过，标签保留
            kept.append(tag)
        return kept[:limit], tag_updated


class ConversationDistiller:
    """长期记忆沉淀：正则抓硬事实（比让模型抽又稳又省），重要的事写进向量库。"""

    # 硬事实抓取规则：字段名 -> 正则
    _FACT_PATTERNS = [
        ("nickname", re.compile(r"我(?:叫|的名字是|叫做)([\u4e00-\u9fa5A-Za-z0-9]{1,10})")),
        ("city", re.compile(r"我(?:住在|住|生活)在?([\u4e00-\u9fa5]{2,8}?)(?:市|区|省)")),
        ("age", re.compile(r"我[\u4e00-\u9fa5，, ]{0,10}?(\d{1,2})岁")),
        ("birthday", re.compile(r"(?:我的)?生日(?:是)?(\d{1,2}月\d{1,2}[号日])")),
        ("occupation", re.compile(
            r"我(?:是|在做|从事)(?:一名)?(学生|老师|程序员|医生|护士|工程师|设计师|公务员|销售|司机|工人|厨师|会计|律师)"
        )),
    ]
    _FIELD_LABELS = {
        "nickname": "叫", "city": "住在", "age": "今年",
        "birthday": "生日是", "occupation": "职业是",
    }

    # 匹配点跟前出现这些字就算否定（"我不叫X""我不住在Y"），别把反话当事实收
    _NEGATION_WORDS = ("不", "没", "别", "未")

    @classmethod
    def _is_negated(cls, text: str, start: int) -> bool:
        """匹配起点前两个字内出现否定词就当没说。"""
        window = text[max(0, start - 2):start]
        return any(n in window for n in cls._NEGATION_WORDS)

    def apply_fact(self, session_id: str, field: str, value) -> bool:
        """硬事实入库的唯一通道：值没变一条记忆都不加，变了连改口一起记。

        正则和 LLM 抽取都汇到这，防污染的两个关键动作（去重、改口更新）只写这一遍。
        """
        if field not in ProfileUpdater._FIELDS:
            return False
        try:
            old = (services.get("kv_store").read("profile", session_id) or {}).get(field)
        except Exception:
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
            if not m or self._is_negated(user_text, m.start()):
                continue
            self.apply_fact(session_id, field, m.group(1))

    def distill_day(self, session_id: str, texts: list) -> None:
        """收工时把今天聊的归成一段总结存起来，明天聊起来有话说。"""
        if not texts:
            return
        try:
            joined = "\n".join(str(t) for t in texts[-40:])
            summary = services.get("llm").chat(
                [
                    {"role": "system", "content": "把这段聊天总结成一段话，重点记用户提到的事、约定和情绪，120 字以内。"},
                    {"role": "user", "content": joined},
                ],
                temperature=0.3,
                max_tokens=250,
            )
            self._remember(session_id, summary, "daily_summary", 3)
        except Exception:
            pass

    def remember_flaw(self, session_id: str, flaw: str) -> None:
        """把用户的缺点/翻车记进长期记忆，语义召回时能捞出来当调侃素材。"""
        if flaw:
            self._remember(session_id, f"他的缺点/翻车：{flaw}", "user_flaw", 4)

    def _remember(self, session_id: str, content: str, kind: str, importance: int) -> None:
        item = MemoryItem(
            memory_id=uuid.uuid4().hex,
            session_id=session_id,
            kind=kind,
            content=content,
            importance=importance,
            timestamp=ClockTool().now(),
        )
        try:
            services.get("vector_store").upsert_memory(item)
        except Exception:
            pass  # 向量库挂了就先不记，聊天照常


class RelationshipTracker:
    """关系数值：每次聊完按情绪微调，数值攒着决定关系阶段。"""

    # 不同情绪带来的数值变化，正负都写明白
    _EMOTION_DELTA = {
        "happy": {"intimacy": 2, "affection": 2, "trust": 1},
        "sad": {"intimacy": 1, "affection": 3, "trust": 1},
        "crisis": {"intimacy": 0, "affection": 4, "trust": 2},
        "angry": {"intimacy": -1, "affection": -2, "trust": -1},
        "tired": {"intimacy": 1, "affection": 1, "trust": 0},
    }
    _STAGES = ((70, "挚友"), (45, "亲近"), (20, "熟悉"), (0, "初识"))

    def update(self, session_id: str, emotion: str = "neutral") -> dict:
        kv = services.get("kv_store")
        rel = kv.read("relationship", session_id) or {}
        if not rel:
            # 第一次聊天，从人设配置里拿初始亲密度
            try:
                default = kv.read("persona_config", "").default_intimacy
            except Exception:
                default = 0
            rel = {
                "intimacy": default, "affection": 0, "trust": 0,
                "mood_baseline": "平静", "interaction_count": 0,
                "stage": "初识", "last_interaction": "",
            }

        delta = self._EMOTION_DELTA.get(emotion, {"intimacy": 1, "affection": 1, "trust": 0})
        rel["intimacy"] = max(0, min(100, rel.get("intimacy", 0) + delta["intimacy"]))
        rel["affection"] = max(0, min(100, rel.get("affection", 0) + delta["affection"]))
        rel["trust"] = max(0, min(100, rel.get("trust", 0) + delta["trust"]))
        rel["interaction_count"] = rel.get("interaction_count", 0) + 1
        if emotion and emotion != "neutral":
            rel["mood_baseline"] = emotion
        rel["last_interaction"] = ClockTool().now()
        rel["stage"] = self._stage_of(rel["intimacy"])
        try:
            kv.write("relationship", session_id, rel)
        except Exception:
            pass
        return rel

    def get_stage(self, session_id: str) -> str:
        try:
            rel = services.get("kv_store").read("relationship", session_id) or {}
            return rel.get("stage", "初识")
        except Exception:
            return "初识"

    @staticmethod
    def _stage_of(intimacy: int) -> str:
        for threshold, name in RelationshipTracker._STAGES:
            if intimacy >= threshold:
                return name
        return "初识"
