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
from capability.quirks import MoodEngine
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
            self._summarize(session_id, old_part)

    def _summarize(self, session_id: str, old_part: list) -> None:
        """把被挤出去的旧对话压成摘要。慢调用在锁外做，做完再持锁写回。

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
        except Exception:
            return  # 摘要失败就丢掉最老的一半，聊天不能停
        with self._lock:
            if needs_recompress:
                # 重压缩是把"旧摘要+新段落"融成一段：直接替换，不能再追加
                # （追加会让旧摘要的内容出现两份）
                self._summaries[session_id] = got.strip()
            else:
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
        except Exception:
            return False

    def get_field(self, session_id: str, field_name: str):
        try:
            return (services.get("kv_store").read("profile", session_id) or {}).get(field_name)
        except Exception:
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
    """

    _PROMOTION_HITS = 2

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
        except Exception:
            return  # 候选池挂了，本轮放弃（和记忆写入同一兜底哲学）
        if hits < self._PROMOTION_HITS:
            return  # 证据不足，继续攒

        # 达到晋升门槛：仲裁新旧值
        distiller = ConversationDistiller()
        try:
            old = (services.get("kv_store").read("profile", session_id) or {}).get(field)
        except Exception:
            old = None
        verdict = self._arbitrate(field, old, content, quote)
        if verdict == "keep":
            return  # 仲裁认定口误/玩笑：保留旧值，候选标晋升（不再反复仲裁）
        if verdict is None:
            return  # 仲裁失败（LLM 挂/输出坏）：不标晋升，下次出现重试
        if distiller.apply_fact(session_id, field, content):
            try:
                from data.sqlite_store import get_db as _gdb

                _gdb().mark_candidate_promoted(cand_id)
            except Exception:
                pass

    def _arbitrate(self, field: str, old, new: str, quote: str) -> "str | None":
        """矛盾仲裁：返回 "update" / "keep" / None（失败）。无冲突直接 update。"""
        if not old or str(old).strip() == new.strip():
            return "update"
        try:
            raw = services.get("llm").chat(
                [
                    {"role": "system", "content": (
                        "你是档案仲裁员。档案里已有一个旧值，对话里出现了新信息。"
                        "判断新信息是【真变更】（他生活变了，应更新）还是【临时状态/口误/玩笑】"
                        "（不应更新）。只输出 JSON：{\"action\":\"update\"或\"keep\","
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
            gate = MemoryGatekeeper()
            # 硬事实入库（S4）：标量事实走"候选→晋升→仲裁"闸门，绝不直接入档；
            # 列表字段（likes/dislikes）是合并语义（apply_fact 内部去重），保留直通
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
                if isinstance(value, list):
                    if confidence >= threshold:
                        distiller.apply_fact(session_id, field, value)
                else:
                    gate.process(session_id, field, str(value), confidence=confidence)

            # 画像整包读改写走原子闭包（F3）：锁内只做纯合并计算，
            # 期间别的写方（语音写回/巡检）落的字段不会被这次覆盖掉
            new_flaws: list = []  # 闭包里算出的"这次真正新出现的缺点"，出锁后喂向量库

            def _apply(old: dict) -> dict:
                old = old or {}
                pre_flaws = set(old.get("user_flaws") or [])
                for flaw in data.get("flaws") or []:
                    flaw = str(flaw).strip()
                    if flaw and flaw not in pre_flaws:
                        new_flaws.append(flaw)
                kept_tags, tag_updated = self._decay_tags(old, data.get("portrait_tags"), limit)
                return {
                    "portrait_tags": kept_tags,
                    "core_needs": data.get("core_needs") or old.get("core_needs", ""),
                    "interests": self._merge(old.get("interests"), data.get("interests"), limit),
                    "relationship_assessment": old.get("relationship_assessment", ""),
                    "user_flaws": self._merge_flaws(old.get("user_flaws"), data.get("flaws")),
                    "tag_updated": tag_updated,
                    "updated_at": ClockTool().now(),
                }

            kv.update("portrait", session_id, _apply)

            # 缺点/翻车记录：画像存一份（上面闭包里），向量库存一份（语义召回）
            for flaw in new_flaws[:3]:
                distiller.remember_flaw(session_id, flaw)
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
                summary = str(data.get("summary") or "").strip()
                feeling = str(data.get("feeling") or "").strip()[:60]
                appraisal = str(data.get("appraisal") or "").strip()[:80]
                try:
                    valence = max(-1.0, min(1.0, float(data.get("valence") or 0)))
                    arousal = max(0.0, min(1.0, float(data.get("arousal") or 0.3)))
                except (TypeError, ValueError):
                    valence, arousal = 0.0, 0.3
            except Exception:
                summary, feeling, appraisal, valence, arousal = (raw or "").strip(), "", "", 0.0, 0.3
            if summary:
                self._remember(session_id, summary, "daily_summary", 3,
                               feeling=feeling, appraisal=appraisal,
                               valence=valence, arousal=arousal)
        except Exception:
            pass

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
            services.get("vector_store").upsert_memory(item)
        except Exception:
            pass  # 向量库挂了就先不记，聊天照常


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
        try:
            diff = (now - datetime.fromisoformat(entry.get("time", ""))).days
            label = {0: "今天", 1: "昨天", 2: "前天"}.get(diff, f"{diff}天前")
        except (ValueError, TypeError):
            label = ""
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
    except Exception:
        return ""  # 氛围线是增强项，任何异常都不注入


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
               intensity: float = 0.5, extras=None) -> dict:
        """按情绪微调数值，一次原子闭包落好数值 + 心情 + 心事 + 收敛 + 阶段标记。

        was_poor（F8）：本轮是否被判敷衍/违规重写，收进闭包一起写。以前 pipeline
        在闭包返回后又整包 kv.write 一次 relationship，那正是 F3 要消灭的
        "读整包 -> 内存改 -> 整包覆盖写"——并发的 REST 增量会被这次陈旧覆盖吞掉。
        放进闭包后天然只生效一轮（下一轮写回会带着新值覆盖），且不再有第二次写。
        默认 False 保证旧调用方（如语音写回）不受影响。

        intensity：本轮情绪强度（N2 惯性时长随它走）；extras（S2/N2）：感知的
        附加产出（心事设置/加减 + 语气反馈），没有传 None。全部收进同一把锁。
        """
        kv = services.get("kv_store")
        before = {"intimacy": None}  # 闭包里带出来，账本在锁外记（锁不可重入）

        def _bump(rel: dict) -> dict:
            if not rel:
                # 第一次聊天，从人设配置里拿初始亲密度（读人设文件不经库锁，无死锁风险）
                try:
                    default = kv.read("persona_config", "").default_intimacy
                except Exception:
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
            # 心事（S2）：感知可设置/加减，安抚化解（减半），每轮自动衰减（时间冲淡）
            concern = dict(rel.get("concern") or {})
            if extras is not None and getattr(extras, "concern_text", ""):
                concern["text"] = extras.concern_text
                old_i = float(concern.get("intensity") or 0)
                concern["intensity"] = round(
                    max(0.0, min(1.0, old_i + float(extras.concern_delta or 0))), 3
                )
            if comfort_mode:
                # 哄/道歉：心事强度减半（化解），不立刻翻篇
                concern["intensity"] = round(float(concern.get("intensity") or 0) * 0.5, 3)
            ci = float(concern.get("intensity") or 0) * 0.9
            if ci <= 0.05:
                concern = {"text": "", "intensity": 0.0}  # 归零即翻篇
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
        except Exception:
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
        except Exception:
            pass  # 账本是审计面，写失败不影响数值本身
        return rel

    @staticmethod
    def _stage_rank(stage: str) -> int:
        ranks = {"初识": 0, "熟悉": 1, "亲近": 2, "挚友": 3}
        return ranks.get(stage, 0)

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
