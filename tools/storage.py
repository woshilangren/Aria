"""存储工具：上层要存取数据，统一经过这两个类，不许直接摸数据层的 Store。

- VectorStoreTool：长期记忆的向量检索（Chroma）
- KVStoreTool：档案、画像、关系、聊天记录这些 JSON 存储
"""

from config.settings import get_settings, load_app_config
from data.chroma_client import ChromaClient
from data.embedding_client import EMBED_BATCH_SIZE, QwenEmbedding
from data.schemas import DialogueRecord, MemoryItem
from data.stores import (
    ImageAssetStore,
    LogStore,
    PersonaConfigStore,
    ProfileStore,
    PortraitStore,
    RelationshipStore,
    RouteConfigStore,
    SelfStore,
    SessionStore,
    new_id,
)

# 长期记忆和日记各占一个集合，互不干扰（换模型重建时也是各建各的）
_DISTILLED_COLLECTION = "distilled_memory"
_DIARY_COLLECTION = "diary"


class VectorStoreTool:
    """长期记忆的存取：写入按向量存，查询按语义搜。

    向量由本类自己算好再传给 Chroma（不把向量化函数挂到集合上），
    这样换向量化模型、重建集合的节奏都握在自己手里。
    """

    def __init__(self):
        # 构造只做便宜且无副作用的事：建 client 和 embedder。
        # 建集合（可能触发"换模型 -> 删库重建"的重活）挪到 ensure_ready()，
        # 由组合根显式调一次——否则任何人 new 一下都可能悄悄删库，这是个陷阱。
        self._chroma = ChromaClient()
        self._embedder = QwenEmbedding()
        self._cols = {}  # 集合名 -> Collection，用的时候现取

    def ensure_ready(self) -> None:
        """建好两个集合（含维度守护），由组合根在装配时显式调用一次。

        与改造前的行为等价：都在 lifespan startup 阶段跑一次；区别只是
        "隐式藏在构造函数里"变成"组合根里看得见、grep 得到的一步"。
        """
        cfg = get_settings()
        # key 没配（还是占位符）或 Chroma 挂了，都降级成"记忆不可用"，聊天不受影响
        has_key = bool(cfg.embedding_api_key) and not cfg.embedding_api_key.startswith("your-")
        if not (self._chroma.available and has_key):
            return
        for name in (_DISTILLED_COLLECTION, _DIARY_COLLECTION):
            try:
                self._cols[name] = self._ensure_collection(name, cfg)
            except Exception as exc:
                print(f"[memory] 集合 {name} 初始化失败，暂时不可用: {exc}")

    def _col_of(self, collection_name: str):
        """拿某个集合，没建出来（初始化失败/降级中）就返回 None。"""
        return self._cols.get(collection_name)

    def _ensure_collection(self, collection_name: str, cfg):
        """拿指定集合，顺手做维度守护：向量化模型换了就重建集合，旧内容搬回去重新嵌入。

        集合 metadata 里记着当前用的模型和维度，启动时对不上就说明换过模型——
        旧向量全是旧模型算的，和新查询没法比，必须导出原文重算一遍。

        重建走两阶段（F2），旧版"读原文进内存 → 删库 → 逐批重嵌入"有个致命洞：
        重嵌入中途失败（网络抖动/key 欠费，恰是换模型后第一次启动最容易发生的时刻）
        = 旧集合已删、新集合只写了一半，全部长期记忆和日记不可恢复地丢失。
        现在改成：先把新向量灌进 `原名__migrating` 临时集合 → 全部成功后才删旧集合 →
        把临时集合的数据（连同算好的向量，零 API 调用）拷进正式集合 → 删临时。
        任何一步失败，旧集合原样健在，下次启动重试——最坏结果只是"还在用旧模型"，绝不丢数据。
        """
        metadata = {"embedding_model": cfg.embedding_model, "dimension": cfg.embedding_dimension}
        col = self._chroma.get_collection(collection_name, metadata=metadata)
        meta = col.metadata or {}
        if (
            meta.get("embedding_model") == cfg.embedding_model
            and meta.get("dimension") == cfg.embedding_dimension
        ):
            return col
        # 走到这说明换过模型/维度：旧向量不作数了
        old = col.get(include=["documents", "metadatas"])
        ids = old.get("ids") or []
        docs = old.get("documents") or []
        metas = old.get("metadatas") or []
        if not ids:
            # 空集合（或只有旧向量没原文）无需迁移，直接按新维度重开
            self._chroma.delete_collection(collection_name)
            return self._chroma.get_collection(collection_name, metadata=metadata)

        tmp_name = f"{collection_name}__migrating"
        try:
            self._chroma.delete_collection(tmp_name)
        except Exception:
            pass  # 临时集合不存在是常态，上次迁移正常收尾就不会留下它
        tmp = self._chroma.get_collection(tmp_name, metadata=metadata)
        for i in range(0, len(ids), EMBED_BATCH_SIZE):
            batch_docs = docs[i : i + EMBED_BATCH_SIZE]
            tmp.upsert(
                ids=ids[i : i + EMBED_BATCH_SIZE],
                embeddings=self._embedder.embed_documents(batch_docs),
                documents=batch_docs,
                metadatas=metas[i : i + EMBED_BATCH_SIZE],
            )
        # 临时集合灌满了才动旧集合：从这里往后只做本地拷贝，不可能再失败
        self._chroma.delete_collection(collection_name)
        final = self._chroma.get_collection(collection_name, metadata=metadata)
        copied = tmp.get(include=["embeddings", "documents", "metadatas"])
        c_ids = copied.get("ids") or []
        c_vecs = copied.get("embeddings") or []
        c_docs = copied.get("documents") or []
        c_metas = copied.get("metadatas") or []
        for i in range(0, len(c_ids), EMBED_BATCH_SIZE):
            final.upsert(
                ids=c_ids[i : i + EMBED_BATCH_SIZE],
                embeddings=c_vecs[i : i + EMBED_BATCH_SIZE],
                documents=c_docs[i : i + EMBED_BATCH_SIZE],
                metadatas=c_metas[i : i + EMBED_BATCH_SIZE],
            )
        self._chroma.delete_collection(tmp_name)
        return final

    def upsert_memory(self, item: MemoryItem) -> bool:
        col = self._col_of(_DISTILLED_COLLECTION)
        if col is None:
            return False
        try:
            col.upsert(
                ids=[item.memory_id or new_id()],
                embeddings=self._embedder.embed_documents([item.content]),
                documents=[item.content],
                metadatas=[
                    {
                        "session_id": item.session_id,
                        "kind": item.kind,
                        "importance": item.importance,
                        "timestamp": item.timestamp,
                        # S1 感受字段（生成时冻结，召回只调制不重写）
                        "feeling": item.feeling,
                        "appraisal": item.appraisal,
                        "valence": item.valence,
                        "arousal": item.arousal,
                    }
                ],
            )
            return True
        except Exception as exc:
            print(f"[memory] 记忆写入失败: {exc}")
            return False

    def search_memory(self, query: str, top_k: int = 5) -> list:
        """按语义搜记忆，重排后返回 [{content, kind, importance, timestamp, feeling, ...}]。

        重排（S1 调制 / S5 生命力雏形）：score = 相似度 × 时间衰减。
        衰减半衰期按 valence 调——**负面刺痛褪得比正面快**（自我保护，
        否则记仇到死；正面/中性 30 天，负面 12 天）。高唤醒衰减稍慢。
        任何字段缺失/坏值都退中性值，行为不比纯语义排序差。
        """
        col = self._col_of(_DISTILLED_COLLECTION)
        if col is None:
            return []
        try:
            vector = self._embedder.embed_query(query)
            res = col.query(
                query_embeddings=[vector], n_results=max(top_k * 2, 8),
                include=["documents", "metadatas", "distances"],
            )
            res_ids = (res.get("ids") or [[]])[0]
        except Exception as exc:
            # 静默返空 = "她忘了所有过去"且零痕迹。至少留一行，能分辨
            # "向量库挂了"和"真没相关记忆"
            print(f"[memory] 记忆检索失败，本轮无召回: {exc}")
            return []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        from datetime import datetime

        from data.sqlite_store import get_db

        # S5 生命力：召回热度批量取一次（常想起的更牢固），最后对最终入选再 touch
        ids_ok = len(res_ids) == len(docs)
        stats = get_db().memory_stats_bulk([i for i in res_ids if i]) if ids_ok else {}
        now = datetime.now()
        items = []
        for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            meta = meta or {}
            doc_id = res_ids[idx] if ids_ok and idx < len(res_ids) else ""
            try:
                sim = 1.0 - float(dist)  # l2 距离转粗相似度（仅用于相对排序）
            except (TypeError, ValueError):
                sim = 0.5
            valence = meta.get("valence", 0.0)
            arousal = meta.get("arousal", 0.3)
            try:
                valence = max(-1.0, min(1.0, float(valence)))
            except (TypeError, ValueError):
                valence = 0.0
            try:
                arousal = max(0.0, min(1.0, float(arousal)))
            except (TypeError, ValueError):
                arousal = 0.3
            half_life = 12.0 if valence < -0.2 else 30.0   # 负面褪得快
            half_life *= (1.0 + 0.5 * arousal)              # 唤醒高的更耐忘
            ts = meta.get("timestamp", "") or ""
            try:
                days = max(0.0, (now - datetime.fromisoformat(ts)).total_seconds() / 86400)
            except (ValueError, TypeError):
                days = 0.0
            decay = 0.5 ** (days / max(half_life, 1.0))
            # 访问加成（S5）：常想起的更牢固，log 压缩防热记忆垄断
            hot = stats.get(doc_id) or {}
            boost = 1.0 + 0.3 * _log1p_safe(hot.get("count", 0))
            items.append(
                {
                    "id": doc_id,
                    "content": doc,
                    "kind": meta.get("kind", ""),
                    "importance": meta.get("importance", 3),
                    "timestamp": ts,
                    "feeling": meta.get("feeling", ""),
                    "appraisal": meta.get("appraisal", ""),
                    "valence": valence,
                    "score": sim * decay * boost,
                }
            )
        items.sort(key=lambda x: x["score"], reverse=True)
        picked = items[:top_k]
        # 召回即 touch：只在最终入选的条目上记热度，失败不影响召回
        try:
            get_db().touch_memories([p["id"] for p in picked if p.get("id")])
        except Exception:
            pass
        return picked


# KV 路由表：store 名 -> (读函数, 写函数)。read/write 都走这张表，加新存储只改这里。
#
# 注意：read 和 write 的 key 集合**故意不对称**，别顺手"对齐"成一样：
#   - read 有 persona_config、write 没有 —— 因为 persona_config 的正主是
#     data/persona_config.json，人设配置本来就不该经 KV 写入，只许读。
#   - read 里的 image 是"列出全部图片"（无 key 概念），write 是"存一张"，
#     两边语义本就不是一对，所以写函数签名上留了 value。
# 真要新加可写的存储，请同时确认它是不是也有"正主在别处"的问题。
    # ---- 日记集合的四个口子：写、搜、列、删。翻篇策略在 capability 层，这里只管存取 ----

    def upsert_diary(self, diary_id: str, content: str, meta: dict) -> bool:
        """写一篇日记。id 带日期（diary-2026-08-31），同一天重写就是覆盖。"""
        col = self._col_of(_DIARY_COLLECTION)
        if col is None:
            return False
        try:
            col.upsert(
                ids=[diary_id],
                embeddings=self._embedder.embed_documents([content]),
                documents=[content],
                metadatas=[meta],
            )
            return True
        except Exception as exc:
            print(f"[diary] 日记写入失败: {exc}")
            return False

    def search_diary(self, query: str, top_k: int = 3) -> list:
        """按语义搜日记，返回 [{content, date, timestamp, session_id}, ...]，挂了返回空。"""
        col = self._col_of(_DIARY_COLLECTION)
        if col is None:
            return []
        try:
            vector = self._embedder.embed_query(query)
            res = col.query(query_embeddings=[vector], n_results=top_k)
        except Exception:
            return []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        return [
            {"content": doc, **(meta or {})}
            for doc, meta in zip(docs, metas)
        ]

    def list_diaries(self) -> list:
        """列出所有日记的 [{id, ...全部元数据}]，按日期从旧到新排，翻篇时挑最旧的用。

        元数据整包透传（date/session_id/timestamp 都在），调用方各取所需。
        """
        col = self._col_of(_DIARY_COLLECTION)
        if col is None:
            return []
        try:
            res = col.get(include=["metadatas"])
        except Exception:
            return []
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        entries = [
            {"id": entry_id, **(meta or {})}
            for entry_id, meta in zip(ids, metas)
        ]
        # 缺 date 的脏数据不能让整个排序炸 KeyError——那会向上炸到 _trim_old_diaries，
        # 被 diary.py 的大 try 吞成"日记没写成"，闲置日记从此永久静默停摆
        entries.sort(key=lambda e: e.get("date", ""))
        return entries

    def delete_diary(self, diary_id: str) -> bool:
        col = self._col_of(_DIARY_COLLECTION)
        if col is None:
            return False
        try:
            col.delete(ids=[diary_id])
            return True
        except Exception as exc:
            print(f"[diary] 日记删除失败: {exc}")
            return False

    def list_diaries_enriched(self) -> list:
        """列出日记并逐条补上正文与元数据，按日期倒序；读不到的条目留空、不抛。

        api.py 展示日记要正文，list_diaries 只给元数据，这里把两步合成一个口子，
        免得外层去摸私有的 _col_of。
        """
        enriched = []
        for d in self.list_diaries():
            try:
                col = self._col_of(_DIARY_COLLECTION)
                res = col.get(ids=[d["id"]], include=["documents", "metadatas"])
                doc = (res.get("documents") or [None])[0] or ""
                meta = (res.get("metadatas") or [{}])[0] or {}
            except Exception:
                doc = ""
                meta = {}
            enriched.append({**d, "content": doc, **meta})
        enriched.sort(key=lambda e: e.get("date", ""), reverse=True)
        return enriched

    # ---- 长期记忆集合的展示/删除口子：列、取、删。语义检索另有 search_memory ----

    def list_memories(self, limit: int = 20) -> list:
        """列出长期记忆 [{id, content, ...meta}]，按 timestamp 倒序取前 limit 条；挂了返回空。"""
        col = self._col_of(_DISTILLED_COLLECTION)
        if col is None:
            return []
        try:
            res = col.get(include=["documents", "metadatas"])
        except Exception:
            return []
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        items = [
            {"id": iid, "content": doc, **(meta or {})}
            for iid, doc, meta in zip(ids, docs, metas)
        ]
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return items[:limit]

    def delete_memory(self, memory_id: str) -> bool:
        col = self._col_of(_DISTILLED_COLLECTION)
        if col is None:
            return False
        try:
            col.delete(ids=[memory_id])
            return True
        except Exception:
            return False


_KV_DISPATCH = {
    "profile": ("_profile", True),
    "portrait": ("_portrait", True),
    "relationship": ("_relationship", True),
    "self": ("_self", True),
    "session": ("_session", True),
    "image": ("_image", True),
    "route_config": ("_route", True),
    "persona_config": ("_persona_config", False),  # 只读：正主在 data/persona_config.json
}

# 支持"读-改-写原子闭包"的 store 白名单：整包语义的 KV 表。
# 不能只看 _KV_DISPATCH 的 writable 标记：session 是"按条追加"、image 是
# "存一张"、route_config 是配置——它们的 Store 上压根没有 update 方法，
# 放它们进来会抛 AttributeError；而 writable=False 的 persona_config 抛 ValueError，
# 错误类型不统一，将来若给某个 Store 补了同名 update 方法，会变成"能调但语义错"。
# 这里显式收紧，语义不符的一律 ValueError，与错用只读表保持同一种失败方式。
# self（她自己的身份）也是整包语义，批次0 收进来。
_KV_UPDATE_TABLES = ("profile", "portrait", "relationship", "self")
class KVStoreTool:
    """各类 JSON 存储的统一入口，按 store 名字路由到对应的 Store。"""

    def __init__(self):
        self._profile = ProfileStore()
        self._portrait = PortraitStore()
        self._relationship = RelationshipStore()
        self._self = SelfStore()
        self._session = SessionStore()
        self._image = ImageAssetStore()
        self._route = RouteConfigStore()
        self._log = LogStore()
        self._persona_config = PersonaConfigStore()

    def read(self, store: str, key: str, default=None):
        entry = _KV_DISPATCH.get(store)
        if entry is None:
            return default
        attr, _writable = entry
        target = getattr(self, attr)
        # session 是聊天记录：按条存。条数以前写死 20、与 memory.max_turns_short_term
        # 脱钩——重启后恢复的短期记忆比运行时长一截（"她醒来记得的比聊着的多"）。
        # 现在读同一份配置，两边一致。
        if store == "session":
            n = int(load_app_config()["memory"].get("max_turns_short_term", 12) or 12)
            return target.get_recent(key, n)
        # image 是列表式资源：读即列出全部，没有 key 的概念
        if store == "image":
            return target.list_images()
        # persona_config 整包一份，没有 key
        if store == "persona_config":
            return target.load()
        return target.load(key)

    def write(self, store: str, key: str, value) -> bool:
        entry = _KV_DISPATCH.get(store)
        # 未知 store、或表里标明只读的（persona_config），一律拒绝
        if entry is None or not entry[1]:
            return False
        target = getattr(self, entry[0])
        if store == "session":
            # value 是一条记录的 dict，进来就追加
            target.append(key, DialogueRecord(**value))
        elif store == "image":
            target.save(value)
        else:
            target.save(key, value)
        return True

    def update(self, store: str, key: str, fn):
        """读-改-写原子闭包（F3）：只对 profile/portrait/relationship 三张整包 KV 表开放。

        上层凡是"读整包 -> 内存改 -> 整包覆盖写"的更新（亲密度、画像、档案）
        都必须走这里，否则两个并发写方会互相覆盖增量（静默漏账）。
        fn 在 SQLite 锁内执行，必须是纯计算——锁内禁网络、禁 LLM。

        白名单见 _KV_UPDATE_TABLES：本次显式收紧，不再依赖 _KV_DISPATCH 的
        writable 标记（那会让 session/image/route_config 抛 AttributeError、
        persona_config 抛 ValueError，错误类型不统一）。
        """
        if store not in _KV_UPDATE_TABLES:
            raise ValueError(f"store {store} 不是整包 KV 表，不支持原子更新")
        # 白名单已保证表名在 dispatch 里且可写，直接取 attr
        return getattr(self, _KV_DISPATCH[store][0]).update(key, fn)

    def log(self, entry: dict) -> None:
        self._log.append(entry)

    def log_relationship_change(self, session_id: str, old: float, new: float,
                                reason: str = "", source_quote: str = "") -> None:
        """关系数值账本（S3）：每次亲密度/信任变动留痕，"你为什么生气"有据可答。"""
        self._relationship.append_ledger(session_id, old, new, reason, source_quote)

    def recent_ledger(self, session_id: str, n: int = 8) -> list:
        """最近 n 条关系账本（旧 -> 新），氛围线聚合趋势用。"""
        return self._relationship.recent_ledger(session_id, n)

    def chats_between(self, session_id: str, start_iso: str, end_iso: str) -> list:
        """按时间段捞聊天记录（含头不含尾），写日记要用某一天的完整对话就靠它。"""
        return self._session.get_chats_between(session_id, start_iso, end_iso)

    def last_chat_per_session(self) -> list:
        """每个会话最后一次聊天的时间，闲置检测器点名用。"""
        return self._session.last_chat_per_session()

    def proactive_after(self, session_id: str, after_iso: str, n: int = 5) -> list:
        """某时刻之后她主动发的话（N3 轮询），旧 -> 新。"""
        return self._session.proactive_after(session_id, after_iso, n)

    def recent_chat(self, session_id: str, n: int = 100) -> list:
        """取某会话最近 n 条聊天记录。

        read("session", key) 内部写死 get_recent(key, 20)、传不进 n，
        api.py 拉历史要自定义条数，走这个专门的口子。
        """
        return self._session.get_recent(session_id, n)




def _log1p_safe(count) -> float:
    """log1p 的安全包装：热度字段坏了退 0（不加成，不炸排序）。"""
    try:
        import math

        return math.log1p(max(0, int(count or 0)))
    except (TypeError, ValueError):
        return 0.0

