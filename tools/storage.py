"""存储工具：上层要存取数据，统一经过这两个类，不许直接摸数据层的 Store。

- VectorStoreTool：长期记忆的向量检索（Chroma）
- KVStoreTool：档案、画像、关系、聊天记录这些 JSON 存储
"""

from config.settings import get_settings
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
        cfg = get_settings()
        self._chroma = ChromaClient()
        self._embedder = QwenEmbedding()
        self._cols = {}  # 集合名 -> Collection，用的时候现取
        # key 没配（还是占位符）或 Chroma 挂了，都降级成"记忆不可用"，聊天不受影响
        has_key = bool(cfg.embedding_api_key) and not cfg.embedding_api_key.startswith("your-")
        if self._chroma.available and has_key:
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
        """
        metadata = {"embedding_model": cfg.embedding_model, "dimension": cfg.embedding_dimension}
        col = self._chroma.get_collection(collection_name, metadata=metadata)
        meta = col.metadata or {}
        if (
            meta.get("embedding_model") == cfg.embedding_model
            and meta.get("dimension") == cfg.embedding_dimension
        ):
            return col
        # 走到这说明换过模型/维度：旧向量不作数了，导出原文 -> 删库重建 -> 重新嵌入
        old = col.get(include=["documents", "metadatas"])
        ids = old.get("ids") or []
        docs = old.get("documents") or []
        metas = old.get("metadatas") or []
        self._chroma.delete_collection(collection_name)
        col = self._chroma.get_collection(collection_name, metadata=metadata)
        for i in range(0, len(ids), EMBED_BATCH_SIZE):
            batch_docs = docs[i : i + EMBED_BATCH_SIZE]
            col.upsert(
                ids=ids[i : i + EMBED_BATCH_SIZE],
                embeddings=self._embedder.embed_documents(batch_docs),
                documents=batch_docs,
                metadatas=metas[i : i + EMBED_BATCH_SIZE],
            )
        return col

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
                    }
                ],
            )
            return True
        except Exception as exc:
            print(f"[memory] 记忆写入失败: {exc}")
            return False

    def search_memory(self, query: str, top_k: int = 5) -> list:
        """按语义搜记忆，返回 [{content, kind, importance, timestamp}, ...]。

        出任何问题都返回空列表，记忆检索失败不该影响聊天本身。
        """
        col = self._col_of(_DISTILLED_COLLECTION)
        if col is None:
            return []
        try:
            vector = self._embedder.embed_query(query)
            res = col.query(query_embeddings=[vector], n_results=top_k)
        except Exception:
            return []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        items = []
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            items.append(
                {
                    "content": doc,
                    "kind": meta.get("kind", ""),
                    "importance": meta.get("importance", 3),
                    "timestamp": meta.get("timestamp", ""),
                }
            )
        return items

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
        entries.sort(key=lambda e: e["date"])
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


class KVStoreTool:
    """各类 JSON 存储的统一入口，按 store 名字路由到对应的 Store。"""

    def __init__(self):
        self._profile = ProfileStore()
        self._portrait = PortraitStore()
        self._relationship = RelationshipStore()
        self._session = SessionStore()
        self._image = ImageAssetStore()
        self._route = RouteConfigStore()
        self._log = LogStore()
        self._persona_config = PersonaConfigStore()

    def read(self, store: str, key: str, default=None):
        if store == "profile":
            return self._profile.load(key)
        if store == "portrait":
            return self._portrait.load(key)
        if store == "relationship":
            return self._relationship.load(key)
        if store == "session":
            # 聊天记录按条存，读的时候取最近 20 条
            return self._session.get_recent(key, 20)
        if store == "route_config":
            return self._route.load(key)
        if store == "persona_config":
            return self._persona_config.load()
        if store == "image":
            return self._image.list_images()
        return default

    def write(self, store: str, key: str, value) -> bool:
        if store == "profile":
            self._profile.save(key, value)
        elif store == "portrait":
            self._portrait.save(key, value)
        elif store == "relationship":
            self._relationship.save(key, value)
        elif store == "session":
            # value 是一条记录的 dict，进来就追加
            self._session.append(key, DialogueRecord(**value))
        elif store == "image":
            self._image.save(value)
        elif store == "route_config":
            self._route.save(key, value)
        else:
            return False
        return True

    def log(self, entry: dict) -> None:
        self._log.append(entry)

    def chats_between(self, session_id: str, start_iso: str, end_iso: str) -> list:
        """按时间段捞聊天记录（含头不含尾），写日记要用某一天的完整对话就靠它。"""
        return self._session.get_chats_between(session_id, start_iso, end_iso)

    def last_chat_per_session(self) -> list:
        """每个会话最后一次聊天的时间，闲置检测器点名用。"""
        return self._session.last_chat_per_session()
