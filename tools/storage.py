"""存储工具：上层要存取数据，统一经过这两个类，不许直接摸数据层的 Store。

- VectorStoreTool：长期记忆的向量检索（Chroma）
- KVStoreTool：档案、画像、关系、聊天记录这些 JSON 存储；
  另外挂着 J9 的磁盘保留期修剪口子（chat_log / uploads / images）——
  只提供方法，挂不挂巡检由 capability 层决定
"""

import json
import math
from datetime import datetime
from pathlib import Path

from config.settings import get_settings, load_app_config
from data.chroma_client import ChromaClient
from shared.timeutils import safe_delta_seconds
from data.embedding_client import EMBED_BATCH_SIZE, QwenEmbedding
from data.schemas import DialogueRecord, MemoryItem
from data.sqlite_store import get_db
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


def _norm_embeddings(value):
    """把 Chroma 返回的 embeddings 归一成可切片的列表；空/None 归一成 []。

    为什么不能用 `or []`：Chroma（1.x）`get(include=["embeddings"])` 返回的是
    **ndarray**，对它做真值判断直接抛 "truth value of an array is ambiguous"。
    崩点还偏偏在**旧集合已删之后**（迁移的复制阶段）——抛了就留下"主集合空、
    `__migrating` 里还有全量数据"的半迁移现场，下次启动因 metadata 匹配直接
    接受空集合（R01a 隔离探针复现过）。所以这里只用 None 判断 + len()：
    list 和 ndarray 都安全，真值判断一个都不做。
    """
    if value is None:
        return []
    return value if len(value) > 0 else []


# R01b：迁移状态文件放 DATA_DIR（跟 chroma 数据同区，绝不放仓库根）。
# 记录"哪个集合迁移到哪一步"——进程在迁移中途死掉后，下次启动靠它续命；
# 文件缺失/损坏按"无记录"处理，恢复逻辑另有遗留事故探测兜底，绝不能因为
# 状态读不出来就把唯一副本当垃圾清掉。
_MIGRATION_STATE_FILE = "vector_migration_state.json"


def _load_migration_state(path: Path) -> dict:
    """读迁移状态。缺失返回 {}；损坏也返回 {} 但大声告警——宁可退化成
    "无状态文件的遗留现场"（靠探测恢复），不许静默。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[memory] 迁移状态文件损坏，按无记录处理（不得据此删任何数据）: {exc}")
        return {}


def _save_migration_state(path: Path, state: dict) -> None:
    """写迁移状态。写失败不致命：最坏退化为"无状态文件的遗留现场"，
    恢复逻辑照样能靠'主集合空 + __migrating 有货'探测兜底，所以只告警不抛。"""
    try:
        path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as exc:
        print(f"[memory] 迁移状态写入失败（恢复将退化为遗留现场探测）: {exc}")


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
        # R01b：迁移状态随实例加载（"重启"= 新实例从磁盘重新读到上次现场）
        self._state_file = get_settings().data_dir / _MIGRATION_STATE_FILE
        self._migration_state = _load_migration_state(self._state_file)

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
                # R01b：走到这里的失败含"唯一副本不可恢复"——明确停用该集合并大声
                # 告警，绝不许把"空主集合 + metadata 正确"当成健康继续用
                print(f"[memory] 集合 {name} 初始化失败，该集合停用（聊天不受影响，此集合记忆检索不可用）: {exc}")

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

        R01b 追加的可恢复性：
        - **先恢复、再判 metadata**：入口先查未完成的迁移（状态文件 + 无状态文件的
          遗留现场），处理完才轮到 metadata 匹配判断——否则"主集合空 + metadata
          正确"会被当成健康集合直接返回，伪装成数据丢了。
        - **状态文件**：迁移各阶段落 DATA_DIR/vector_migration_state.json，崩在
          任何一步，下次启动都能从现场恢复。
        - **删除前的验证门**：删旧集合之前必须逐 ID 验证临时集合完整；删临时
          集合之前必须逐 ID 验证正式集合完整。验证不过就抛错，一个集合都不许删。
        """
        metadata = {"embedding_model": cfg.embedding_model, "dimension": cfg.embedding_dimension}
        # ① 未完成迁移优先处理（恢复 / 清场 / 明确停用），之后才轮到 metadata 匹配
        self._recover_incomplete(collection_name, cfg, metadata)
        col = self._chroma.get_collection(collection_name, metadata=metadata)
        meta = col.metadata or {}
        if (
            meta.get("embedding_model") == cfg.embedding_model
            and meta.get("dimension") == cfg.embedding_dimension
        ):
            # metadata 匹配 = 迁移已收尾：状态记录一并清掉（可能有收尾前崩的残留）
            if self._migration_state.pop(collection_name, None) is not None:
                self._save_state()
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
        # 登记迁移状态（嵌入阶段）。崩在这段=旧主集合完好，恢复路径会清场重走。
        self._migration_state[collection_name] = {
            "phase": "embedding",
            "tmp_name": tmp_name,
            "target": dict(metadata),
            "expected_ids": list(ids),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_state()
        for i in range(0, len(ids), EMBED_BATCH_SIZE):
            batch_docs = docs[i : i + EMBED_BATCH_SIZE]
            tmp.upsert(
                ids=ids[i : i + EMBED_BATCH_SIZE],
                embeddings=self._embedder.embed_documents(batch_docs),
                documents=batch_docs,
                metadatas=metas[i : i + EMBED_BATCH_SIZE],
            )
        # 删除旧集合之前必须逐 ID 验证临时集合完整——这是"删除旧集合"的唯一门票
        self._verify_complete(tmp, set(ids), f"临时集合 {tmp_name}")
        self._migration_state[collection_name]["phase"] = "copying"
        self._save_state()
        # 从这里往后只做本地拷贝；任何一步崩了，恢复逻辑从 tmp 整包拷回
        self._chroma.delete_collection(collection_name)
        final = self._chroma.get_collection(collection_name, metadata=metadata)
        self._copy_rows(tmp, final)
        # 删临时集合之前必须逐 ID 验证正式集合完整——唯一副本此时才允许换手
        self._verify_complete(final, set(ids), f"正式集合 {collection_name}")
        self._chroma.delete_collection(tmp_name)
        self._migration_state.pop(collection_name, None)
        self._save_state()
        return final

    # ---------------- R01b：迁移状态与恢复 ----------------

    def _save_state(self) -> None:
        _save_migration_state(self._state_file, self._migration_state)

    def _col_ids(self, col) -> list:
        """集合里现存的全部 ID（只取 ids，不搬数据）。集合不可读返回空。"""
        try:
            return list(col.get()["ids"] or [])
        except Exception:
            return []

    def _verify_complete(self, col, expected_ids: set, label: str) -> None:
        """逐 ID 校验集合与预期完全一致；不过就抛错。任何删除动作前必须先过这道门：
        宁可迁移重来（数据都在），不可在副本不完整时删掉另一份。"""
        got = self._col_ids(col)
        missing = expected_ids - set(got)
        extra = set(got) - expected_ids
        if missing or extra or len(got) != len(expected_ids):
            raise RuntimeError(
                f"[memory] {label} 校验失败：预期 {len(expected_ids)} 条，实际 {len(got)} 条"
                f"（缺 {len(missing)}，多 {len(extra)}）——该状态下禁止删除任何集合"
            )

    def _copy_rows(self, src, dst) -> None:
        """把 src 的全部数据（连同向量，零 API 调用）整包拷进 dst。"""
        data = src.get(include=["embeddings", "documents", "metadatas"])
        ids = data.get("ids") or []
        vecs = _norm_embeddings(data.get("embeddings"))
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        for i in range(0, len(ids), EMBED_BATCH_SIZE):
            dst.upsert(
                ids=ids[i : i + EMBED_BATCH_SIZE],
                embeddings=vecs[i : i + EMBED_BATCH_SIZE],
                documents=docs[i : i + EMBED_BATCH_SIZE],
                metadatas=metas[i : i + EMBED_BATCH_SIZE],
            )

    def _recover_incomplete(self, collection_name: str, cfg, metadata: dict) -> None:
        """处理未完成的迁移。三种结局：
        ① `__migrating` 有完整副本 → 不管主集合是旧数据/空/半拷/不存在，整包拷回收尾；
        ② 嵌入阶段中断（临时集合不完整）且旧主集合健在 → 清掉半截临时集合，重走迁移；
        ③ 主集合缺失/为空且临时集合不完整 → 唯一副本已不可恢复，抛错让 ensure_ready
           明确停用该集合并告警——绝不许"空主集合 + metadata 正确"伪装健康。
        无状态文件的遗留事故（升级前留下的现场）也在这里兜住：主集合 metadata 已是
        目标值但空、`__migrating` 有货 → 恢复。
        """
        raw = self._chroma.client
        state = self._migration_state
        rec = state.get(collection_name)
        tmp_name = f"{collection_name}__migrating"

        def _try_get(name):
            # 只查不建：get_collection（chroma 裸客户端）对不存在的名字抛错
            try:
                return raw.get_collection(name)
            except Exception:
                return None

        tmp = _try_get(tmp_name)
        tmp_ids = self._col_ids(tmp) if tmp is not None else []

        if rec and rec.get("phase") in ("embedding", "copying") and rec.get("expected_ids"):
            expected = set(rec["expected_ids"])
            tmp_full = tmp is not None and set(tmp_ids) == expected and len(tmp_ids) == len(expected)
            if tmp_full:
                # ① 完整副本在场：主集合无论什么状态，整包拷回即收尾
                main = _try_get(collection_name)
                if main is None:
                    main = self._chroma.get_collection(collection_name, metadata=metadata)
                self._copy_rows(tmp, main)
                self._verify_complete(main, expected, f"恢复后的主集合 {collection_name}")
                try:
                    # 主集合可能还挂着旧模型的 metadata，拷完后对齐目标值，
                    # 否则下次启动会把这些新向量当旧向量再迁一遍（浪费但不出错）
                    main.modify(metadata=metadata)
                except Exception:
                    pass  # 对齐失败最坏是重迁一遍，不影响正确性
                self._chroma.delete_collection(tmp_name)
                state.pop(collection_name, None)
                self._save_state()
                print(f"[memory] 集合 {collection_name} 从 __migrating 恢复 {len(expected)} 条（上次迁移中断）")
                return
            # ② 临时集合不完整：旧主集合健在就清场，重走完整迁移
            main = _try_get(collection_name)
            if main is not None and main.count() > 0:
                try:
                    self._chroma.delete_collection(tmp_name)
                except Exception:
                    pass
                state.pop(collection_name, None)
                self._save_state()
                print(f"[memory] 集合 {collection_name} 上次迁移中断于嵌入阶段，旧数据完好，重新迁移")
                return
            # ③ 唯一副本已丢：明确停用，绝不能伪装成健康空集合
            raise RuntimeError(
                f"[memory] 向量迁移不可恢复：{collection_name} 主集合缺失或为空，"
                f"且 __migrating 不完整（预期 {len(expected)} 条，临时集合 {len(tmp_ids)} 条）"
                f"——该集合停用，需要从备份恢复"
            )

        # 遗留事故（无状态文件）：主集合已是目标元数据但空、__migrating 有货 → 恢复
        if tmp is not None and tmp_ids:
            main = _try_get(collection_name)
            if main is None:
                # 主集合整个没了但副本在场：按目标元数据重建后恢复
                main = self._chroma.get_collection(collection_name, metadata=metadata)
            if main.count() == 0:
                if (main.metadata or {}).get("embedding_model") != cfg.embedding_model:
                    # metadata 还没切到目标值：这不是"删了主集合后崩"的现场，
                    # 交给正常迁移流程处理（旧数据健在时 tmp 会被清掉重来）
                    return
                self._copy_rows(tmp, main)
                got = set(self._col_ids(main))
                if got and got == set(tmp_ids):
                    self._chroma.delete_collection(tmp_name)
                    print(
                        f"[memory] 集合 {collection_name} 检测到历史遗留的半迁移现场，"
                        f"已从 __migrating 恢复 {len(got)} 条"
                    )
            elif set(self._col_ids(main)) == set(tmp_ids):
                # 收尾前崩的另一种形态：主集合已完整、只差删临时。ID 集合一致才删，
                # 不一致就留着并告警——禁止盲删 __migrating（它可能是唯一完整副本）
                self._chroma.delete_collection(tmp_name)
                print(f"[memory] 集合 {collection_name} 清理了上次迁移残留的临时集合（主集合已完整）")
            else:
                print(
                    f"[memory] 集合 {collection_name} 存在与主集合不一致的 __migrating 残留"
                    f"（主 {main.count()} 条 / 临 {len(tmp_ids)} 条），保守保留待人工检查"
                )
        elif tmp is not None and not tmp_ids:
            # 空临时集合残留：没有任何数据价值，清掉
            try:
                self._chroma.delete_collection(tmp_name)
            except Exception:
                pass

    def upsert_memory(self, item: MemoryItem) -> bool:
        """写一条长期记忆。契约（R04）：**明确 True 才算成功**；False 表示
        失败（集合未装配 / 内部写失败，内部已告警），调用方必须消费这个
        False——走它自己的失败入口，不许当成功吞掉。异常向上抛（同样必须
        被调用方消费），本方法不自作主张吞异常。"""
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

    def search_memory(self, query: str, top_k: int = 5, touch: bool = True) -> list:
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
            # R08：语义置信分 = 1/(1+d)，非负且随距离单调递减。
            # 旧 1-d 在 l2 距离 >1 时为负，负分再乘衰减/热度会让**更旧**的记忆
            # 反而排更前（越负乘得越负）。NaN/Infinity/负距离是异常数据，
            # 直接跳过——宁可少一条，不许它进榜首冒充"最相关"。
            try:
                d = float(dist)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(d) or d < 0:
                continue
            similarity = 1.0 / (1.0 + d)
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
            # R09b：统一解析——坏值/未来时间戳降级为 0（= 当作刚发生，旧行为）
            age_seconds = safe_delta_seconds(now, ts)
            days = max(0.0, (age_seconds or 0.0) / 86400)
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
                    # 语义置信分（只看距离，S7 措辞用它）与排序分（乘了衰减与
                    # 热度，只用于排序）是**两个量**——混用会让热门召回把模糊
                    # 记忆说得斩钉截铁（R08）
                    "similarity": similarity,
                    "score": similarity * decay * boost,
                }
            )
        items.sort(key=lambda x: x["score"], reverse=True)
        picked = items[:top_k]
        # 召回即 touch：只在最终入选的条目上记热度，失败不影响召回。
        # R17d：对话轮内的召回**不在召回当场记**——取消/降级轮不许留下学习
        # 热度（业务更新必须等提交裁决），调用方传 touch=False 并把入选 id
        # 带进 PreparedTurn，由写回协调器在提交成功后补记。turn 之外的独立
        # 检索口子（记忆搜索 API）维持原行为。
        if touch:
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
        elif store == "self":
            # E7：她的身份不许被整包覆盖，走冻结版写入（见 _write_self_frozen）
            return self._write_self_frozen(key, value)
        else:
            target.save(key, value)
        return True

    def _write_self_frozen(self, session_id: str, value) -> bool:
        """`self` 表的冻结在 write() 这条路上也要成立（E7）：只填空键，绝不覆盖已冻结的键。

        为什么需要：`_merge_freeze`（capability/self_identity.py）走的是 `update()`
        原子闭包，锁内逐键跳过非空值——那条路是原子的、正确的。但 `write()` 是整包
        `save()`：任何调用方都能一次性把她已经定下的名字/年龄/城市改写掉，
        "第一次说出口即冻结"（CLAUDE.md 人设立场）在这条路上形同虚设。
        查过现有代码：身份冻结、char_life 的 life 状态、self_narrative **全都**走
        `update()`，没有任何一处正当地调 `write("self", ...)`——所以这道闸只挡误用，
        不挡任何正在干活的人。

        规则：
        - 目标键空 → 填进去；
        - 目标键已非空、传入值**相同** → 当幂等重写，不算冲突
          （不然"读出来原样存回去"这种无害调用会被误杀）；
        - 目标键已非空、传入值不同 → **整包拒绝**（返回 False）并记 warning。
          不选"悄悄丢掉冲突键、把其余的写进去"：半成功的写入让调用方没法判断
          self 表现在到底是什么状态，比一次明确的失败难查得多；
        - value 不是 dict / 当前身份读不出来 → 也拒绝。读不出当前值就没法判断
          哪些键已冻结，这时候放行等于把冻结整个摘掉。

        ⚠ 这**不是**原子操作（读-改-写，中间没有锁）。真正原子的冻结仍然只在
        `update()` 的闭包里——这里只是把"任何人都能整包覆盖"这扇敞开的门关上，
        没有把 SQLite 锁搬到 tools 层来重造一遍 update()。
        """
        if not isinstance(value, dict):
            # self 表从头到尾都是整包 JSON dict；传别的类型是调用方写错了代码。
            # 按兜底哲学，"程序自己的错误"不许静默吞掉——喊一声并拒绝
            print(f"[kv] write('self') 被拒：value 应为 dict，实到 {type(value).__name__}")
            return False
        try:
            current = self._self.load(session_id) or {}
        except Exception as exc:
            print(f"[kv] write('self') 被拒：读不出当前身份（{exc}），无法确认冻结范围")
            return False

        def _empty(v) -> bool:
            return v is None or v == "" or v == [] or v == {}

        conflicts = [
            k for k, v in value.items()
            if not _empty(current.get(k)) and current.get(k) != v
        ]
        if conflicts:
            print(
                f"[kv] write('self', {session_id}) 整包被拒：试图覆盖已冻结的 {conflicts}。"
                f"她的身份第一次说出口即冻结；确有需要请走 update() 闭包那条路。"
            )
            return False
        # 只填空键：current 里已有的（含同值的）一律保留原样，value 没提到的键也原样保留
        # ——write() 绝不允许把已冻结的内容删掉或清空
        merged = dict(current)
        for k, v in value.items():
            if _empty(merged.get(k)):
                merged[k] = v
        self._self.save(session_id, merged)
        return True

    def update(self, store: str, key: str, fn):
        """读-改-写原子闭包（F3）：对 profile / portrait / relationship / self 四张
        整包 KV 表开放（self 是批次0 收进来的，见 _KV_UPDATE_TABLES 上的注释）。

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

    def commit_turn(self, **kwargs) -> dict:
        """R15c：一轮提交的事务门面（参数见 sqlite_store.commit_turn）。

        调用方注入 relation_fn（R15b 纯计算）——本门面只透传，人格计算
        不进 data 层。返回回执 dict（committed/already_committed/conflict/
        processing/failed）。"""
        return get_db().commit_turn(**kwargs)

    def get_commit_receipt(self, session_id: str, request_id: str):
        """按 (session, request) 查已提交回执；没有返回 None。"""
        return get_db().get_commit_receipt(session_id, request_id)

    def chats_between(self, session_id: str, start_iso: str, end_iso: str) -> list:
        """按时间段捞聊天记录（含头不含尾），写日记要用某一天的完整对话就靠它。"""
        return self._session.get_chats_between(session_id, start_iso, end_iso)

    def last_chat_per_session(self) -> list:
        """每个会话最后一次聊天的时间，闲置检测器点名用。"""
        return self._session.last_chat_per_session()

    def proactive_after(self, session_id: str, after_iso: str, n: int = 5) -> list:
        """某时刻之后她主动发的话（N3 轮询），旧 -> 新。"""
        return self._session.proactive_after(session_id, after_iso, n)


    def chat_log_hour_distribution(self, session_id: str, days: int = 14) -> dict:
        """近 N 天 chat_log 的小时分布 {hour: count}；拿不到返回 {}。

        K7 修的接口：以前 capability/proactive.py 自己 from data.sqlite_store import get_db
        再摸 _lock/_conn，违反分层纪律。现在 SQL 收在 SessionStore 里，capability 只调门面。
        """
        return self._session.chat_log_hour_distribution(session_id, days)


    def recent_chat(self, session_id: str, n: int = 100, learnable: bool = False) -> list:
        """取某会话最近 n 条聊天记录。

        read("session", key) 的条数**跟着 `memory.max_turns_short_term` 走**（默认 12；
        以前是写死的 20，那才是这条 docstring 原来描述的毛病，早已解耦），它表达的是
        "她醒来时记得多少"，而且传不进 n。api.py 拉历史要的是另一个条数（默认 100），
        语义不同，所以走这个专门的口子，不去动 read() 的含义。

        R14c：learnable=True 只取可学习的已提交正常轮——身份/画像/事实提炼的
        学习素材一律走这个口子；降级轮的兜底正文不许变成人格证据。旧记录分类
        字段为空的 legacy 行按可学习对待（不假装能自动辨认旧污染）。
        """
        return self._session.get_recent(session_id, n, learnable=learnable)

    # ---- J9：磁盘只增不减 —— 保留期修剪口子 ----
    #
    # 这三个方法**只提供能力，不自己挂到任何巡检上**：什么时候清、清多狠是
    # capability/proactive.py（巡检）的决定，tools 层不该替它拍板。接线前它们
    # 就是三个没人调的口子，这是有意的——比"顺手挂上去结果把她的聊天记录删了"
    # 安全得多。
    #
    # 共同纪律：keep_days <= 0 一律拒绝执行并返回 0。0/负数会让 cutoff 落到
    # 未来，等于"全删"——那是配置写错，不是修剪。兜底哲学：宁可什么都不做，
    # 也不要做不可逆的事。

    def prune_chat_log(self, keep_days: int = 365) -> int:
        """删掉早于 keep_days 的 chat_log 行，返回删掉的条数（J9）。

        为什么需要：chat_log 一句话一行、只追加不删，24/7 跑上一年就是几万行；
        写日记要用的 get_chats_between 和巡检要用的 chat_log_hour_distribution
        都在这张表上做区间扫描。长期记忆已经蒸馏进 Chroma、日记另有集合，
        原始逐字记录留一个窗口就够。

        SQL 正主在 data/sqlite_store.py 的 `SQLiteStorage.prune_chat_log`
        （它旁边还有账本用的 `prune_ledger`）——tools 层不再自己抄一份：
        两份实现在"删了多少"上漂移是最难查的那种不一致。本方法早先确实在这里
        直摸过 `get_db()._lock/_conn`（当时 data 层还没有删除口子），现在收回一行委托。

        两处有意的取舍，都不是疏忽：
        1. `keep_days <= 0` 的拒绝闸**留在门面层**。data 层的做法是"钳到 1 天"，
           可 cutoff = now - 1 天几乎照样等于清空整张表——接线方把配置读成 0 时，
           正确行为是一条都别删，而不是"少删一点"。删除不可逆，宁可什么都不做。
        2. 门面默认 365，比 data 层自己的 90 更保守。真消费方回看确实都不长
           （写日记取当天、小时分布取 14 天、短期记忆取十来条），90 天余量够；
           但门面是 capability 层直接看见的那个口子，一次**漏传参数**的误调用
           就该尽量少删她的逐字历史。接线时请一律显式传配置值，别吃默认。
        """
        if keep_days <= 0:
            print(f"[kv] prune_chat_log 拒绝执行：keep_days={keep_days} 会把整张表清空")
            return 0
        from data.sqlite_store import get_db

        try:
            # get_db() 也放在 try 里：库文件坏了/盘满了的时候它自己就会抛，
            # 修剪失败必须退化成"一条没删 + 喊一声"，不能让巡检线程跟着炸
            return get_db().prune_chat_log(keep_days)
        except Exception as exc:
            # 清不掉不影响任何人聊天，但必须留痕：静默失败就是"磁盘一直在涨"
            print(f"[kv] chat_log 修剪失败（保持原样，一条没删）：{exc}")
            return 0

    def prune_ledger(self, keep_days: int = 365) -> int:
        """删掉早于 keep_days 的关系账本（affection_history），返回删掉的条数（J9）。

        SQL 正主在 data/sqlite_store.py 的 `SQLiteStorage.prune_ledger`——它早就有了，
        但 capability 层按分层纪律够不着 data 层，缺这一行委托就等于账本永远只增不减。
        账本是审计面不是数据面：亲密度/信任的当前值在 relationship 那一行里，
        删旧账只丢掉"很久以前那次为什么变"的可追溯性，所以默认留一整年。
        """
        if keep_days <= 0:
            print(f"[kv] prune_ledger 拒绝执行：keep_days={keep_days} 会把整张账本清空")
            return 0
        from data.sqlite_store import get_db

        try:
            return get_db().prune_ledger(keep_days)
        except Exception as exc:
            print(f"[kv] 关系账本修剪失败（保持原样，一条没删）：{exc}")
            return 0

    def prune_candidates(self, days: int = 14) -> int:
        """放弃入池超过 N 天还没晋升的记忆候选，返回清理条数（E1）。

        参数名跟 data 层一致叫 `days`（不叫邻居那个 keep_days）：语义是反的——
        这里是"超过 N 天没凑够证据就**放弃**"，不是"保留 N 天"。

        这一行是**承重的**：capability/proactive.py 的 `_prune_candidates_once`
        用 getattr 探测本方法，探不到就静默跳过。没有它，"候选池 14 天过期"
        这条写进了约束文档的机制就一次也没跑过——池子无界增长，几个月前的
        一条幻觉只要重现一次就能凑满晋升计数。
        """
        if days <= 0:
            print(f"[kv] prune_candidates 拒绝执行：days={days} 会把整个候选池清空")
            return 0
        from data.sqlite_store import get_db

        try:
            return get_db().prune_candidates(days)
        except Exception as exc:
            print(f"[kv] 记忆候选池清理失败（保持原样，一条没删）：{exc}")
            return 0

    def prune_uploads(self, keep_days: int = 90) -> int:
        """清掉 `DATA_DIR/uploads` 下过期的用户上传文件，返回删掉的个数（J9）。

        用户上传的图/文件只在当轮对话里被模型看（api.py 转 base64 内嵌），
        之后留着只是为了前端历史里还能显示。默认 90 天。
        """
        if keep_days <= 0:
            print(f"[kv] prune_uploads 拒绝执行：keep_days={keep_days} 会清空整个上传目录")
            return 0
        return _prune_dir_by_mtime(
            get_settings().data_dir / "uploads", keep_days, "uploads"
        )

    def prune_image_assets(self, keep_days: int = 180) -> int:
        """清掉 `DATA_DIR/images` 下过期的生成图，并同步 index.json，返回删掉的张数（J9）。

        图片是磁盘最大头（1-2MB/张）。**只删文件不够**：ImageAssetStore 的
        index.json 里会留下指向已不存在路径的条目，read("image") 照样把它们吐出去，
        前端拿到就是一堆 404——所以文件和索引必须一起清，顺序是先删文件、
        再按"文件还在不在"过滤索引（这样连历史遗留的孤儿条目也一并收掉）。

        index.json 自己绝不能被当成"过期文件"删掉：它没被重写过就说明很久没出图，
        mtime 正好会老过阈值——所以传给目录清理助手时显式跳过它。
        """
        if keep_days <= 0:
            print(f"[kv] prune_image_assets 拒绝执行：keep_days={keep_days} 会清空整个图库")
            return 0
        image_dir = get_settings().data_dir / "images"
        removed = _prune_dir_by_mtime(image_dir, keep_days, "images", skip_names=("index.json",))
        # 索引同步：只留文件还真在的条目
        try:
            index_path = self._image._index()  # noqa: SLF001 — 用 store 自己的路径口子，别在这儿抄一遍字面量（抄了就会在 data 层改路径时静默失配）
            assets = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
            if not isinstance(assets, list):
                raise ValueError(f"index.json 不是数组，是 {type(assets).__name__}")
            kept = []
            for a in assets:
                if not isinstance(a, dict):
                    continue  # 脏条目：顺手收掉，别让它一直躺在索引里
                p = (a.get("image_path") or "").strip()
                if p and Path(p).exists():
                    kept.append(a)
            if len(kept) != len(assets):
                index_path.write_text(
                    json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[kv] 图库索引同步：{len(assets)} -> {len(kept)} 条（删掉了指向已清理文件的条目）")
        except Exception as exc:
            # 索引同步失败不能让"文件已经删了"这件事变得不可见：喊一声，
            # 顶多是前端看到几条 404，下次修剪会再试一遍
            print(f"[kv] 图库索引同步失败（文件已清理，索引里可能残留失效条目）：{exc}")
        return removed




def _log1p_safe(count) -> float:
    """log1p 的安全包装：热度字段坏了退 0（不加成，不炸排序）。"""
    try:
        import math

        return math.log1p(max(0, int(count or 0)))
    except (TypeError, ValueError):
        return 0.0


def _prune_dir_by_mtime(directory, keep_days: int, what: str, skip_names=()) -> int:
    """删掉 directory 下 mtime 老于 keep_days 的**文件**，返回删掉的个数（J9）。

    为什么按 mtime 判老，而不是去解析文件名或索引里的时间字段：这两个目录的
    命名规则完全不同——images 是 external.py 生成的 `%Y%m%d_%H%M%S_xxxxxx.png`，
    uploads 是 api.py 落盘的用户原文件名（可能是 `IMG_2049.jpeg`，可能带中文和空格），
    而 index.json 的 created_at 是另一套 ISO 串。三套格式各写一个解析器，
    每一个都是"格式一变就静默失配"的雷。mtime 是文件系统免费给的、
    格式无关的"这东西最后一次真正动过是什么时候"，一套逻辑覆盖所有情况。

    三条纪律：
    - **只删文件、不碰子目录**（也不递归进去）：目录名撞上了阈值就把整棵子树删掉，
      那是不可逆事故；
    - skip_names 里的名字一律跳过——`images/index.json` 就靠它保命：很久没出图时
      它的 mtime 正好会老过阈值，被当成"过期图片"删掉就等于把图库索引整个丢了；
    - 单个文件删不掉（被占用/权限）就跳过继续，不让一个文件把整轮修剪打断。
      失败会喊一声：静默跳过 = 磁盘继续涨而没人知道。
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0  # 目录还没被创建过（从没上传/从没出图），不是错误
    cutoff = datetime.now().timestamp() - keep_days * 86400
    removed = 0
    for entry in directory.iterdir():
        if entry.name in skip_names or not entry.is_file():
            continue
        try:
            if entry.stat().st_mtime >= cutoff:
                continue  # 还在保留期内
            entry.unlink()
            removed += 1
        except OSError as exc:
            print(f"[kv] {what} 清理：{entry.name} 删不掉，跳过（{exc}）")
    if removed:
        print(f"[kv] {what} 清理：删掉 {removed} 个超过 {keep_days} 天的文件")
    return removed

