"""R01a：向量迁移的数组处理 + 真实 Chroma 两条数据迁移回归（本地假向量，不触网）。

《计划与设计.md》R01a：`_ensure_collection` 复制阶段对 embeddings 做 `or []`，
而 Chroma（1.x）`get(include=["embeddings"])` 返回 **ndarray**——对它做真值判断
触发 "truth value of an array is ambiguous"，且崩点在**旧集合已删之后**：
主集合 0 条、`__migrating` 里还有全量数据，下次启动因 metadata 匹配直接接受
空集合。本文件用真实 Chroma + 本地假向量锁死两件事：

1. embeddings 的三种形态（list / ndarray / 空、None）都必须能处理；
2. 换模型触发的迁移：两条数据迁移后 ID / 正文 / 元数据 / 向量维度全部一致。

不触网：embedder 用本地替身（返回固定维度假向量），Chroma 本地持久化
（conftest 的 DATA_DIR 指向 tmp_path）。
"""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import get_settings
from tools.storage import _DISTILLED_COLLECTION, _norm_embeddings, VectorStoreTool


# --------------------------------------------------------------------------
# 1. 数组处理原语：三种形态都不能抛（ndarray 真值异常的回归闸）
# --------------------------------------------------------------------------

def test_norm_embeddings_handles_none_and_empty():
    assert _norm_embeddings(None) == []
    assert _norm_embeddings([]) == []


def test_norm_embeddings_handles_plain_list():
    lst = [[0.1, 0.2], [0.3, 0.4]]
    out = _norm_embeddings(lst)
    assert len(out) == 2
    assert list(out[0]) == [0.1, 0.2]


def test_norm_embeddings_handles_ndarray_without_truthiness_error():
    """对 ndarray 做 `or []` 会抛 ValueError——这就是迁移中途炸掉的原缺陷。"""
    arr = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    out = _norm_embeddings(arr)
    assert len(out) == 3
    assert len(out[0]) == 2


# --------------------------------------------------------------------------
# 2. 真实 Chroma 迁移回归：两条数据，迁移后字段全一致
# --------------------------------------------------------------------------

class _FakeEmbedder:
    """本地假 embedder：不触网，返回固定维度的假向量，并记录调用。"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.calls: list = []

    def embed_documents(self, docs):
        self.calls.append(list(docs))
        return [[0.25] * self.dimension for _ in docs]


@pytest.fixture()
def tool():
    """真 Chroma（tmp_path 持久化）+ 假 embedder 的 VectorStoreTool。"""
    t = VectorStoreTool()
    t._embedder = _FakeEmbedder(get_settings().embedding_dimension)
    return t


def test_migration_two_rows_survive_with_intact_fields(tool):
    """换模型触发迁移：2 条数据迁完后 ID/正文/元数据/向量维度必须原样。

    修复前这条会在复制阶段炸（ndarray 真值异常），留下"主集合空 +
    __migrating 有全量数据"的半迁移现场——正是 R01a 要消灭的缺陷。
    """
    cfg = get_settings()
    dim = cfg.embedding_dimension

    # 第一次：集合不存在 → 按 cfg 元数据新建，直接返回（无迁移）
    col = tool._ensure_collection(_DISTILLED_COLLECTION, cfg)
    col.upsert(
        ids=["m1", "m2"],
        embeddings=[[0.1] * dim, [0.2] * dim],
        documents=["第一条长期记忆", "第二条长期记忆"],
        metadatas=[{"kind": "event", "session_id": "s1"}, {"kind": "diary", "session_id": "s2"}],
    )

    # 模拟"换过模型"：把集合 metadata 的模型名改掉 → 下次 ensure 触发迁移
    col.modify(metadata={"embedding_model": "old-model", "dimension": dim})

    # 迁移：读旧原文 → 假 embedder 重嵌 → 删主集合 → 从 __migrating 拷回
    final = tool._ensure_collection(_DISTILLED_COLLECTION, cfg)

    got = final.get(include=["embeddings", "documents", "metadatas"])
    assert sorted(got["ids"]) == ["m1", "m2"], f"迁移后必须还是两条，实测 {got['ids']}"
    assert sorted(got["documents"]) == ["第一条长期记忆", "第二条长期记忆"]
    for vec in got["embeddings"]:
        assert len(vec) == dim, "向量维度必须与配置一致"
    metas = dict(zip(got["ids"], got["metadatas"]))
    assert metas["m1"]["kind"] == "event" and metas["m1"]["session_id"] == "s1"
    assert metas["m2"]["kind"] == "diary" and metas["m2"]["session_id"] == "s2"
    # 迁移确实发生了：重嵌调用过假 embedder（不是 metadata 碰巧匹配直接返回）
    assert tool._embedder.calls, "必须真的走过重嵌路径"
    # 元数据已按新配置落回
    assert final.metadata.get("embedding_model") == cfg.embedding_model


def test_migration_empty_collection_rebuilds_clean(tool):
    """空集合（只有旧元数据没原文）无需迁移，直接按新维度重开。"""
    cfg = get_settings()
    dim = cfg.embedding_dimension
    col = tool._ensure_collection(_DISTILLED_COLLECTION, cfg)
    col.modify(metadata={"embedding_model": "old-model", "dimension": dim})
    final = tool._ensure_collection(_DISTILLED_COLLECTION, cfg)
    assert final.count() == 0
    assert final.metadata.get("embedding_model") == cfg.embedding_model


# --------------------------------------------------------------------------
# 3. R01b：可恢复切换——六处故障注入 + 遗留事故 + 状态文件损坏
# --------------------------------------------------------------------------

import tools.storage as storage_mod  # noqa: E402
from tools.storage import _DIARY_COLLECTION, _MIGRATION_STATE_FILE  # noqa: E402


def _seed_old_collection(tool, name, n, cfg):
    """造一个"旧模型"集合：n 条数据，metadata 标成旧模型 → 下次 ensure 触发迁移。"""
    dim = cfg.embedding_dimension
    col = tool._chroma.get_collection(
        name, metadata={"embedding_model": cfg.embedding_model, "dimension": dim}
    )
    col.upsert(
        ids=[f"id{i}" for i in range(n)],
        embeddings=[[0.1 * (i + 1)] * dim for i in range(n)],
        documents=[f"文档{i}" for i in range(n)],
        metadatas=[{"idx": i} for i in range(n)],
    )
    col.modify(metadata={"embedding_model": "old-model", "dimension": dim})
    return col


class _FailingEmbedder(_FakeEmbedder):
    """第 N 次调用起必炸的 embedder（模拟重嵌入中途失败）。"""

    def __init__(self, dimension, fail_on_call):
        super().__init__(dimension)
        self._fail_on = fail_on_call
        self._calls = 0

    def embed_documents(self, docs):
        self._calls += 1
        if self._calls >= self._fail_on:
            raise RuntimeError("注入：重嵌入失败")
        return super().embed_documents(docs)


class _UpsertFailProxy:
    """包一层集合：第 fail_after 次 upsert 之后必炸（模拟临时/正式集合写一半）。"""

    def __init__(self, col, fail_after):
        self._col = col
        self._fail_after = fail_after
        self._n = 0

    def upsert(self, **kw):
        self._n += 1
        if self._n > self._fail_after:
            raise RuntimeError("注入：写入中途故障")
        return self._col.upsert(**kw)

    def __getattr__(self, item):
        return getattr(self._col, item)


def _restart(tool):
    """模拟进程重启：全新 VectorStoreTool 实例（状态文件从磁盘重读）。"""
    t = VectorStoreTool()
    t._embedder = _FakeEmbedder(get_settings().embedding_dimension)
    return t


def _assert_migrated_clean(t, name, expected_ids, cfg):
    final = t._chroma.client.get_collection(name)
    assert sorted(final.get()["ids"]) == sorted(expected_ids)
    assert final.metadata.get("embedding_model") == cfg.embedding_model
    # 临时集合必须已收尾
    with pytest.raises(Exception):
        t._chroma.client.get_collection(f"{name}__migrating")
    # 状态记录必须清干净
    assert name not in t._migration_state


def test_reembed_failure_main_intact_then_restart_recovers(tool, monkeypatch):
    """断点①重嵌入失败：旧主集合完好；重启后清场重走迁移，数据一条不丢。"""
    cfg = get_settings()
    n = 3
    _seed_old_collection(tool, _DISTILLED_COLLECTION, n, cfg)
    tool._embedder = _FailingEmbedder(cfg.embedding_dimension, fail_on_call=1)
    with pytest.raises(RuntimeError, match="重嵌入失败"):
        tool._ensure_collection(_DISTILLED_COLLECTION, cfg)
    # 现场：旧主集合原样（还在旧 metadata 下、数据全在）
    old = tool._chroma.client.get_collection(_DISTILLED_COLLECTION)
    assert old.count() == n
    assert old.metadata.get("embedding_model") == "old-model"

    t2 = _restart(tool)
    final = t2._ensure_collection(_DISTILLED_COLLECTION, cfg)
    _assert_migrated_clean(t2, _DISTILLED_COLLECTION, [f"id{i}" for i in range(n)], cfg)
    assert final.count() == n


def test_tmp_half_written_then_restart_recovers(tool, monkeypatch):
    """断点②临时集合写一半（多批次中途炸）：旧主集合完好；重启重走迁移。"""
    cfg = get_settings()
    n = 5
    _seed_old_collection(tool, _DISTILLED_COLLECTION, n, cfg)
    monkeypatch.setattr(storage_mod, "EMBED_BATCH_SIZE", 2)  # 5 条 → 3 批
    # 临时集合第 2 批写完后炸
    orig_get = tool._chroma.get_collection

    def wrapped_get(name, metadata=None):
        col = orig_get(name, metadata=metadata)
        if name == f"{_DISTILLED_COLLECTION}__migrating":
            return _UpsertFailProxy(col, fail_after=2)
        return col

    monkeypatch.setattr(tool._chroma, "get_collection", wrapped_get)
    with pytest.raises(RuntimeError, match="写入中途故障"):
        tool._ensure_collection(_DISTILLED_COLLECTION, cfg)
    # 现场：临时集合 4 条（写一半）、主集合旧数据完好
    tmp = tool._chroma.client.get_collection(f"{_DISTILLED_COLLECTION}__migrating")
    assert tmp.count() == 4

    t2 = _restart(tool)
    t2._ensure_collection(_DISTILLED_COLLECTION, cfg)
    _assert_migrated_clean(t2, _DISTILLED_COLLECTION, [f"id{i}" for i in range(n)], cfg)


def test_fail_after_main_delete_recovers_from_tmp(tool, monkeypatch):
    """断点③删除主集合之后崩：重启从 __migrating 整包恢复（主集合缺失分支）。"""
    cfg = get_settings()
    n = 3
    _seed_old_collection(tool, _DISTILLED_COLLECTION, n, cfg)
    orig_del = tool._chroma.delete_collection

    def wrapped_del(name):
        orig_del(name)  # 真删完才炸——现场 = 主集合没了、tmp 完整、状态 copying
        if name == _DISTILLED_COLLECTION:
            raise RuntimeError("注入：删除主集合后故障")

    monkeypatch.setattr(tool._chroma, "delete_collection", wrapped_del)
    with pytest.raises(RuntimeError, match="删除主集合后故障"):
        tool._ensure_collection(_DISTILLED_COLLECTION, cfg)

    t2 = _restart(tool)
    t2._ensure_collection(_DISTILLED_COLLECTION, cfg)
    _assert_migrated_clean(t2, _DISTILLED_COLLECTION, [f"id{i}" for i in range(n)], cfg)


def test_fail_mid_copy_overwrites_full(tool, monkeypatch):
    """断点④/⑤重建主集合后复制一半崩：重启整包覆盖拷回，不许重不完整集合。"""
    cfg = get_settings()
    n = 5
    _seed_old_collection(tool, _DISTILLED_COLLECTION, n, cfg)
    monkeypatch.setattr(storage_mod, "EMBED_BATCH_SIZE", 2)
    orig_get = tool._chroma.get_collection
    orig_del = tool._chroma.delete_collection
    flag = {"main_deleted": False}

    def wrapped_del(name):
        orig_del(name)
        if name == _DISTILLED_COLLECTION:
            flag["main_deleted"] = True

    def wrapped_get(name, metadata=None):
        if flag["main_deleted"] and name == _DISTILLED_COLLECTION:
            col = orig_get(name, metadata=metadata)
            return _UpsertFailProxy(col, fail_after=2)  # 两批成功（4 条）、第 3 批炸
        return orig_get(name, metadata=metadata)

    monkeypatch.setattr(tool._chroma, "delete_collection", wrapped_del)
    monkeypatch.setattr(tool._chroma, "get_collection", wrapped_get)
    with pytest.raises(RuntimeError, match="写入中途故障"):
        tool._ensure_collection(_DISTILLED_COLLECTION, cfg)
    # 现场：主集合只有前 4 条（半拷）、tmp 完整 5 条
    main = tool._chroma.client.get_collection(_DISTILLED_COLLECTION)
    assert main.count() == 4

    t2 = _restart(tool)
    t2._ensure_collection(_DISTILLED_COLLECTION, cfg)
    _assert_migrated_clean(t2, _DISTILLED_COLLECTION, [f"id{i}" for i in range(n)], cfg)


def test_fail_before_tmp_cleanup_finishes_on_restart(tool, monkeypatch):
    """断点⑥收尾前崩（正式集合已验证完整、只差删临时）：重启直接收尾。"""
    cfg = get_settings()
    n = 3
    _seed_old_collection(tool, _DISTILLED_COLLECTION, n, cfg)
    orig_del = tool._chroma.delete_collection

    def wrapped_del(name):
        if name == f"{_DISTILLED_COLLECTION}__migrating":
            raise RuntimeError("注入：删临时集合前故障")  # 不真删，留在现场
        orig_del(name)

    monkeypatch.setattr(tool._chroma, "delete_collection", wrapped_del)
    with pytest.raises(RuntimeError, match="删临时集合前故障"):
        tool._ensure_collection(_DISTILLED_COLLECTION, cfg)
    # 现场：主集合已完整（新 metadata）、tmp 仍在
    main = tool._chroma.client.get_collection(_DISTILLED_COLLECTION)
    assert main.count() == n

    t2 = _restart(tool)
    final = t2._ensure_collection(_DISTILLED_COLLECTION, cfg)
    assert final.count() == n
    with pytest.raises(Exception):
        t2._chroma.client.get_collection(f"{_DISTILLED_COLLECTION}__migrating")
    assert _DISTILLED_COLLECTION not in t2._migration_state


@pytest.mark.parametrize("name", [_DISTILLED_COLLECTION, _DIARY_COLLECTION])
def test_legacy_half_migration_recovered(tool, name):
    """无状态文件的遗留事故（升级前现场）：主集合 metadata 已是目标但空、
    __migrating 有货 → 必须恢复，不能接受空集合伪装健康。memories/diaries 都覆盖。"""
    cfg = get_settings()
    dim = cfg.embedding_dimension
    # 手工搭现场：主集合 = 新 metadata + 0 条；tmp = 2 条完整数据
    tool._chroma.get_collection(name, metadata={"embedding_model": cfg.embedding_model,
                                               "dimension": dim})
    tmp = tool._chroma.get_collection(f"{name}__migrating",
                                      metadata={"embedding_model": cfg.embedding_model,
                                                "dimension": dim})
    tmp.upsert(
        ids=["x1", "x2"],
        embeddings=[[0.3] * dim, [0.4] * dim],
        documents=["遗留记忆一", "遗留记忆二"],
        metadatas=[{"kind": "event"}, {"kind": "event"}],
    )
    t2 = _restart(tool)
    final = t2._ensure_collection(name, cfg)
    got = final.get()
    assert sorted(got["ids"]) == ["x1", "x2"], f"遗留现场必须恢复，实测 {got['ids']}"
    assert sorted(got["documents"]) == ["遗留记忆一", "遗留记忆二"]
    with pytest.raises(Exception):
        t2._chroma.client.get_collection(f"{name}__migrating")


def test_corrupt_state_file_never_destroys_data(tool):
    """状态文件损坏：按无记录处理、不据此删任何数据，迁移照常完成。"""
    cfg = get_settings()
    n = 3
    _seed_old_collection(tool, _DISTILLED_COLLECTION, n, cfg)
    # 造半截现场 + 写坏状态文件
    tmp = tool._chroma.get_collection(
        f"{_DISTILLED_COLLECTION}__migrating",
        metadata={"embedding_model": cfg.embedding_model, "dimension": cfg.embedding_dimension},
    )
    tmp.upsert(ids=["id0"], embeddings=[[0.9] * cfg.embedding_dimension],
               documents=["文档0"], metadatas=[{"idx": 0}])
    tool._state_file.write_text("{这不是JSON", encoding="utf-8")

    t2 = _restart(tool)  # 加载到损坏状态 → 按无记录处理
    assert t2._migration_state == {}
    t2._ensure_collection(_DISTILLED_COLLECTION, cfg)
    _assert_migrated_clean(t2, _DISTILLED_COLLECTION, [f"id{i}" for i in range(n)], cfg)


def test_unrecoverable_scene_stops_collection_loudly(tool):
    """唯一副本已丢（主集合没了、tmp 也没了、状态说 copying）：明确停用并告警，
    绝不伪装成健康空集合。"""
    cfg = get_settings()
    # 手工写状态：copying 阶段、预期 3 条；现场无主集合、无临时集合
    tool._state_file.parent.mkdir(parents=True, exist_ok=True)
    tool._state_file.write_text(
        storage_mod.json.dumps(
            {_DISTILLED_COLLECTION: {
                "phase": "copying", "tmp_name": f"{_DISTILLED_COLLECTION}__migrating",
                "target": {"embedding_model": cfg.embedding_model,
                           "dimension": cfg.embedding_dimension},
                "expected_ids": ["a", "b", "c"],
                "updated_at": "2026-09-22T00:00:00",
            }}
        ),
        encoding="utf-8",
    )
    t2 = _restart(tool)
    with pytest.raises(RuntimeError, match="不可恢复"):
        t2._ensure_collection(_DISTILLED_COLLECTION, cfg)
    # 集合保持未建/空，绝不伪装健康
    with pytest.raises(Exception):
        t2._chroma.client.get_collection(_DISTILLED_COLLECTION)
