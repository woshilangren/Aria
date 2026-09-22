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
