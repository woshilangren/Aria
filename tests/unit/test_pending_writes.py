"""R19a：持久待办队列（pending_writes）数据与门面契约。

验收（《计划与设计.md》R19a）：
- 表结构与字段齐全（迁移 v4）；
- task_id 主键防重复登记：重复 enqueue 幂等跳过，不覆盖旧载荷；
- 人工重新入队：当前重试预算清零、累计审计（attempts_total/last_error）保留；
- 任务原文受本地数据保护：payload 只走这张表，门面/日志不外带全文。
"""

from __future__ import annotations

import pytest

from data.sqlite_store import get_db
from tools.storage import KVStoreTool


@pytest.fixture()
def kv():
    return KVStoreTool()


def _tables(db):
    return {r[0] for r in db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def test_fresh_db_has_pending_writes_table():
    db = get_db()
    assert "pending_writes" in _tables(db)
    cols = {r[1] for r in db._conn.execute("PRAGMA table_info(pending_writes)")}
    # R19a 字段清单：至少这些
    need = {"task_id", "kind", "source_turn_id", "payload_version", "payload",
            "status", "attempts", "next_attempt_at", "lease_until",
            "last_error", "created_at", "updated_at"}
    assert need <= cols, f"缺字段: {need - cols}"
    assert "attempts_total" in cols, "人工重试的累计审计列必须有"


def test_enqueue_is_idempotent_on_task_id(kv):
    """task_id 主键防重复：同 ID 重复登记幂等跳过，且不覆盖旧载荷。"""
    assert kv.enqueue_pending("pw-1", "vector_memory",
                              {"memory_id": "m1", "content": "第一版"}) is True
    # 同 ID 再登记（比如重放）：False，旧载荷原封不动
    assert kv.enqueue_pending("pw-1", "vector_memory",
                              {"memory_id": "m1", "content": "被篡改的版本"}) is False
    row = kv.get_pending("pw-1")
    assert row["payload"]["content"] == "第一版"
    assert row["status"] == "pending" and row["attempts"] == 0


def test_enqueue_requires_ids(kv):
    from pytest import raises

    with raises(ValueError):
        kv.enqueue_pending("", "vector_memory", {})
    with raises(ValueError):
        kv.enqueue_pending("pw-x", "", {})


def test_requeue_keeps_audit_and_resets_budget(kv):
    """人工重新入队：预算清零重计，累计审计（attempts_total/last_error）保留。"""
    kv.enqueue_pending("pw-2", "vector_memory", {"memory_id": "m2"}, source_turn_id="t1")
    db = get_db()
    # 模拟已经失败过几次
    db._conn.execute(
        "UPDATE pending_writes SET attempts=3, attempts_total=3, "
        "last_error='upsert failed', status='exhausted' WHERE task_id='pw-2'")
    db._conn.commit()
    assert kv.requeue_pending("pw-2") is True
    row = kv.get_pending("pw-2")
    assert row["status"] == "pending"
    assert row["attempts"] == 0, "新预算从零开始"
    assert row["attempts_total"] == 3, "累计审计不许清零"
    assert row["last_error"] == "upsert failed", "旧失败原因保留供审计"


def test_requeue_missing_task_returns_false(kv):
    assert kv.requeue_pending("不存在的任务") is False


def test_get_pending_missing_returns_none(kv):
    assert kv.get_pending("没有这条") is None
