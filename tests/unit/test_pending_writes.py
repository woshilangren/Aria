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

# ------------------- R19b：生产端持久化 -------------------

def test_commit_turn_registers_pending_tasks_atomically(kv):
    """R19b：预先可知的后台任务随 commit_turn 同事务登记——

    - 提交成功：任务行在册；
    - 提交回滚（账本参数坏）：任务行一并消失，无"提交没了任务还在"的孤儿。
    """
    from shared.types import PreparedTurn

    db = get_db()
    turn = PreparedTurn(
        session_id="pw-t1", turn_id="turn-pw-1", request_id="req-pw-1",
        request_digest="d", user_text="hi", assistant_text="你好",
    )
    rc = db.commit_turn(
        session_id=turn.session_id, turn_id=turn.turn_id,
        request_id=turn.request_id, request_digest=turn.request_digest,
        user_text=turn.user_text, assistant_text=turn.assistant_text,
        pending_tasks=[{"task_id": "turn-pw-1:identity_freeze",
                        "kind": "identity_freeze", "source_turn_id": turn.turn_id,
                        "payload": {"session_id": turn.session_id}}],
    )
    assert rc["status"] == "committed"
    row = kv.get_pending("turn-pw-1:identity_freeze")
    assert row and row["kind"] == "identity_freeze" and row["status"] == "pending"

    # 回滚路径：账本参数非法 → 整体回滚，任务登记一并消失
    turn2 = PreparedTurn(
        session_id="pw-t1", turn_id="turn-pw-2", request_id="req-pw-2",
        request_digest="d2", user_text="hi", assistant_text="你好",
    )
    rc2 = db.commit_turn(
        session_id=turn2.session_id, turn_id=turn2.turn_id,
        request_id=turn2.request_id, request_digest=turn2.request_digest,
        user_text=turn2.user_text, assistant_text=turn2.assistant_text,
        ledger={"field": "intimacy", "old": "不是数字", "new": 21},
        pending_tasks=[{"task_id": "turn-pw-2:identity_freeze",
                        "kind": "identity_freeze", "source_turn_id": turn2.turn_id,
                        "payload": {}}],
    )
    assert rc2["status"] == "failed"
    assert kv.get_pending("turn-pw-2:identity_freeze") is None,         "回滚后不许留下任务登记（孤儿）"


def test_guarded_success_marks_pending_done(kv, register_services):
    """R19b：领取守卫执行成功 → 同 ID 的 pending 行标 done，消费者不再重试。"""
    from orchestration import writeback as wb
    from shared.types import DeferredTask

    register_services(kv_store=kv)
    kv.enqueue_pending("turn-x:portrait", "portrait", {}, source_turn_id="turn-x")
    ran = []
    run = wb.schedule_commit_tasks(
        {"status": "committed", "turn_id": "turn-x"},
        [DeferredTask(kind=wb.TASK_PORTRAIT, fn=lambda: ran.append(1))],
    )
    run[0]()
    assert ran == [1]
    row = kv.get_pending("turn-x:portrait")
    assert row["status"] == "done", f"执行成功必须标 done，实测 {row}"


def test_remember_note_failure_enqueues_full_item(kv, register_services):
    """R19b：向量库挂 → 完整 MemoryItem 载荷落队，稳定 task_id=vm-{memory_id}。"""
    from capability.memory import remember_note

    class _DownVS:
        def upsert_memory(self, item):
            raise RuntimeError("vector down")

    register_services(kv_store=kv, vector_store=_DownVS())
    remember_note("pw-t2", "她提到喜欢夜跑", kind="event", importance=3,
                  memory_id="fixed-mem-1")
    row = kv.get_pending("vm-fixed-mem-1")
    assert row and row["status"] == "pending"
    assert row["payload"]["memory_id"] == "fixed-mem-1"
    assert row["payload"]["content"] == "她提到喜欢夜跑", "完整可重建字段必须随载荷"
    # 重试同一条（同 memory_id）：幂等，不产生第二行
    remember_note("pw-t2", "她提到喜欢夜跑", kind="event", importance=3,
                  memory_id="fixed-mem-1")
    assert kv.get_pending("vm-fixed-mem-1") is not None
    assert kv.get_pending("vm-fixed-mem-1")["created_at"] == row["created_at"]


def test_summarize_failure_enqueues_full_snapshot(kv, register_services):
    """R19b：摘要失败落**完整原文快照**——不能只存 head[:200]。"""
    from capability.memory import SessionMemoryKeeper

    class _DownLLM:
        def chat(self, messages, **kwargs):
            raise RuntimeError("llm down")

    register_services(kv_store=kv, llm=_DownLLM())
    keeper = SessionMemoryKeeper()
    old_part = [{"role": "user", "content": "第一句老话" * 10},
                {"role": "assistant", "content": "第一句回复" * 10}]
    assert keeper._summarize("pw-t3", old_part) is False
    rows = get_db()._conn.execute(
        "SELECT task_id, payload FROM pending_writes WHERE kind='session_summarize'"
    ).fetchall()
    assert rows, "摘要失败必须落队"
    import json as _json

    payload = _json.loads(rows[0][1])
    assert payload["msgs"][0]["content"] == "第一句老话" * 10,         "原文必须完整快照，不许截断"
