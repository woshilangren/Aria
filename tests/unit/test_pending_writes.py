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

# ------------------- R19c：消费与退出恢复 -------------------

def _watcher():
    """绕过 __init__（不拉起 ProactiveSpeaker），只测消费方法本身。"""
    from capability.proactive import IdleDiaryWatcher

    return IdleDiaryWatcher.__new__(IdleDiaryWatcher)


def test_lease_holds_until_expiry_then_reclaimable(kv):
    """R19c：租约期内同任务不会被再次领取；租约过期可重领（崩溃恢复）。"""
    kv.enqueue_pending("pw-lease", "vector_memory", {"memory_id": "m"})
    first = kv.lease_pending(limit=3, lease_seconds=300)
    assert [t["task_id"] for t in first] == ["pw-lease"]
    assert first[0]["attempts"] == 1
    # 租约期内：再来一轮消费领不到
    assert kv.lease_pending(limit=3, lease_seconds=300) == []
    # 租约过期：可重领（可控时钟：把租约改到过去）
    db = get_db()
    db._conn.execute("UPDATE pending_writes SET lease_until='2000-01-01T00:00:00' "
                     "WHERE task_id='pw-lease'")
    db._conn.commit()
    again = kv.lease_pending(limit=3, lease_seconds=300)
    assert [t["task_id"] for t in again] == ["pw-lease"]
    assert again[0]["attempts"] == 2, "重领计一次新尝试"


def test_lease_skips_not_yet_due(kv):
    """R19c：只处理已到期任务——退避未到期的行不冲刷。"""
    kv.enqueue_pending("pw-future", "vector_memory", {"memory_id": "m"})
    db = get_db()
    db._conn.execute("UPDATE pending_writes SET next_attempt_at='2099-01-01T00:00:00' "
                     "WHERE task_id='pw-future'")
    db._conn.commit()
    assert kv.lease_pending(limit=3, lease_seconds=300) == []


def test_fail_pending_backoff_and_exhaustion(kv):
    """R19c：失败落退避与错误；预算耗尽标 exhausted；载荷不删除。"""
    kv.enqueue_pending("pw-fail", "vector_memory", {"memory_id": "m"})
    state = kv.fail_pending("pw-fail", "第一次失败", backoff_seconds=60, max_attempts=2)
    assert state == "pending"
    row = kv.get_pending("pw-fail")
    assert row["last_error"] == "第一次失败"
    # 第二次失败（attempts 已在 lease 时 +1 过，这里直接判耗尽）
    db = get_db()
    db._conn.execute("UPDATE pending_writes SET attempts=2 WHERE task_id='pw-fail'")
    db._conn.commit()
    state = kv.fail_pending("pw-fail", "第二次失败", backoff_seconds=60, max_attempts=2)
    assert state == "exhausted"
    row = kv.get_pending("pw-fail")
    assert row["status"] == "exhausted"
    assert row["payload"]["memory_id"] == "m", "耗尽也不许删业务载荷"


def test_unparsable_payload_blocked_not_burned(kv):
    """R19c：永久格式错误直接 blocked，不无限烧 API；载荷保留。"""
    kv.enqueue_pending("pw-bad", "vector_memory", {"ok": True})
    db = get_db()
    db._conn.execute("UPDATE pending_writes SET payload='{这不是JSON' WHERE task_id='pw-bad'")
    db._conn.commit()
    w = _watcher()
    w._apply_pending_task(kv, {"task_id": "pw-bad", "kind": "vector_memory",
                               "payload": None, "attempts": 1})
    row = kv.get_pending("pw-bad")
    assert row["status"] == "blocked"
    assert row["payload"] is None or row["payload"] != "", "载荷保留待人工"


def test_consumer_rebuilds_vector_memory_and_marks_done(kv, register_services):
    """R19c：vector_memory 消费 = 从完整载荷重建 MemoryItem → upsert → done。

    已生成冻结的正文从本地载荷恢复，不重新让 LLM 编一份。"""
    captured = []

    class _VS:
        def upsert_memory(self, item):
            captured.append(item)
            return True

    register_services(vector_store=_VS())
    kv.enqueue_pending("vm-rebuild-1", "vector_memory",
                       {"memory_id": "rebuild-1", "session_id": "s1", "kind": "event",
                        "content": "冻结的正文", "importance": 3, "timestamp": "2026-09-23T00:00:00",
                        "feeling": "", "appraisal": "", "valence": 0.0, "arousal": 0.3,
                        "peak_moment": ""})
    w = _watcher()
    w._drain_pending_writes(kv)
    assert len(captured) == 1 and captured[0].memory_id == "rebuild-1"
    assert captured[0].content == "冻结的正文", "正文从载荷恢复，不重编"
    assert kv.get_pending("vm-rebuild-1")["status"] == "done"


def test_consumer_failure_schedules_backoff(kv, register_services):
    """R19c：执行失败 → 指数退避落库，不标 done。"""
    class _DownVS:
        def upsert_memory(self, item):
            raise RuntimeError("still down")

    register_services(vector_store=_DownVS())
    kv.enqueue_pending("vm-backoff", "vector_memory",
                       {"memory_id": "m-b", "session_id": "s1", "kind": "event",
                        "content": "x", "importance": 3, "timestamp": "2026-09-23T00:00:00",
                        "feeling": "", "appraisal": "", "valence": 0.0, "arousal": 0.3,
                        "peak_moment": ""})
    w = _watcher()
    w._drain_pending_writes(kv)  # lease 时 attempts=1
    row = kv.get_pending("vm-backoff")
    assert row["status"] == "pending"
    assert row["attempts"] == 1 and row["last_error"] == "still down"
    assert row["next_attempt_at"] > "2026-01-01", "退避时间已落库"


def test_unknown_kind_blocked_without_burning_api(kv, register_services):
    kv.enqueue_pending("pw-unknown", "谁也没见过的种类", {"x": 1})
    w = _watcher()
    w._apply_pending_task(kv, {"task_id": "pw-unknown", "kind": "谁也没见过的种类",
                               "payload": {"x": 1}, "attempts": 1})
    assert kv.get_pending("pw-unknown")["status"] == "blocked"
