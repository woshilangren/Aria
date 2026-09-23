"""R15c：一轮提交的事务门面测试（8.11.3 提交契约的低层能力）。

验收（《计划与设计.md》R15c）：
- 一次短事务写 user/assistant、关系（事务内读最新 + 注入纯计算）、账本、回执；
- 第二条 chat INSERT / 账本 INSERT 失败 → 整体回滚，无半轮、无回执；
- 同 request_id 重试返回**同一回执**（不重复计数）；同 ID 不同 digest → conflict；
- Event 控制三种交错：提交前（并发竞争）/ 事务中（另一入口必须等待）/
  提交后（重试拿原回执）。
- 边界：取消与提交的原子门竞争（取消错序不误杀）属 R15d。

relation_fn 由测试注入（R15b 纯函数偏应用）——data 层不 import capability。
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from data.sqlite_store import get_db, new_turn_id
from orchestration.cancellation import CommitGate
from tools.storage import KVStoreTool


@pytest.fixture()
def kv():
    return KVStoreTool()


def _relation_fn(delta=1):
    """注入的"纯计算"：读现态加 delta（模拟 R15b 的偏应用）。"""
    from capability.memory import compute_relationship_update
    from capability.quirks import MoodEngine

    def _apply(rel):
        new_rel, _ = compute_relationship_update(
            rel, emotion="happy", mood_engine=MoodEngine(), default_intimacy=20
        )
        return new_rel

    return _apply


def _commit(kv, sid, req, turn=None, digest="d1", assistant="她的回答。",
            user="你好", disposition="normal", relation_fn=None, ledger=None):
    return kv.commit_turn(
        session_id=sid, turn_id=turn or new_turn_id(), request_id=req,
        request_digest=digest, disposition=disposition,
        source_review_status="accepted",
        user_text=user, assistant_text=assistant,
        intent="chat", emotion="happy", mode="text",
        relation_fn=relation_fn if relation_fn is not None else _relation_fn(),
        ledger=ledger,
    )


def test_commit_happy_path_writes_all_in_one_transaction(kv):
    """正常提交：chat 双行带 turn_id/disposition、关系更新、账本、回执齐全。"""
    turn = new_turn_id()
    rc = _commit(kv, "s-tc1", "req-1", turn=turn, ledger={
        "field": "intimacy", "old": 20, "new": 22,
        "reason": "聊得开心", "quote": "你好", "event_id": "ev-1",
    })
    assert rc["status"] == "committed"
    assert rc["message_ids"] and len(rc["message_ids"]) == 2

    rows = kv.read("session", "s-tc1") or []
    assert [r.get("role") for r in rows] == ["user", "assistant"]
    # chat_log 新列经 store 读不到（store 只暴露老列）——直查库验证 R15a 列
    from data.sqlite_store import get_db as _gdb

    raw = _gdb()._conn.execute(
        "SELECT role, turn_id, disposition, source_review_status FROM chat_log "
        "WHERE session_id='s-tc1' ORDER BY id"
    ).fetchall()
    assert all(r[1] == turn for r in raw)
    assert all(r[2] == "normal" and r[3] == "accepted" for r in raw)

    rel = kv.read("relationship", "s-tc1") or {}
    assert rel.get("intimacy", 0) >= 20, "事务内的关系纯计算必须生效"
    led = kv.recent_ledger("s-tc1")
    assert led and led[-1]["field"] == "intimacy" and led[-1]["event_id"] == "ev-1"

    receipt = kv.get_commit_receipt("s-tc1", "req-1")
    assert receipt and receipt["status"] == "committed"
    assert receipt["message_ids"] == rc["message_ids"]


def test_retry_same_request_returns_same_receipt_no_double_count(kv):
    """同 request 重试：返回**原回执**（同 committed_at/同 message_ids），
    chat/账本/关系**不重复计数**。"""
    turn = new_turn_id()
    first = _commit(kv, "s-tc2", "req-x", turn=turn)
    assert first["status"] == "committed"
    n_chat = len(kv.read("session", "s-tc2") or [])
    n_led = len(kv.recent_ledger("s-tc2", n=50))

    second = _commit(kv, "s-tc2", "req-x", turn=turn)
    assert second["status"] == "already_committed"
    assert second["committed_at"] == first["committed_at"]
    assert second["message_ids"] == first["message_ids"]
    assert len(kv.read("session", "s-tc2") or []) == n_chat, "重试不许再写 chat"
    assert len(kv.recent_ledger("s-tc2", n=50)) == n_led, "重试不许再记账本"


def test_same_request_different_digest_is_conflict(kv):
    """同 request 不同内容（digest 不同）→ 明确 conflict，不默默当旧轮。"""
    turn = new_turn_id()
    assert _commit(kv, "s-tc3", "req-y", turn=turn, digest="d1")["status"] == "committed"
    rc = _commit(kv, "s-tc3", "req-y", turn=new_turn_id(), digest="d2-不同内容",
                 assistant="另一条回答。")
    assert rc["status"] == "conflict"
    rows = kv.read("session", "s-tc3") or []
    assert len(rows) == 2, "冲突重试不许写入新内容"


def test_chat_insert_failure_rolls_back_everything(kv, monkeypatch):
    """第二条 chat INSERT 失败：整体回滚——无半轮、无回执、无请求登记、关系不动。"""
    from data.sqlite_store import get_db as _gdb

    store = _gdb()
    real_conn = store._conn

    class _FailSecondChatInsert:
        """代理连接：第二条 chat_log INSERT 时抛错（确定性注入）。"""

        def __init__(self, conn):
            self._conn = conn
            self._n = 0

        def execute(self, sql, *args):
            if "INSERT INTO chat_log" in sql:
                self._n += 1
                if self._n == 2:
                    raise sqlite3.OperationalError("注入：第二条 chat INSERT 失败")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(store, "_conn", _FailSecondChatInsert(real_conn))
    rc = _commit(kv, "s-tc4", "req-f")
    assert rc["status"] == "failed"

    monkeypatch.setattr(store, "_conn", real_conn)
    # 四类痕迹全部为零：无半轮
    assert kv.read("session", "s-tc4") in (None, [], {})
    assert kv.read("relationship", "s-tc4") in (None, {}, [])
    assert kv.recent_ledger("s-tc4") == []
    assert kv.get_commit_receipt("s-tc4", "req-f") is None
    raw = store._conn.execute(
        "SELECT COUNT(*) FROM turn_requests WHERE request_id='req-f'"
    ).fetchone()[0]
    assert raw == 0, "失败事务连请求登记也一并回滚"


def test_ledger_insert_failure_rolls_back_everything(kv, monkeypatch):
    """账本 INSERT 失败：整体回滚——**无账的关系变化也不许存在**。"""
    from data.sqlite_store import get_db as _gdb

    store = _gdb()
    real_conn = store._conn

    class _FailLedger:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args):
            if "INSERT INTO affection_history" in sql:
                raise sqlite3.OperationalError("注入：账本 INSERT 失败")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(store, "_conn", _FailLedger(real_conn))
    rc = _commit(kv, "s-tc5", "req-g", ledger={
        "field": "intimacy", "old": 20, "new": 25, "reason": "x",
    })
    assert rc["status"] == "failed"

    monkeypatch.setattr(store, "_conn", real_conn)
    assert kv.read("session", "s-tc5") in (None, [], {}), "无半轮"
    assert kv.read("relationship", "s-tc5") in (None, {}, []), "账本失败不许留下关系变化"
    assert kv.recent_ledger("s-tc5") == []
    assert kv.get_commit_receipt("s-tc5", "req-g") is None


def test_relation_fn_failure_rolls_back(kv, monkeypatch):
    """注入的纯计算炸了（本地缺陷）：整体回滚，不留任何痕迹。"""
    from data.sqlite_store import get_db as _gdb

    def boom(rel):
        raise NameError("注入：纯计算本地缺陷")

    store = _gdb()
    real_conn = store._conn
    rc = _commit(kv, "s-tc6", "req-h", relation_fn=boom)
    assert rc["status"] == "failed"
    monkeypatch.setattr(store, "_conn", real_conn)
    assert kv.read("session", "s-tc6") in (None, [], {})
    assert kv.get_commit_receipt("s-tc6", "req-h") is None


def test_event_interleaving_commit_before_during_after(kv):
    """Event 控制三种交错：
    - 事务中：入口 A 在事务内被 Event 挡住时，入口 B 必须等它出结果；
    - 提交后：B 拿到 already_committed 原回执；
    - 提交前（并发竞争）：两线程同 request 同时提交 → 恰好一个 committed。
    """
    store = get_db()
    sid = "s-tc7"
    turn = new_turn_id()
    release = threading.Event()

    def blocking_relation(rel):
        release.wait(timeout=5)  # 事务中挂起：放大交错窗口
        return _relation_fn()(rel)

    done = {}

    def _worker(req, tag, rel_fn=None):
        rc = kv.commit_turn(
            session_id=sid, turn_id=turn or new_turn_id(), request_id=req,
            request_digest="d1", disposition="normal",
            user_text="你好", assistant_text="回答。", mode="text",
            relation_fn=rel_fn or _relation_fn(), ledger={
                "field": "intimacy", "old": 20, "new": 21, "reason": "r",
            },
        )
        done[tag] = rc["status"]

    # 事务中：A 挂在事务内，B 等待（BEGIN IMMEDIATE 串行化）
    t_a = threading.Thread(target=_worker, args=("req-i", "A", blocking_relation))
    t_a.start()
    import time as _t

    deadline = _t.monotonic() + 5
    while "A" not in done and _t.monotonic() < deadline:
        _t.sleep(0.01)
    # A 还没出结果（被 Event 挡住）——此时 B 提交同 request 必须排队而不是并行写
    t_b = threading.Thread(target=_worker, args=("req-i", "B"))
    t_b.start()
    release.set()
    t_a.join(timeout=5)
    t_b.join(timeout=5)
    assert sorted(done.values()) == ["already_committed", "committed"], f"实测 {done}"
    assert len(kv.read("session", sid) or []) == 2, "两线程交错也不许双写"

    # 提交后：再次重试拿原回执
    _worker("req-i", "C")
    assert done["C"] == "already_committed"

    # 提交前（并发竞争）：全新 request，两线程同时冲 → 恰好一个 committed
    done2 = {}
    barrier = threading.Barrier(2)

    def _race(tag):
        barrier.wait()
        rc = kv.commit_turn(
            session_id=sid, turn_id=new_turn_id(), request_id="req-race",
            request_digest="d1", user_text="你好", assistant_text="答。",
            mode="text", relation_fn=_relation_fn(),
        )
        done2[tag] = rc["status"]

    ths = [threading.Thread(target=_race, args=(t,)) for t in ("R1", "R2")]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout=5)
    assert sorted(done2.values()) == ["already_committed", "committed"], f"实测 {done2}"


# ------------------- R15d：取消与提交互斥（同一原子门） -------------------

def test_cancel_wins_before_commit_blocks_data_transaction(kv):
    """取消先赢：run_commit 直接返回 cancelled，数据事务**根本不开始**
    （无 chat/账本/回执/关系痕迹——用门内 spy 证明 fn 未被调用）。"""
    from orchestration.cancellation import CommitGate

    gate = CommitGate()
    assert gate.cancel_turn("turn-c1", reason="用户取消")["status"] == "cancelled"

    called = {"n": 0}

    def fn():
        called["n"] += 1
        return {"status": "committed", "turn_id": "turn-c1"}

    rc = gate.run_commit("turn-c1", fn)
    assert rc["status"] == "cancelled"
    assert called["n"] == 0, "取消先赢时数据事务不许开始"
    assert gate.is_committed("turn-c1") is False


def test_commit_wins_late_cancel_returns_already_committed(kv):
    """提交先成功：迟到取消返回 already_committed（带原回执），不误取消。"""
    from orchestration.cancellation import CommitGate

    gate = CommitGate()
    receipt = {"status": "committed", "turn_id": "turn-c2", "committed_at": "T1",
               "message_ids": [1, 2]}
    rc = gate.run_commit("turn-c2", lambda: receipt)
    assert rc["status"] == "committed"
    assert gate.is_committed("turn-c2") is True

    late = gate.cancel_turn("turn-c2", reason="迟到取消")
    assert late["status"] == "already_committed"
    assert late["receipt"] is receipt, "迟到取消必须指向原回执"


def test_commit_failure_does_not_keep_committed_status():
    """提交失败：失败回执不登记——此后的取消返回 cancelled（不保留 COMMITTED）。"""
    from orchestration.cancellation import CommitGate

    gate = CommitGate()
    rc = gate.run_commit("turn-c3", lambda: {"status": "failed",
                                             "reason_code": "commit_failed"})
    assert rc["status"] == "failed"
    assert gate.is_committed("turn-c3") is False
    late = gate.cancel_turn("turn-c3")
    assert late["status"] == "cancelled", "提交失败后取消必须是 cancelled"


def test_cancel_during_commit_waits_for_exact_result(kv):
    """处理中（事务内）来的取消：在门上排队，**等待确切结果**——
    事务提交成功 → 取消得到 already_committed（不存在假 cancelled）。"""
    from orchestration.cancellation import CommitGate

    gate = CommitGate()
    release = threading.Event()
    in_txn = threading.Event()

    def slow_fn():
        in_txn.set()  # 已进事务并持有门锁
        release.wait(timeout=5)  # 事务中挂起
        return {"status": "committed", "turn_id": "turn-c4", "message_ids": [9]}

    result = {}

    def committer():
        result["commit"] = gate.run_commit("turn-c4", slow_fn)

    t = threading.Thread(target=committer)
    t.start()
    assert in_txn.wait(timeout=5), "提交者必须先进入事务（持门锁）"
    # 事务进行中：取消请求到达 → 在门上排队，等确切结果
    def canceller():
        result["cancel"] = gate.cancel_turn("turn-c4", reason="处理中取消")

    t2 = threading.Thread(target=canceller)
    t2.start()
    import time as _t

    _t.sleep(0.05)
    assert "cancel" not in result, "取消必须等待事务出结果，不许立刻假答"
    release.set()
    t.join(timeout=5)
    t2.join(timeout=5)
    assert result["commit"]["status"] == "committed"
    assert result["cancel"]["status"] == "already_committed", (
        f"事务成功的轮，处理中到达的取消必须拿到 already_committed，实测 {result}"
    )


def test_new_turn_start_does_not_kill_committed_turn_tasks(kv):
    """新轮 start 顶掉旧轮 handle——但已提交轮的派生任务看门（is_committed），
    不看 handle：不许被新轮误杀。"""
    from orchestration.cancellation import COMMIT_GATE, TurnRegistry

    gate = CommitGate()
    turn_old = new_turn_id()
    rc = gate.run_commit(turn_old, lambda: {"status": "committed",
                                            "turn_id": turn_old})
    assert rc["status"] == "committed"

    reg = TurnRegistry()
    reg.start("sess-new")          # 新轮启动：会顶掉同会话旧 handle
    # 旧轮已提交：门说它的派生任务仍然有效
    assert gate.is_committed(turn_old) is True
    assert COMMIT_GATE is not None  # 全局唯一门存在
    # F4：不带 turn_id 的取消依旧被拒（不误杀）
    assert reg.cancel("sess-new", turn_id=None) is False


def test_receipt_type_carries_contract_fields():
    """CommitReceipt 类型（8.11.3 最小字段）可构造且字段齐全。"""
    from shared.types import CommitReceipt

    rc = CommitReceipt(status="committed", session_id="s", request_id="r",
                       turn_id="t", disposition="degraded",
                       committed_at="2026-09-22T00:00:00",
                       message_ids=[1, 2], reason_code="")
    assert rc.disposition == "degraded" and rc.message_ids == [1, 2]


# ------------------- R18a：派生任务新调度入口 -------------------

def test_schedule_commit_tasks_only_after_committed():
    """只有 committed 回执才调度：cancelled/failed/缺回执一律不产生任务。"""
    from orchestration import writeback as wb
    from shared.types import DeferredTask

    tasks = [DeferredTask(kind=wb.TASK_IDENTITY_FREEZE, fn=lambda: None)]
    assert wb.schedule_commit_tasks({"status": "failed", "turn_id": "t-x"}, tasks) == []
    assert wb.schedule_commit_tasks({"status": "cancelled", "turn_id": "t-x"}, tasks) == []
    assert wb.pending_commit_tasks() == {}, "未提交轮不许进登记表"

    run = wb.schedule_commit_tasks({"status": "committed", "turn_id": "t-ok"}, tasks)
    assert len(run) == 1
    key = wb.make_task_key("t-ok", wb.TASK_IDENTITY_FREEZE)
    assert key in wb.pending_commit_tasks()


def test_schedule_commit_tasks_idempotent():
    """同轮同任务重复调度（重试/重启重放）：幂等键拦截，不重复登记。"""
    from orchestration import writeback as wb
    from shared.types import DeferredTask

    tasks = [DeferredTask(kind=wb.TASK_PORTRAIT, fn=lambda: None)]
    wb.schedule_commit_tasks({"status": "committed", "turn_id": "t-idem"}, tasks)
    run2 = wb.schedule_commit_tasks({"status": "committed", "turn_id": "t-idem"}, tasks)
    assert run2 == [], "同轮同类任务不许重复调度"
    assert len(wb.pending_commit_tasks()) >= 1


def test_coordinator_returns_typed_deferred_tasks(kv, register_services):
    """协调器产出的待办已升级为 DeferredTask 载荷（kind 绑定）；legacy 适配器
    解包 fn 的路径不破（handle 全跑一轮验证）。"""
    import asyncio

    from orchestration import writeback as wb
    from orchestration.pipeline import DialoguePipeline
    from shared.types import InputMessage
    from tests.unit.test_pipeline_e2e import FakeLLM as _E2ELLM
    from tools.speech import split_emotion  # noqa: F401

    from tests.unit.test_reply_contract import _register as _reg

    llm = _E2ELLM(stream_script=["今天也是元气满满的一天。"])
    _reg(register_services, llm)
    sid = "r18a-typed"
    rc = asyncio.run(
        DialoguePipeline().handle(InputMessage(text="随便聊聊", session_id=sid))
    )
    assert rc.text, "正常轮应有正文"
    # 登记表里的键必须是 (turn_id, kind) 形状——legacy 适配器解包 fn 时
    # 不破坏载荷语义
    for key in wb.pending_commit_tasks():
        assert key.count(":") >= 1

# ------------------- R17d：提交前隐式写入清点（热度/素材消费移到提交后） -------------------

class _KeeperStub:
    """post_commit 里的 KEEPER 替身：追加与上下文都进内存。"""

    def __init__(self):
        self.msgs = []

    def append_turn(self, sid, user, assistant):
        self.msgs += [{"role": "user", "content": user},
                      {"role": "assistant", "content": assistant}]

    def get_context(self, sid):
        return list(self.msgs)

    def has_pending_summary(self, sid):
        return False

    def run_pending_summary(self, sid):
        return True


def _prepared(**kw):
    from shared.types import PreparedTurn

    base = dict(session_id="r17d", turn_id="t-r17d", request_id="req-r17d",
                request_digest="d", user_text="hi", assistant_text="你好")
    base.update(kw)
    return PreparedTurn(**base)


def test_post_commit_deferred_writes_only_on_committed_normal(kv, register_services, monkeypatch):
    """R17d：召回热度与素材烧计数只在提交成功的正常轮补账。

    - receipt 非 committed（取消/失败/重试 already_committed）→ 一概不动；
    - degraded 轮即使提交成功也不烧素材、不记热度（副作用表）；
    - normal committed → 每种效果恰好一次，重试轮不重复补账。
    """
    from capability import char_life
    from orchestration import writeback as wb
    from orchestration.writeback import WritebackCoordinator

    register_services(kv_store=kv)
    touched, burned = [], []
    db = get_db()
    monkeypatch.setattr(db, "touch_memories", touched.extend)
    monkeypatch.setattr(char_life, "commit_topic",
                        lambda sid, topic: burned.append((sid, topic)))
    monkeypatch.setattr(wb, "ConversationDistiller", lambda: type(
        "D", (), {"distill_turn": lambda *a, **k: None})())

    coord = WritebackCoordinator(keeper=_KeeperStub())
    ok = _prepared(memory_ids=["m1", "m2"], burn_topic="那本书")
    coord.post_commit(ok, {"status": "committed", "turn_id": ok.turn_id})
    assert touched == ["m1", "m2"]
    assert burned == [("r17d", "那本书")]

    # 重试拿原回执：post_commit 早退，不重复补账
    coord.post_commit(ok, {"status": "already_committed", "turn_id": ok.turn_id})
    assert touched == ["m1", "m2"]
    assert burned == [("r17d", "那本书")]

    # 取消/失败回执：不动
    coord.post_commit(ok, {"status": "cancelled", "turn_id": ok.turn_id})
    assert touched == ["m1", "m2"]
    assert burned == [("r17d", "那本书")]

    # degraded 提交成功也不许消耗（副作用表：零学习更新）
    deg = _prepared(turn_id="t-r17d-d", request_id="req-r17d-d",
                    disposition="degraded", memory_ids=["m1"], burn_topic="那本书")
    coord.post_commit(deg, {"status": "committed", "turn_id": deg.turn_id})
    assert touched == ["m1", "m2"]
    assert burned == [("r17d", "那本书")]
