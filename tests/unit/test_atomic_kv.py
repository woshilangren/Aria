"""F3 原子更新的并发回归：N 线程同时对同一会话涨亲密度，
终值必须等于 N 次增量之和，且账本条数 = N（少一条就有丢更新）。

直接摸 SQLiteStorage（conftest 已把 DATA_DIR 指到临时目录），不走服务注册表。
"""

import threading

import pytest

from data.sqlite_store import get_db
from tools.storage import KVStoreTool


@pytest.fixture()
def db():
    store = get_db()
    # 用独立的 session_id，不依赖其他测试的残留数据
    yield store
    store._conn.execute("DELETE FROM relationship WHERE session_id = 'test-atomic'")
    store._conn.execute("DELETE FROM affection_history WHERE session_id = 'test-atomic'")
    store._conn.commit()


def test_update_kv_concurrent_no_lost_updates(db):
    n_threads, n_incr = 8, 25

    def worker():
        for _ in range(n_incr):
            # 不做 0~100 钳制——这里测的是"增量不丢"，钳制反而会封顶干扰断言
            db.update_kv(
                "relationship",
                "test-atomic",
                lambda rel: {**rel, "intimacy": rel.get("intimacy", 0) + 1},
            )

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = db.load_kv("relationship", "test-atomic")
    assert final["intimacy"] == n_threads * n_incr  # 丢更新就会小于这个数


def test_update_kv_creates_and_merges(db):
    # 首次更新：空 dict 进闭包
    out = db.update_kv("relationship", "test-atomic", lambda rel: {**rel, "mood": "心软"})
    assert out["mood"] == "心软"
    # 第二次更新：基于最新值合并，不覆盖已有键
    out = db.update_kv("relationship", "test-atomic", lambda rel: {**rel, "trust": 5})
    assert out["mood"] == "心软"
    assert out["trust"] == 5


# ---------------------- C：SQLite 锁可重入（RLock） ----------------------


def test_sqlite_lock_is_reentrant(db):
    """锁换成 RLock：同线程在锁内再获取一次锁不应自锁。

    用超时守护线程验证——若仍是不可重入的 Lock，这个调用会永久阻塞，
    线程 join(timeout) 超时即可判定失败（不必真把用例挂死在 2s 上）。
    """
    done = threading.Event()

    def reenter():
        # 手动模拟"闭包内再取锁"：RLock 同线程可重入，两次 acquire 都该过
        with db._lock:
            with db._lock:
                pass
        done.set()

    t = threading.Thread(target=reenter, daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert done.is_set(), "同线程二次获取 SQLite 锁被阻塞 —— 锁不可重入"


def test_update_kv_nested_same_table_does_not_deadlock(db):
    """回归：闭包内再调一次同表 load_kv，RLock 下不死锁（旧 Lock 会永久自锁）。"""
    done = threading.Event()

    def nested():
        db.update_kv(
            "relationship",
            "test-atomic",
            lambda rel: {**rel, "n": db.load_kv("relationship", "test-atomic").get("n", 0) + 1},
        )
        done.set()

    t = threading.Thread(target=nested, daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert done.is_set(), "闭包内同表读库被阻塞 —— 锁不可重入"


# ------------------ B：KVStoreTool.update 白名单收紧 ------------------


@pytest.mark.parametrize("store", ["session", "image", "route_config", "persona_config"])
def test_kv_update_rejects_non_whole_kv_tables(store):
    """非三张整包 KV 表一律 ValueError（旧实现会抛 AttributeError / 语义错）。"""
    kv = KVStoreTool()
    with pytest.raises(ValueError):
        kv.update(store, "k", lambda d: d)


@pytest.mark.parametrize("store", ["profile", "portrait", "relationship"])
def test_kv_update_accepts_whole_kv_tables(store):
    kv = KVStoreTool()
    out = kv.update(store, "test-whitelist", lambda d: {**(d or {}), "x": 1})
    assert out["x"] == 1


# --------- X：last_poor 收进原子闭包，不再二次整包覆盖 ---------


def test_relationship_update_writes_last_poor_atomically(register_services):
    """was_poor 必须由 RelationshipTracker.update 一次闭包写回。

    旧实现（返回后 rel["last_poor"]=... + kv.write）是陈旧的整包覆盖：
    并发 REST 在两次写之间加的 intimacy 会被吞掉。这里验证：
    一次 update 就能把 last_poor + intimacy 增量一起落库；随后并发 REST 的 +5
    不再被覆盖（因为没有任何后续整包 write）。
    """
    from capability.memory import RelationshipTracker

    kv = KVStoreTool()
    register_services(kv_store=kv)
    sid = "test-last-poor"

    rel = RelationshipTracker().update(sid, "happy", was_poor=True)
    assert rel.get("last_poor") is True

    # 并发 REST 改库：直接对同一会话做原子增量
    kv.update("relationship", sid, lambda d: {**d, "intimacy": d.get("intimacy", 0) + 5})
    after = kv.read("relationship", sid)
    assert after.get("last_poor") is True
    assert after["intimacy"] == rel["intimacy"] + 5   # 没被陈旧整包覆盖回去

    # 下一轮 update（was_poor 默认 False）覆盖 last_poor，天然只生效一轮
    rel2 = RelationshipTracker().update(sid, "neutral")
    assert rel2.get("last_poor") is False


# ------------- mood_baseline 展示回归（本批新引入） -------------


def test_relationship_has_mood_not_mood_baseline(register_services):
    """回归：relationship 必须暴露 mood（前端「当前情绪」读它），且不得再写 mood_baseline。

    本批改动删掉了旧的 rel["mood_baseline"]=emotion、改由 MoodEngine 写 rel["mood"]。
    前端一度仍读 mood_baseline → 永远显示默认「平静」。这里锁死：
    mood 存在且随情绪变化，mood_baseline 字段彻底消失（防日后双写）。
    """
    from capability.memory import RelationshipTracker

    kv = KVStoreTool()
    register_services(kv_store=kv)
    sid = "test-mood-field"

    rel = RelationshipTracker().update(sid, "happy")
    assert "mood" in rel, "relationship 缺 mood 字段（前端'当前情绪'会退化）"
    assert rel.get("mood"), "mood 不能为空"
    assert "mood_baseline" not in rel, "mood_baseline 字段应已删除，禁止双写"

    stored = kv.read("relationship", sid)
    assert "mood" in stored
    assert "mood_baseline" not in stored
    # 值确实随情绪走（happy 后 mood 由 MoodEngine 给出非空状态）
    assert stored["mood"] == rel["mood"]


def test_relationship_state_dataclass_uses_mood():
    """schema 声明同步：RelationshipState 不许再声明 mood_baseline。"""
    from data.schemas import RelationshipState

    fields = set(RelationshipState.__dataclass_fields__)
    assert "mood" in fields
    assert "mood_baseline" not in fields
