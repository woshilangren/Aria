"""J6 schema 迁移的单测：她的全部记忆都在 knowledge.db 这一个文件里。

为什么这一批值得单独立一个测试文件：改造前 `_init_tables` 是一堆
`CREATE TABLE IF NOT EXISTS`——**加新表能过，给老表加一列就会让上个月的老库
启动即 OperationalError**。迁移是本项目里少数"写错一次就真丢数据"的代码，
而它跑在启动路径上、平时一行都不执行，靠人工点检等于没有。

覆盖的四条不变式：
1. 全新库建到当前版本，表和索引一个不少（DDL 被误删要立刻红）；
2. 幂等：重复初始化不改版本、不动数据；
3. 老库自动补齐且**一行数据都不丢**（这是 J6 存在的理由）；
4. 失败整批回滚：版本号停在上一版、同批已建的表也撤掉、不留悬挂事务；
5. 库比代码新时只告警不动库（用户回滚了旧版本代码，硬跑迁移只会把新库改坏）。
"""

import sqlite3

import pytest

import data.sqlite_store as sm

# v0->v1 那批建出来的全部对象。写死清单是有意的：迁移里的 CREATE 语句被谁
# 顺手删掉一行，这里就该红——比"启动后某个功能莫名报 no such table"早得多。
_EXPECTED_TABLES = {
    "profile", "portrait", "relationship", "self", "chat_log",
    "affection_history", "memory_candidates", "memory_stats",
}
_EXPECTED_INDEXES = {"idx_chat_session_time", "idx_ledger_session", "idx_cand_session"}


def _version(store) -> int:
    return int(store._conn.execute("PRAGMA user_version").fetchone()[0])


def _objects(store, kind: str) -> set:
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type=?", (kind,)
    ).fetchall()
    return {r[0] for r in rows}


def test_fresh_db_reaches_current_version():
    """全新库：一次建到 _SCHEMA_VERSION，表和索引齐全。"""
    store = sm.get_db()
    assert _version(store) == sm._SCHEMA_VERSION
    assert _EXPECTED_TABLES <= _objects(store, "table")
    assert _EXPECTED_INDEXES <= _objects(store, "index")


def test_migration_is_idempotent_and_keeps_data():
    """重复跑迁移：版本号不动、已建对象不重复、数据一行不少。

    这条防的是"每次启动都重跑一遍 DDL"——IF NOT EXISTS 让它不报错，
    但版本号若也跟着乱走，后面的批次判断就全错了。
    """
    store = sm.get_db()
    store.save_kv("profile", "s1", {"nickname": "老王"})
    before_tables = _objects(store, "table")

    store._migrate()          # 第二次
    store._migrate()          # 第三次

    assert _version(store) == sm._SCHEMA_VERSION
    assert _objects(store, "table") == before_tables
    assert store.load_kv("profile", "s1") == {"nickname": "老王"}


def test_legacy_v0_db_is_upgraded_without_losing_rows(tmp_path, capsys):
    """老库（user_version=0、表是上一版代码建的）启动即自动补齐，数据不丢。

    这就是 J6 要解决的原始事故形状：老库里没有新列，指名它的 INSERT/SELECT 直接炸。
    """
    legacy = tmp_path / "knowledge.db"
    conn = sqlite3.connect(str(legacy))
    conn.execute("CREATE TABLE profile (session_id TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO profile VALUES ('s1', '{\"nickname\":\"老王\"}')")
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()

    store = sm.get_db()
    assert _version(store) == sm._SCHEMA_VERSION
    # 老数据一行不少，且新表也补齐了
    assert store.load_kv("profile", "s1") == {"nickname": "老王"}
    assert _EXPECTED_TABLES <= _objects(store, "table")
    assert "schema v0 ->" in capsys.readouterr().out


def test_db_newer_than_code_is_left_alone(tmp_path, capsys):
    """库比代码新（用户回滚了旧版本代码）：只告警，绝不动库。

    硬跑迁移会把新库改坏——旧代码认不出新列，但也没资格删它。
    """
    newer = tmp_path / "knowledge.db"
    conn = sqlite3.connect(str(newer))
    conn.execute("CREATE TABLE profile (session_id TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO profile VALUES ('s1', '{\"nickname\":\"老王\"}')")
    conn.execute("CREATE TABLE from_the_future (id INTEGER PRIMARY KEY)")
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()

    store = sm.get_db()
    assert _version(store) == 99                              # 版本号没被改小
    assert "from_the_future" in _objects(store, "table")      # 新表没被删
    assert store.load_kv("profile", "s1") == {"nickname": "老王"}
    out = capsys.readouterr().out
    assert "比这份代码期望的" in out and "跳过迁移" in out


def test_failed_batch_rolls_back_and_keeps_old_version(monkeypatch):
    """一批里有一句坏 SQL：整批回滚，版本号停在上一版，不留悬挂事务。

    这是迁移最要命的一条——"半张 schema + 新版本号"意味着下次启动认为已经迁好了，
    缺的那张表永远不会被补上。所以版本号必须和 DDL 在**同一个事务**里落地。
    """
    store = sm.get_db()
    assert _version(store) == sm._SCHEMA_VERSION

    good = "CREATE TABLE IF NOT EXISTS migration_probe (id INTEGER PRIMARY KEY)"
    bad = "THIS IS NOT SQL"
    # 追加一批"先建一张表、再炸"的迁移。两个模块级常量都要打补丁：
    # _SCHEMA_VERSION 是 import 时按 len(_MIGRATIONS) 算死的，只改列表不会跟着变。
    monkeypatch.setattr(sm, "_MIGRATIONS", sm._MIGRATIONS + ((good, bad),))
    monkeypatch.setattr(sm, "_SCHEMA_VERSION", sm._SCHEMA_VERSION + 1)

    with pytest.raises(sqlite3.Error):
        store._migrate()

    assert _version(store) == sm._SCHEMA_VERSION - 1          # 停在上一版，下次启动重试
    assert "migration_probe" not in _objects(store, "table")  # 同批已建的那张也撤了
    assert store._conn.in_transaction is False                # 没留悬挂事务
    # 库还能正常用：回滚没把连接搞坏
    store.save_kv("profile", "s1", {"nickname": "老王"})
    assert store.load_kv("profile", "s1") == {"nickname": "老王"}


def test_add_column_is_idempotent():
    """_add_column 重复调用不抛、只加一次。

    下一批迁移给老表加列全靠它：`ALTER TABLE ADD COLUMN` 撞上已存在的列会抛
    duplicate column name，而"这列在不在"取决于这个库当年是哪版代码建的。
    """
    store = sm.get_db()
    conn = store._conn

    def _cols():
        return [str(r[1]) for r in conn.execute("PRAGMA table_info(chat_log)").fetchall()]

    sm._add_column(conn, "chat_log", "verify_col", "TEXT DEFAULT ''")
    after_first = _cols()
    sm._add_column(conn, "chat_log", "verify_col", "TEXT DEFAULT ''")   # 第二次不该炸
    assert after_first == _cols()
    assert "verify_col" in after_first


# ------------------- R15a：提交边界与回执的持久结构 -------------------

# R15a 新增的持久对象。追加进"写死清单"：谁删了这里的 CREATE，这条先红。
_R15A_TABLES = {"turn_requests", "turn_commits", "derived_task_done"}
_R15A_INDEXES = {"idx_req_session", "idx_commit_request"}
_CHAT_LOG_NEW_COLS = ("turn_id", "disposition", "source_review_status", "reason_code")
_LEDGER_NEW_COLS = ("field", "old_value", "new_value", "event_id")


def test_r15a_fresh_db_has_new_structures():
    """全新库：R15a 的三张表、两个索引、chat_log/账本新列一个不少。"""
    store = sm.get_db()
    assert _R15A_TABLES <= _objects(store, "table")
    assert _R15A_INDEXES <= _objects(store, "index")
    for col in _CHAT_LOG_NEW_COLS:
        names = {str(r[1]) for r in store._conn.execute("PRAGMA table_info(chat_log)")}
        assert col in names, f"chat_log 缺新列 {col}"
    for col in _LEDGER_NEW_COLS:
        names = {str(r[1]) for r in store._conn.execute("PRAGMA table_info(affection_history)")}
        assert col in names, f"affection_history 缺新列 {col}"


def test_r15a_legacy_v1_db_upgraded_keeping_rows(tmp_path, monkeypatch):
    """v1 老库（无新表新列）打开即升到 v2：旧行保留、新列回填默认空串。"""
    import sqlite3 as s3

    db_path = tmp_path / "knowledge.db"  # get_db() 只认这个名字
    conn = s3.connect(db_path)
    # 手工搭 v1 schema：只建 chat_log 与账本（老列、无新列），user_version=1
    conn.execute(
        "CREATE TABLE chat_log (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,"
        " role TEXT NOT NULL, content TEXT NOT NULL, intent TEXT DEFAULT '', emotion TEXT DEFAULT '',"
        " mode TEXT DEFAULT '', created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE affection_history (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " session_id TEXT NOT NULL, old_score REAL NOT NULL, new_score REAL NOT NULL,"
        " delta REAL NOT NULL, reason TEXT DEFAULT '', source_quote TEXT DEFAULT '',"
        " created_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO chat_log (session_id, role, content, created_at) VALUES ('s1','user','老消息','2026-01-01T00:00:00')"
    )
    conn.execute(
        "INSERT INTO affection_history (session_id, old_score, new_score, delta, created_at)"
        " VALUES ('s1', 20, 21, 1, '2026-01-01T00:00:00')"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from config.settings import get_settings

    get_settings.cache_clear()
    store = sm.get_db()
    assert _version(store) == sm._SCHEMA_VERSION, "老库必须升到当前版本"
    assert _R15A_TABLES <= _objects(store, "table")
    # 旧行保留 + 新列默认空串
    row = store._conn.execute(
        "SELECT content, turn_id, disposition FROM chat_log WHERE session_id='s1'"
    ).fetchone()
    assert row[0] == "老消息" and row[1] == "" and row[2] == ""
    led = store._conn.execute(
        "SELECT old_score, new_score, field FROM affection_history WHERE session_id='s1'"
    ).fetchone()
    assert led[0] == 20 and led[1] == 21 and led[2] == "", "旧行的账目字段默认空串"
    # 新写入走默认 intimacy：旧分数列语义不被污染
    store.append_ledger("s1", 21.0, 22.0, reason="聊天升温")
    store.append_ledger("s1", "旧心情", "新心情", reason="心情变化", field="mood",
                        old_value="旧心情", new_value="新心情", event_id="ev-1")
    mood_row = store._conn.execute(
        "SELECT old_score, new_score, delta, field, old_value, new_value, event_id "
        "FROM affection_history WHERE field='mood'"
    ).fetchone()
    assert mood_row[0] == 0.0 and mood_row[1] == 0.0 and mood_row[2] == 0.0, (
        "非 intimacy 字段不许占用旧分数列"
    )
    assert mood_row[4] == '"旧心情"' and mood_row[5] == '"新心情"' and mood_row[6] == "ev-1"
    intimacy_rows = store._conn.execute(
        "SELECT old_score, new_score FROM affection_history WHERE field='intimacy' OR field=''"
    ).fetchall()
    assert all(r[1] - r[0] == 1 for r in intimacy_rows), "intimacy 行分数语义不变"
    # 旧读接口兼容：recent_ledger 的旧键仍在、新增键就位
    entries = store.recent_ledger("s1", n=5)
    assert {"old", "new", "delta", "reason", "quote", "time"} <= set(entries[0].keys())
    assert {"field", "old_value", "new_value", "event_id"} <= set(entries[0].keys())
    get_settings.cache_clear()


def test_new_turn_id_never_collides():
    """轮 ID 生成器：跨调用不碰撞（持久唯一键的前提）。"""
    ids = {sm.new_turn_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(isinstance(i, str) and len(i) == 32 for i in ids)
