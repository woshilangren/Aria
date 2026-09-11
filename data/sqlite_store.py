"""SQLite 知识库：结构化数据统一进一个库文件（storage/knowledge.db）。

管四张表：
- profile / portrait / relationship：一 session 一行，value 存整个 JSON。
  这三类数据都是"整包读、整包写"，JSON 一列最省事，字段以后随便加。
- chat_log：聊天记录按行存（列化），写日记时要捞"今天的对话"，
  按时间和会话查就行，不用把整个 JSON 文件读出来翻。

和 JSON 文件时代的分工：KVStoreTool 的接口一点不变，只是底下换成这里。
"""

import json
import sqlite3
import threading
from typing import List, Optional

from config.settings import get_settings

# 允许当 KV 表用的表名白名单，防呆：写错名字宁可报错也别建出脏表
_KV_TABLES = ("profile", "portrait", "relationship")


class SQLiteStorage:
    """项目唯一的 SQLite 入口：建库、建表、读写都从这走。"""

    def __init__(self):
        cfg = get_settings()
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = cfg.data_dir / "knowledge.db"
        # check_same_thread=False：闲置主动关怀等功能会在别的线程摸库，
        # 加一把锁保证同一时刻只有一个线程在写，比每句话开新连接省事
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile (
                    session_id TEXT PRIMARY KEY,
                    value      TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portrait (
                    session_id TEXT PRIMARY KEY,
                    value      TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS relationship (
                    session_id TEXT PRIMARY KEY,
                    value      TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    intent     TEXT DEFAULT '',
                    emotion    TEXT DEFAULT '',
                    mode       TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            # 日记生成按"会话 + 时间段"捞对话，这个索引就是给那一路用的
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_session_time ON chat_log(session_id, created_at)"
            )

    # ---------- 三张 KV 表：整包 JSON 读 / 整包 JSON 写 ----------

    def load_kv(self, table: str, session_id: str) -> dict:
        """按表名 + 会话读整包 JSON，没有就给空 dict。"""
        if table not in _KV_TABLES:
            raise ValueError(f"不认识的 KV 表: {table}")
        with self._lock:
            row = self._conn.execute(
                f"SELECT value FROM {table} WHERE session_id = ?", (session_id,)
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return {}

    def save_kv(self, table: str, session_id: str, value: dict) -> None:
        """整包写回：有就覆盖，没有就插入。"""
        if table not in _KV_TABLES:
            raise ValueError(f"不认识的 KV 表: {table}")
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO {table} (session_id, value) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET value = excluded.value",
                (session_id, json.dumps(value, ensure_ascii=False)),
            )

    # ---------- 聊天记录：一行一条，追加着写 ----------

    def append_chat(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = "",
        emotion: str = "",
        mode: str = "",
        created_at: str = "",
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO chat_log (session_id, role, content, intent, emotion, mode, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, role, content, intent, emotion, mode, created_at),
            )

    def get_recent_chat(self, session_id: str, n: int = 10) -> List[dict]:
        """最近 n 条对话，按时间正序返回（老 -> 新），给恢复短期记忆用。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, intent, emotion, mode, created_at FROM chat_log "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, n),
            ).fetchall()
        # 倒序取的，翻回来才是对话本来的顺序
        return [
            {
                "role": r[0],
                "text": r[1],
                "intent": r[2],
                "emotion": r[3],
                "mode": r[4],
                "time": r[5],
            }
            for r in reversed(rows)
        ]

    def get_chats_between(
        self, session_id: str, start_iso: str, end_iso: str
    ) -> List[dict]:
        """捞某个时间段内的对话（含头不含尾），写日记时取"今天聊了啥"用。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, intent, emotion, mode, created_at FROM chat_log "
                "WHERE session_id = ? AND created_at >= ? AND created_at < ? "
                "ORDER BY id ASC",
                (session_id, start_iso, end_iso),
            ).fetchall()
        return [
            {
                "role": r[0],
                "text": r[1],
                "intent": r[2],
                "emotion": r[3],
                "mode": r[4],
                "time": r[5],
            }
            for r in rows
        ]

    def count_chat_days(self, session_id: str) -> Optional[int]:
        """这个会话一共聊过几天（按日期去重），暂时只给日记功能做参考。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT substr(created_at, 1, 10)) FROM chat_log WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row[0] if row else None

    def last_chat_per_session(self) -> List[dict]:
        """每个会话最后一条聊天的时间，闲置检测按这个判断谁该写日记了。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, MAX(created_at) FROM chat_log GROUP BY session_id"
            ).fetchall()
        return [{"session_id": r[0], "last_time": r[1]} for r in rows if r[1]]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# 全项目共用一个 SQLite 连接，别到处 new
_db: Optional[SQLiteStorage] = None
_db_lock = threading.Lock()


def get_db() -> SQLiteStorage:
    """懒加载单例：第一次用时建库建表，之后一直用同一份连接。

    双重检查锁：首次并发调用时，只判一次空会建出两份连接（各写各的、互相覆盖），
    所以拿到锁之后必须再判一次空。锁只在"还没建好"这条路上生效，
    建好之后的每次调用都只是读一次全局变量，不进锁。
    """
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = SQLiteStorage()
    return _db
