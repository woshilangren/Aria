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
from datetime import datetime, timedelta
from typing import Callable, List, Optional

from config.settings import get_settings

# 允许当 KV 表用的表名白名单，防呆：写错名字宁可报错也别建出脏表
# self：她自己定下的身份（名字/年龄/城市/自我认知），懒生成冻结（批次0）
_KV_TABLES = ("profile", "portrait", "relationship", "self")


class SQLiteStorage:
    """项目唯一的 SQLite 入口：建库、建表、读写都从这走。"""

    def __init__(self):
        cfg = get_settings()
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = cfg.data_dir / "knowledge.db"
        # check_same_thread=False：闲置主动关怀等功能会在别的线程摸库，
        # 加一把锁保证同一时刻只有一个线程在写，比每句话开新连接省事。
        #
        # 用 RLock（可重入）而不是 Lock：update_kv 的闭包在**锁内**执行，
        # 如果闭包内再调一次同表 update_kv / load_kv（同线程二次获取），
        # 不可重入的 Lock 会**永久自锁**。当前不死锁只是因为 PersonaConfigStore
        # 走文件读、不碰 SQLite 锁；一旦将来把 persona_config 迁到 SQLite，
        # 就会立刻变成硬死锁。RLock 同线程可重入，直接消除这个陷阱。
        #
        # 注意：可重入**不**等于闭包内可以随便调库——闭包内仍禁止做任何
        # 会再次获取 SQLite 锁的方法（read/update/save/append），
        # 账本 append_ledger 必须在锁外调用（见 update_kv 的 docstring）。
        self._lock = threading.RLock()
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
            # 她自己的身份：名字、年龄、城市、生活、自我认知——全部由她本人
            # 在对话里"第一次说出口"时定下并冻结（懒生成，批次0），代码零预设
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS self (
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
            # 关系数值账本（S3）：每一次数值变动都留痕（old/new/delta/reason/原话），
            # 她能回答"你为什么生气"，用户可审计，"最近关系升温/变僵"也从这里聚合
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS affection_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    old_score   REAL NOT NULL,
                    new_score   REAL NOT NULL,
                    delta       REAL NOT NULL,
                    reason      TEXT DEFAULT '',
                    source_quote TEXT DEFAULT '',
                    created_at  TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ledger_session ON affection_history(session_id, id)"
            )
            # 记忆候选池（S4）：LLM 抽的"事实"先入池，同内容出现 >=2 次才晋升入档
            # ——对 LLM 抽取的记忆"先怀疑、再验证、慢接受"，幻觉不许焊死在档案里
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_candidates (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    field      TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    quote      TEXT DEFAULT '',
                    hits       INTEGER NOT NULL DEFAULT 1,
                    first_seen TEXT NOT NULL,
                    promoted   INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cand_session ON memory_candidates(session_id, field)"
            )
            # 记忆生命力（S5）：召回即 touch；常想起的更牢，没人提的自然淡
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_stats (
                    doc_id    TEXT PRIMARY KEY,
                    count     INTEGER NOT NULL DEFAULT 0,
                    last_seen TEXT NOT NULL
                )
                """
            )

    # ---------- 三张 KV 表：整包 JSON 读 / 整包 JSON 写 ----------

    def load_kv(self, table: str, session_id: str) -> dict:
        """按表名 + 会话读整包 JSON，没有就给空 dict。

        JSON 损坏时不能静默当"第一次聊天"——上层（RelationshipTracker）会把空 dict
        当新会话、用 default_intimacy 重置整段关系。所以这里把损坏原文备份到磁盘
        并大声警告，数据虽然救不回来，至少有迹可循、可手工恢复。
        """
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
            try:
                bak = get_settings().data_dir / f"{table}_{session_id}.corrupt.bak"
                bak.parent.mkdir(parents=True, exist_ok=True)
                bak.write_text(row[0], encoding="utf-8")
            except OSError:
                pass
            print(f"[store] {table}.{session_id} 的 JSON 损坏了！原文已备份到 "
                  f"{table}_{session_id}.corrupt.bak，本次按空数据处理——"
                  f"请从备份手工恢复，别让关系数值被当新会话重置")
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

    def update_kv(self, table: str, session_id: str, fn: Callable[[dict], dict], default: Optional[dict] = None) -> dict:
        """读-改-写原子闭包（F3）：锁内读整包 → 执行 fn → 锁内写回。

        profile/portrait/relationship 是"整包读改写"，写方有四个并发来源
        （聊天线程池、语音写回、巡检线程、REST 端点）——两写方同时读到同一份
        旧值时后写者会覆盖先写者的增量（好感度静默漏账）。唯一解法是把
        读、改、写收进同一把锁，fn 必须是纯计算（锁内禁网络/禁 LLM）。

        **闭包内禁止调用任何会再次获取 SQLite 锁的方法（load_kv / save_kv /
        update_kv / append_ledger / append_chat / get_recent_chat ...）**；
        需要写账本（append_ledger）时必须在锁外、闭包返回之后再调用。
        self._lock 虽已换成可重入的 RLock（消除同线程二次获取的永久自锁），
        但"可重入"只保证不死锁、不保证语义正确：闭包内嵌套写库会看到中间态、
        把外层的原子性拆开，仍是错误用法。
        """
        if table not in _KV_TABLES:
            raise ValueError(f"不认识的 KV 表: {table}")
        with self._lock:
            row = self._conn.execute(
                f"SELECT value FROM {table} WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row:
                try:
                    current = json.loads(row[0])
                except json.JSONDecodeError:
                    current = dict(default or {})
            else:
                current = dict(default or {})
            updated = fn(current) or {}
            if not updated:
                # 闭包判定"没什么可写"：不落库（落空 dict 会预创建无意义行，
                # 还会让上层的"首次初始化"分支失效——实测踩中过）
                return updated
            self._conn.execute(
                f"INSERT INTO {table} (session_id, value) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET value = excluded.value",
                (session_id, json.dumps(updated, ensure_ascii=False)),
            )
            self._conn.commit()
        return updated

    # ---------- 关系数值账本（S3）：追加写 + 近 N 条读 ----------

    def append_ledger(self, session_id: str, old: float, new: float,
                      reason: str = "", source_quote: str = "") -> None:
        """记一笔数值变动。账本是审计面不是数据面：写失败只告警，不回滚数值。"""
        delta = round(float(new) - float(old), 4)
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO affection_history "
                    "(session_id, old_score, new_score, delta, reason, source_quote, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, float(old), float(new), delta,
                     reason[:150], (source_quote or "")[:150],
                     datetime.now().isoformat(timespec="seconds")),
                )
        except sqlite3.Error as exc:
            print(f"[ledger] 账本写入失败（数值本身已生效）: {exc}")

    def recent_ledger(self, session_id: str, n: int = 8) -> List[dict]:
        """最近 n 条账本，旧 -> 新。S10 氛围线从这里聚合趋势。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT old_score, new_score, delta, reason, source_quote, created_at "
                "FROM affection_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, n),
            ).fetchall()
        return [
            {"old": r[0], "new": r[1], "delta": r[2], "reason": r[3],
             "quote": r[4], "time": r[5]}
            for r in reversed(rows)
        ]

    # ---------- 记忆候选池（S4）：入池计数 / 晋升标记 / 过期清理 ----------

    def candidate_hit(self, session_id: str, field: str, content: str,
                      quote: str = "") -> tuple:
        """同内容候选命中一次：已在池则 hits+1，不在则入池。返回 (hits, id)。"""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT id, hits FROM memory_candidates "
                "WHERE session_id=? AND field=? AND content=? AND promoted=0",
                (session_id, field, content),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE memory_candidates SET hits=hits+1 WHERE id=?", (row[0],)
                )
                return row[1] + 1, row[0]
            cur = self._conn.execute(
                "INSERT INTO memory_candidates (session_id, field, content, quote, hits, first_seen) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                (session_id, field, content, quote[:200],
                 datetime.now().isoformat(timespec="seconds")),
            )
            return 1, cur.lastrowid

    def mark_candidate_promoted(self, cand_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE memory_candidates SET promoted=1 WHERE id=?", (cand_id,)
            )

    def prune_candidates(self, days: int = 14) -> int:
        """入池超过 N 天还没晋升的候选：放弃（证据不足）。返回清理条数。"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM memory_candidates WHERE promoted=0 AND first_seen < ?",
                (cutoff,),
            )
            return cur.rowcount

    # ---------- 记忆生命力（S5）：召回即 touch，热度用于重排 ----------

    def touch_memories(self, doc_ids: List[str]) -> None:
        """召回命中即计数：常想起的更牢固。幂等，失败不影响召回本身。"""
        if not doc_ids:
            return
        now = datetime.now().isoformat(timespec="seconds")
        try:
            with self._lock, self._conn:
                for d in doc_ids:
                    self._conn.execute(
                        "INSERT INTO memory_stats(doc_id, count, last_seen) VALUES(?, 1, ?) "
                        "ON CONFLICT(doc_id) DO UPDATE SET count=count+1, last_seen=?",
                        (d, now, now),
                    )
        except sqlite3.Error:
            pass

    def memory_stats_bulk(self, doc_ids: List[str]) -> dict:
        """批量取热度 {doc_id: {count, last_seen}}，重排用；查不到的键不存在。"""
        if not doc_ids:
            return {}
        marks = ",".join("?" for _ in doc_ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT doc_id, count, last_seen FROM memory_stats WHERE doc_id IN ({marks})",
                list(doc_ids),
            ).fetchall()
        return {r[0]: {"count": r[1], "last_seen": r[2]} for r in rows}

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

    def proactive_after(self, session_id: str, after_iso: str, n: int = 5) -> List[dict]:
        """某时刻之后她主动发的话（N3 轮询通道）：intent='proactive' 的行。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, intent, emotion, mode, created_at FROM chat_log "
                "WHERE session_id = ? AND intent = 'proactive' AND created_at > ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, after_iso, n),
            ).fetchall()
        return [
            {"role": r[0], "text": r[1], "intent": r[2], "emotion": r[3],
             "mode": r[4], "time": r[5]}
            for r in reversed(rows)
        ]

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
