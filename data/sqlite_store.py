"""SQLite 知识库：结构化数据统一进一个库文件（storage/knowledge.db）。

管八张表：
- profile / portrait / relationship / self：一 session 一行，value 存整个 JSON。
  这四类数据都是"整包读、整包写"，JSON 一列最省事，字段以后随便加。
- chat_log：聊天记录按行存（列化），写日记时要捞"今天的对话"，
  按时间和会话查就行，不用把整个 JSON 文件读出来翻。
- affection_history：关系数值账本（S3），只追加不改。
- memory_candidates：LLM 抽取事实的候选池（S4），攒够命中才晋升入档。
- memory_stats：长期记忆的召回热度（S5）。

schema 演进走 `PRAGMA user_version` + `_MIGRATIONS` 线性迁移（J6），不引 alembic：
她的全部记忆都在这一个库文件里，"加新表能过、给老表加一列就让上个月的老库
启动即 OperationalError"是数据安全问题，见 _migrate / _run_migration 的注释。
"""

import json
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Callable, List, Optional

from config.settings import get_settings
from shared.ids import new_turn_id

# 允许当 KV 表用的表名白名单，防呆：写错名字宁可报错也别建出脏表
# self：她自己定下的身份（名字/年龄/城市/自我认知），懒生成冻结（批次0）
_KV_TABLES = ("profile", "portrait", "relationship", "self")

# ------------------------- schema 迁移（J6） -------------------------
#
# 一个批次 = 一个 tuple，把 user_version 往前推 1（第 1 批 0->1，第 2 批 1->2 …）。
# 批次里的每一步：SQL 字符串直接 execute；需要"先查再改"的幂等语句（如 ADD COLUMN）
# 放一个收 conn 的可调用对象，见 _add_column。
#
# 三条纪律（改这里之前先读）：
# 1. **只许往尾部追加批次，不许改已发布的批次**——老库是靠"跑过第 N 批"来认定自己
#    到了版本 N 的；回头改老批次，等于让两种不同形状的库顶着同一个版本号。
# 2. **每批必须幂等**：没有迁移机制的老库（user_version=0）表早就建好了，
#    CREATE 一律带 IF NOT EXISTS；ADD COLUMN 本身不幂等，必须走 _add_column。
# 3. 每批一个事务（_run_migration）：批内任一步失败就整批回滚、版本号停在上一版，
#    下次启动重试——绝不留下"schema 改了一半、版本号却已推进"的库。
_MIGRATIONS: tuple = (
    # ---- version 0 -> 1：初始 schema（自本批发布起冻结，别再往里加东西）----
    (
        """
        CREATE TABLE IF NOT EXISTS profile (
            session_id TEXT PRIMARY KEY,
            value      TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS portrait (
            session_id TEXT PRIMARY KEY,
            value      TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS relationship (
            session_id TEXT PRIMARY KEY,
            value      TEXT NOT NULL
        )
        """,
        # 她自己的身份：名字、年龄、城市、生活、自我认知——全部由她本人
        # 在对话里"第一次说出口"时定下并冻结（懒生成，批次0），代码零预设
        """
        CREATE TABLE IF NOT EXISTS self (
            session_id TEXT PRIMARY KEY,
            value      TEXT NOT NULL
        )
        """,
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
        """,
        # 日记生成按"会话 + 时间段"捞对话，这个索引就是给那一路用的
        "CREATE INDEX IF NOT EXISTS idx_chat_session_time ON chat_log(session_id, created_at)",
        # 关系数值账本（S3）：每一次数值变动都留痕（old/new/delta/reason/原话），
        # 她能回答"你为什么生气"，用户可审计，"最近关系升温/变僵"也从这里聚合
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
        """,
        "CREATE INDEX IF NOT EXISTS idx_ledger_session ON affection_history(session_id, id)",
        # 记忆候选池（S4）：LLM 抽的"事实"先入池，同内容出现 >=2 次才晋升入档
        # ——对 LLM 抽取的记忆"先怀疑、再验证、慢接受"，幻觉不许焊死在档案里
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
        """,
        "CREATE INDEX IF NOT EXISTS idx_cand_session ON memory_candidates(session_id, field)",
        # 记忆生命力（S5）：召回即 touch；常想起的更牢，没人提的自然淡
        """
        CREATE TABLE IF NOT EXISTS memory_stats (
            doc_id    TEXT PRIMARY KEY,
            count     INTEGER NOT NULL DEFAULT 0,
            last_seen TEXT NOT NULL
        )
        """,
    ),
    (
    # ---------------- R15a：提交边界与回执的持久结构（8.11.3） ----------------
    # 技术请求登记：(session_id, request_id) 的持久唯一依据——重试去重、
    # "同 request 不同内容"冲突裁决都查这张表。只登记技术字段，**不存草稿**。
    """
    CREATE TABLE IF NOT EXISTS turn_requests (
        request_id     TEXT PRIMARY KEY,
        session_id     TEXT NOT NULL,
        request_digest TEXT NOT NULL,
        turn_id        TEXT DEFAULT '',
        lifecycle      TEXT NOT NULL DEFAULT 'running',
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_req_session ON turn_requests(session_id, request_id)",
    # 已提交轮回执（CommitReceipt 最小字段）：同 request 重试返回同一回执的
    # 依据；committed_at 处理中不许伪造。turn_id 是跨重启不碰撞的持久键。
    """
    CREATE TABLE IF NOT EXISTS turn_commits (
        turn_id      TEXT PRIMARY KEY,
        session_id   TEXT NOT NULL,
        request_id   TEXT NOT NULL,
        disposition  TEXT NOT NULL DEFAULT 'normal',
        status       TEXT NOT NULL DEFAULT 'committed',
        committed_at TEXT NOT NULL,
        message_ids  TEXT DEFAULT '[]',
        reason_code  TEXT DEFAULT ''
    )
    """,
    # 同一 request 只许有一张回执：重试必须返回原回执，不能生成第二份
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_commit_request ON turn_commits(session_id, request_id)",
    # 派生任务完成标记（R18b 前置结构）：task_key = 源轮+种类+目标版本的
    # 组合唯一键——重复执行/重启重放最多应用一次
    """
    CREATE TABLE IF NOT EXISTS derived_task_done (
        task_key TEXT PRIMARY KEY,
        done_at  TEXT NOT NULL
    )
    """,
    # chat_log 加列：稳定 turn_id、轮分类（normal/degraded/crisis；旧行留空
    # 读作 legacy）、来源审核结果与原因码。旧行保留，旧读接口不受影响。
    lambda c: _add_column(c, "chat_log", "turn_id", "TEXT DEFAULT ''"),
    lambda c: _add_column(c, "chat_log", "disposition", "TEXT DEFAULT ''"),
    lambda c: _add_column(c, "chat_log", "source_review_status", "TEXT DEFAULT ''"),
    lambda c: _add_column(c, "chat_log", "reason_code", "TEXT DEFAULT ''"),
    # 关系账本加列：field 区分账目种类（intimacy 之外的字段**不许**混进
    # old_score/new_score 旧分数列——旧列语义只属于亲密度）；old_value/
    # new_value 是 JSON 序列化的通用值；event_id 供幂等去重。
    lambda c: _add_column(c, "affection_history", "field", "TEXT DEFAULT ''"),
    lambda c: _add_column(c, "affection_history", "old_value", "TEXT DEFAULT ''"),
    lambda c: _add_column(c, "affection_history", "new_value", "TEXT DEFAULT ''"),
    lambda c: _add_column(c, "affection_history", "event_id", "TEXT DEFAULT ''"),
    ),
    (
    # ---------------- R11b/R11c：稳定来源证据与仲裁 keep 复用 ----------------
    # 候选来源：每条候选记录给它投过票的 chat_log 消息 id（JSON 数组）——
    # 同一来源消息重复命中（窗口重叠/重试/重启重放）不再增加 hits；
    # 证据键 = (session, field, normalized_value, source_message_id)。
    # 旧行 source_ids 读作 '[]'（legacy 候选按无来源对待，不回填假装能归属）。
    lambda c: _add_column(c, "memory_candidates", "source_ids", "TEXT DEFAULT '[]'"),
    # 仲裁 keep 复用缓存（R11c）：相同证据集合 + 相同档案版本复用 keep 裁决，
    # 不再付费仲裁；新的独立证据或档案变化可重新仲裁。quotes 存归一化引文集。
    """
    CREATE TABLE IF NOT EXISTS arbitration_keep (
        cache_key  TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        field      TEXT NOT NULL,
        value_key  TEXT NOT NULL,
        old_value  TEXT NOT NULL,
        quotes     TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_keep_session ON arbitration_keep(session_id, field)",
    ),
)

# 当前代码期望的 schema 版本 = 迁移批次数（每批把版本 +1）
_SCHEMA_VERSION = len(_MIGRATIONS)

# new_turn_id 从 shared.ids 导入（R15a 引入，R17b 收口到 shared 层——
# orchestration 的取消门也要用同一生成器，data 层不再自有副本）


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """表在不在。迁移的幂等判断用；表名只来自本文件常量，不接外部输入。"""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _add_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """幂等地给已有表加一列（decl 形如 "TEXT DEFAULT ''"）。

    `ALTER TABLE ... ADD COLUMN` 撞上已存在的列会抛 duplicate column name，
    而"这一列在不在"取决于这个库当年是哪一版代码建的——迁移必须两边都能跑，
    所以先查 PRAGMA table_info 再决定动不动手。

    本批（v0->v1）没有加列，这个 helper 是给下一批用的：将来给 chat_log 加字段时，
    在 _MIGRATIONS 尾部追加 `(lambda c: _add_column(c, "chat_log", "x", "TEXT DEFAULT ''"),)`，
    **不要**去改 v0->v1 那批的 CREATE TABLE。
    """
    if any(str(r[1]) == column for r in conn.execute(f"PRAGMA table_info({table})").fetchall()):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# ------------------- 候选命中的归一化（E6） -------------------

# 行政尾缀，**长的必须排前面**：逐个 endswith 匹配到第一个就剥，
# 若"区"排在"自治区"前面，"内蒙古自治区"会被剥成"内蒙自治"这种四不像。
_ADMIN_SUFFIXES = (
    "特别行政区", "自治区", "自治州", "街道",
    "省", "市", "区", "县", "旗", "镇", "乡", "村",
)


def _to_half_width(text: str) -> str:
    """全角转半角：ASCII 区（U+FF01~U+FF5E）差 0xFEE0，全角空格 U+3000 单独映射。

    中文输入法下"Ｈａｎｇｚｈｏｕ"和"hangzhou"是同一个意思，
    不归一化就永远算两次不同的候选。
    """
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        out.append(chr(code))
    return "".join(out)


def _normalize_candidate(text: str) -> str:
    """把候选内容压成"命中计数用的比对 key"（E6）。

    为什么需要它：candidate_hit 原来按字符串完全相等计数，"杭州"和"杭州市"
    于是各占一行、谁都攒不到 2 次晋升门槛——记忆闸门对同一件事永远不认账，
    表现就是"她说了十遍自己在杭州，档案里还是空的"。

    规则：全角转半角 -> 统一小写 -> 去掉所有空白 -> 从尾部反复剥行政后缀。
    **只用于比对**：存进候选池的仍是她的原话（改原文 = 把她说的话改掉，
    晋升入档时写进 profile 的也就成了被我们修剪过的版本）。

    有意留下的取舍：剥尾缀只看结尾，所以"市"本身是内容一部分的词
    （比如某个叫"××市"的昵称）会被误剥——但两侧剥法一致，命中计数仍然正确，
    代价只是极端情况下两个不同候选被并成一行，比"永远不晋升"轻得多。
    """
    s = re.sub(r"\s+", "", _to_half_width(str(text or "")).lower())
    changed = True
    while changed:
        changed = False
        for suf in _ADMIN_SUFFIXES:
            # len(s) > len(suf)：剥完必须还剩东西，否则单字内容会被剥成空串，
            # 所有空串又会互相命中，把毫不相干的候选并成一行
            if len(s) > len(suf) and s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
                break
    return s


def _candidate_key(text: str) -> str:
    """命中比对 key。归一化后为空（原文是空串或纯空白）时退回原文，
    免得所有空内容互相命中、并成同一行候选。"""
    return _normalize_candidate(text) or str(text or "")


class SQLiteStorage:
    """项目唯一的 SQLite 入口：建库、迁移 schema、读写都从这走。"""

    @staticmethod
    def candidate_key(text: str) -> str:
        """候选命中归一化 key 的公开门面（R11a）。

        capability 层做"同批同字段同值只数一次"的批内去重时，必须用与
        candidate_hit **同一套**归一化（杭州/杭州市算同一件事），否则两套
        口径会互相打架。这是只读纯函数，不走锁。
        """
        return _candidate_key(text)

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
        self._migrate()

    # ---------- schema 迁移（J6）：启动时把落后的批次补齐 ----------

    def _migrate(self) -> None:
        """读 user_version，依次执行落后的迁移批次，每批成功后写回版本号。

        改造前这里只有一堆 CREATE TABLE IF NOT EXISTS：**加新表能过，给老表加一列
        就会让上个月的老库启动即 OperationalError**（老库里没有那一列，
        INSERT/SELECT 指名它就炸）。她的全部记忆在这个库文件里，这是数据安全问题，
        所以迁移必须自动跑、必须幂等、失败必须能停在旧版本上等下次重试。
        """
        with self._lock:
            try:
                current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            except (sqlite3.Error, TypeError, ValueError, IndexError):
                # 版本号读不出来（库损坏/被别的工具改过）：按最老的 0 处理，
                # 反正每批都幂等，重跑一遍只是空操作，比拒绝启动强
                current = 0
            # 全新库（版本 0 且一张表都没有）不值得吭声；老库被本机制接管/升级才报一行
            fresh = current == 0 and not _table_exists(self._conn, "profile")

        if current > _SCHEMA_VERSION:
            # 库比代码新（用户把代码回滚到了旧版本）：不猜也不动它，只告警。
            # 硬跑迁移只会把新库改坏——旧代码认不出新列，但也没资格删。
            print(f"[store] knowledge.db 的 schema 版本是 {current}，比这份代码期望的 "
                  f"{_SCHEMA_VERSION} 新，跳过迁移（是不是回滚了旧版本代码？）")
            return

        for target in range(current + 1, _SCHEMA_VERSION + 1):
            self._run_migration(_MIGRATIONS[target - 1], target)

        if not fresh and current != _SCHEMA_VERSION:
            print(f"[store] knowledge.db schema v{current} -> v{_SCHEMA_VERSION}")

    def _run_migration(self, batch, target_version: int) -> None:
        """跑一批迁移：显式事务 + 失败整批回滚 + 版本号写在同一事务里。

        为什么手写 BEGIN 而不用 `with self._conn`：Python 的 sqlite3 只在
        INSERT/UPDATE/DELETE 前隐式开事务，CREATE TABLE / PRAGMA 都是各自立刻生效的，
        `with self._conn` 根本包不住 DDL——批次跑到一半崩了就会留下
        "半张 schema + 旧版本号"（下次启动重跑还能救）或更糟的"半张 schema + 新版本号"。
        用 BEGIN IMMEDIATE 先拿写锁：WAL 下多个进程同时启动时，避免事务升级死锁。
        """
        with self._lock:
            if self._conn.in_transaction:
                # 防御：迁移只在启动路径跑，正常不该有悬挂事务；真有了就先落地，
                # 否则下面的 BEGIN 会撞 "cannot start a transaction within a transaction"
                self._conn.commit()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for step in batch:
                    if callable(step):
                        step(self._conn)   # 需要"先查再改"的幂等步骤（见 _add_column）
                    else:
                        self._conn.execute(step)
                self._conn.execute(f"PRAGMA user_version = {int(target_version)}")
            except Exception:
                self._conn.rollback()
                raise
            self._conn.execute("COMMIT")

    # ---------- 三张 KV 表：整包 JSON 读 / 整包 JSON 写 ----------

    def _salvage_corrupt(self, table: str, session_id: str, raw: str, consequence: str) -> dict:
        """KV 行的 JSON 坏了：备份原文 + 大声告警 + 返回空 dict（J8）。

        load_kv 和 update_kv **必须共用这一条路径**：以前 load_kv 备份+告警，
        update_kv 却只是静默拿 default 顶上——relationship 行一旦损坏，
        下一次闭包更新就把亲密度/信任无声覆盖成默认值，清零且零痕迹。
        口径统一之后，损坏永远留得下 .corrupt.bak，至少可手工恢复。

        数据救不回来是事实（JSON 都解析不了了），所以这里的职责只有两件：
        留证据 + 让人听见。consequence 用来在告警里说清"这次会怎么样"。

        备份是一次小文件写，只在损坏路径上发生；它不碰 SQLite 也不碰网络，
        调用方在锁内调它没有死锁风险（顺序上还必须是"先备份、再覆盖"）。
        """
        bak_name = f"{table}_{session_id}.corrupt.bak"
        try:
            bak = get_settings().data_dir / bak_name
            bak.parent.mkdir(parents=True, exist_ok=True)
            bak.write_text(raw or "", encoding="utf-8")
        except OSError:
            pass  # 备份写不下去也不能拦住主流程，至少下面这行告警还在
        print(f"[store] {table}.{session_id} 的 JSON 损坏了！原文已备份到 {bak_name}，"
              f"{consequence}——请从备份手工恢复，别让关系数值被当新会话重置")
        return {}

    def load_kv(self, table: str, session_id: str) -> dict:
        """按表名 + 会话读整包 JSON，没有就给空 dict。

        JSON 损坏时不能静默当"第一次聊天"——上层（RelationshipTracker）会把空 dict
        当新会话、用 default_intimacy 重置整段关系。所以走 _salvage_corrupt：
        把损坏原文备份到磁盘并大声警告，数据虽然救不回来，至少有迹可循。
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
            self._salvage_corrupt(table, session_id, row[0], "本次按空数据处理")
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
        # J13：`with self._conn` 取代手动 commit——INSERT 失败（磁盘满/库锁）时
        # 上下文管理器会 rollback，手动 commit 的写法则把事务悬挂在那儿，
        # 下一个持锁的写方会连上这个半截事务，与本文件其余方法的风格也不一致。
        with self._lock, self._conn:
            row = self._conn.execute(
                f"SELECT value FROM {table} WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row:
                try:
                    current = json.loads(row[0])
                except json.JSONDecodeError:
                    # J8：损坏不再静默重置。先备份+告警（与 load_kv 同一条路径），
                    # 再按 default 重算——备份必须发生在下面那句覆盖写之前，
                    # 否则亲密度/信任被清零之后连一点痕迹都找不到。
                    self._salvage_corrupt(
                        table, session_id, row[0], "本次闭包将在默认值上重算并覆盖这一行"
                    )
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
        return updated

    # ---------- R15c：一轮提交的事务门面（8.11.3） ----------

    def commit_turn(self, *, session_id: str, turn_id: str, request_id: str,
                    request_digest: str, disposition: str = "normal",
                    source_review_status: str = "accepted", reason_code: str = "",
                    user_text: str = "", assistant_text: str = "",
                    intent: str = "", emotion: str = "", mode: str = "text",
                    relation_fn: Optional[Callable] = None,
                    relation_ledger: str = "",
                    ledger_event_id: str = "",
                    ledger: Optional[dict] = None,
                    created_at: Optional[str] = None) -> dict:
        """一轮提交 = **一个短事务**：请求登记 → user/assistant 正式记录 →
        关系原子更新 → 账本 → 提交回执。任何一步失败整体回滚——无半轮、
        无无账的关系变化、无回执；同 request 重试返回同一回执（不重复计数）；
        同 request 不同 digest 明确 conflict。

        - relation_fn：**调用方注入的纯计算**（R15b 的
          compute_relationship_update 偏应用）——本层不做人格计算，也不
          import capability；事务内从 relationship 表读**最新**现态传入，
          禁止先读快照再覆盖；
        - ledger：{field, old, new, reason, quote, event_id, old_value,
          new_value}，field != intimacy 的行不占旧分数列（R15a 结构）；
        - 关系/账本可选： Crisis 轮等无普通增长的轮次传 None 即可；
        - 用户转写为空不造假 user 行，助手正文为空不造假 assistant 行。

        返回回执 dict：status ∈ committed / already_committed / conflict /
        processing / failed。本方法只提供**低层能力**——路由接入与生命周期
        推进是 R17 的事，接入前不能宣传所有路由已遵守契约。
        """
        now = created_at or datetime.now().isoformat(timespec="seconds")
        with self._lock:
            if self._conn.in_transaction:
                # 防御：提交不允许在悬挂事务里跑（同 _run_migration 的处理）
                self._conn.commit()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # ① 请求登记去重：同 request 已登记过 → 查回执/冲突/在途
                row = self._conn.execute(
                    "SELECT request_digest FROM turn_requests "
                    "WHERE request_id = ? AND session_id = ?",
                    (request_id, session_id),
                ).fetchone()
                if row is not None:
                    if row[0] != request_digest:
                        self._conn.rollback()
                        return {"status": "conflict", "turn_id": turn_id,
                                "session_id": session_id, "request_id": request_id,
                                "reason_code": "request_digest_mismatch"}
                    rc = self._conn.execute(
                        "SELECT turn_id, committed_at, disposition, message_ids "
                        "FROM turn_commits WHERE session_id = ? AND request_id = ?",
                        (session_id, request_id),
                    ).fetchone()
                    self._conn.rollback()
                    if rc is not None:
                        # 同 request 重试：返回**原回执**，不重复计数
                        import json as _json
                        return {"status": "already_committed", "turn_id": rc[0],
                                "session_id": session_id, "request_id": request_id,
                                "disposition": rc[2], "committed_at": rc[1],
                                "message_ids": _json.loads(rc[3] or "[]")}
                    return {"status": "processing", "turn_id": turn_id,
                            "session_id": session_id, "request_id": request_id}

                # ② 请求登记（lifecycle=committing；随本事务一起落定）
                self._conn.execute(
                    "INSERT INTO turn_requests "
                    "(request_id, session_id, request_digest, turn_id, lifecycle, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, 'committing', ?, ?)",
                    (request_id, session_id, request_digest, turn_id, now, now),
                )
                # ③ 正式记录：user / assistant（空转写不造假行）
                message_ids = []
                if user_text:
                    # R17c：user 行 mode 用本轮参数——文字轮同为 "text" 行为不变，
                    # 语音轮（ASR 转写）不再被冒充成打字。
                    cur = self._conn.execute(
                        "INSERT INTO chat_log (session_id, role, content, intent, emotion, "
                        "mode, created_at, turn_id, disposition, source_review_status, reason_code) "
                        "VALUES (?, 'user', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (session_id, user_text, intent, emotion, mode, now,
                         turn_id, disposition, source_review_status, reason_code),
                    )
                    message_ids.append(cur.lastrowid)
                if assistant_text:
                    cur = self._conn.execute(
                        "INSERT INTO chat_log (session_id, role, content, intent, emotion, "
                        "mode, created_at, turn_id, disposition, source_review_status, reason_code) "
                        "VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (session_id, assistant_text, intent, emotion, mode, now,
                         turn_id, disposition, source_review_status, reason_code),
                    )
                    message_ids.append(cur.lastrowid)
                # ④ 关系：事务内读**最新**现态 → 调用方注入的纯计算 → 写回；
                # relation_ledger 非空时自动记 intimacy 账本（值来自同一事务内的
                # 前后差，与 R15b 纯计算共用现态，不先读快照）
                if relation_fn is not None:
                    rel_row = self._conn.execute(
                        "SELECT value FROM relationship WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                    rel = json.loads(rel_row[0]) if rel_row else {}
                    old_intimacy = float((rel or {}).get("intimacy") or 0)
                    new_rel = relation_fn(rel)
                    new_intimacy = float((new_rel or {}).get("intimacy") or 0)
                    self._conn.execute(
                        "INSERT INTO relationship (session_id, value) VALUES (?, ?) "
                        "ON CONFLICT(session_id) DO UPDATE SET value = excluded.value",
                        (session_id, json.dumps(new_rel, ensure_ascii=False)),
                    )
                    if relation_ledger:
                        self._conn.execute(
                            "INSERT INTO affection_history "
                            "(session_id, old_score, new_score, delta, reason, source_quote, "
                            "created_at, field, old_value, new_value, event_id) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, 'intimacy', ?, ?, ?)",
                            (session_id, old_intimacy, new_intimacy,
                             round(new_intimacy - old_intimacy, 4),
                             relation_ledger[:150], (user_text or "")[:150], now,
                             json.dumps(old_intimacy), json.dumps(new_intimacy),
                             ledger_event_id),
                        )
                # ⑤ 账本（可选；field != intimacy 不占旧分数列，R15a 结构）
                if ledger:
                    fld = ledger.get("field", "intimacy")
                    is_intimacy = fld == "intimacy"
                    try:
                        old_s = float(ledger["old"]) if is_intimacy else 0.0
                        new_s = float(ledger["new"]) if is_intimacy else 0.0
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"账本参数不合法（field={fld}）: {exc}") from exc
                    self._conn.execute(
                        "INSERT INTO affection_history "
                        "(session_id, old_score, new_score, delta, reason, source_quote, "
                        "created_at, field, old_value, new_value, event_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (session_id, old_s, new_s, round(new_s - old_s, 4),
                         (ledger.get("reason") or "")[:150],
                         (ledger.get("quote") or "")[:150], now, fld,
                         json.dumps(ledger.get("old_value"), ensure_ascii=False)[:500],
                         json.dumps(ledger.get("new_value"), ensure_ascii=False)[:500],
                         ledger.get("event_id", "")),
                    )
                # ⑥ 回执：COMMIT 成功才算正式历史（8.11.3 原子边界 3）
                self._conn.execute(
                    "INSERT INTO turn_commits "
                    "(turn_id, session_id, request_id, disposition, status, committed_at, "
                    "message_ids, reason_code) VALUES (?, ?, ?, ?, 'committed', ?, ?, ?)",
                    (turn_id, session_id, request_id, disposition, now,
                     json.dumps(message_ids), reason_code),
                )
                self._conn.execute("COMMIT")
            except Exception as exc:
                # 任何一步失败整体回滚：无半轮、无无账的关系变化、无回执
                self._conn.rollback()
                print(f"[commit] 一轮提交失败，已整体回滚: {type(exc).__name__}: {exc}")
                return {"status": "failed", "turn_id": turn_id,
                        "session_id": session_id, "request_id": request_id,
                        "reason_code": "commit_failed", "detail": str(exc)}
            return {"status": "committed", "turn_id": turn_id,
                    "session_id": session_id, "request_id": request_id,
                    "disposition": disposition, "committed_at": now,
                    "message_ids": message_ids}

    def get_commit_receipt(self, session_id: str, request_id: str) -> Optional[dict]:
        """按 (session, request) 查已提交回执；没有返回 None。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT turn_id, disposition, status, committed_at, message_ids, reason_code "
                "FROM turn_commits WHERE session_id = ? AND request_id = ?",
                (session_id, request_id),
            ).fetchone()
        if row is None:
            return None
        return {"status": row[2], "turn_id": row[0], "session_id": session_id,
                "request_id": request_id, "disposition": row[1],
                "committed_at": row[3], "message_ids": json.loads(row[4] or "[]"),
                "reason_code": row[5] or ""}

    def get_turn_messages(self, session_id: str, turn_id: str) -> List[dict]:
        """按 turn_id 取该轮的全部正式记录（role/content），旧 -> 新。

        R17b 重试回放用：同 request 的已提交轮直接回放已存正文，不再生成。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM chat_log "
                "WHERE session_id = ? AND turn_id = ? ORDER BY id",
                (session_id, turn_id),
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    # ---------- 关系数值账本（S3）：追加写 + 近 N 条读 ----------

    def append_ledger(self, session_id: str, old: float, new: float,
                      reason: str = "", source_quote: str = "",
                      field: str = "intimacy",
                      old_value=None, new_value=None, event_id: str = "") -> None:
        """记一笔数值变动。账本是审计面不是数据面：写失败只告警，不回滚数值。

        R15a 扩展：field 区分账目种类（默认 intimacy 兼容旧调用）——**非
        intimacy 字段的行不许占用 old_score/new_score 旧分数列**（旧列语义
        只属于亲密度，旧读方按数值列聚合会被污染），真实值以 JSON 记入
        old_value/new_value；event_id 供派生幂等去重（R18b 接线）。
        """
        # R15a：旧分数列只属于 intimacy——非 intimacy 字段的 old/new 可能根本
        # 不是数字（心情串、JSON 对象），不许也不需要 float 转换
        is_intimacy = field == "intimacy"
        if is_intimacy:
            delta = round(float(new) - float(old), 4)
            score_old, score_new, score_delta = float(old), float(new), delta
        else:
            score_old = score_new = score_delta = 0.0
        ov = json.dumps(old_value if old_value is not None else old, ensure_ascii=False)
        nv = json.dumps(new_value if new_value is not None else new, ensure_ascii=False)
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO affection_history "
                    "(session_id, old_score, new_score, delta, reason, source_quote, created_at, "
                    "field, old_value, new_value, event_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (session_id, score_old, score_new, score_delta,
                     reason[:150], (source_quote or "")[:150],
                     datetime.now().isoformat(timespec="seconds"),
                     field, ov[:500], nv[:500], event_id),
                )
        except sqlite3.Error as exc:
            print(f"[ledger] 账本写入失败（数值本身已生效）: {exc}")

    def recent_ledger(self, session_id: str, n: int = 8) -> List[dict]:
        """最近 n 条账本，旧 -> 新。S10 氛围线从这里聚合趋势。

        R15a：新增 field/old_value/new_value/event_id 键（additive）；旧键
        语义不变——field 非 intimacy 的行，old/new/delta 旧键恒为 0，消费方
        需按 field 过滤（现有消费方只读 intimacy 语义，不受影响）。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT old_score, new_score, delta, reason, source_quote, created_at, "
                "field, old_value, new_value, event_id "
                "FROM affection_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, n),
            ).fetchall()
        result = []
        for r in reversed(rows):
            entry = {
                "old": r[0], "new": r[1], "delta": r[2], "reason": r[3],
                "quote": r[4], "time": r[5],
                "field": r[6] or "intimacy",
                "old_value": r[7], "new_value": r[8], "event_id": r[9] or "",
            }
            result.append(entry)
        return result

    # ---------- 记忆候选池（S4）：入池计数 / 晋升标记 / 过期清理 ----------

    def candidate_hit(self, session_id: str, field: str, content: str,
                      quote: str = "", source_message_id: str = "") -> tuple:
        """同内容候选命中一次：已在池则 hits+1，不在则入池。返回 (hits, id)。

        E6：命中判定走归一化 key（_normalize_candidate），**入池的 content 仍是原文**
        ——"杭州市"和"杭州"要算同一件事，但她说出口的原话一个字都不许改。

        为什么不用 SQL 直接比 content，而是把该 (session, field) 下未晋升的候选
        全捞出来在 Python 侧比：剥行政尾缀/全半角这类规则 SQL 表达不了；
        而单个 field 的候选本来就被 prune_candidates 的 14 天过期压得很小
        （个位数量级），全捞不心疼。

        R11b：source_message_id 是本次证据的来源消息 id（chat_log.id）。同一来源
        消息重复命中（窗口重叠/重试/重启重放）**不增加 hits**——证据键
        (session, field, normalized_value, source_message_id) 天然去重；来源 id
        记进候选的 source_ids，仲裁与审计都能对得上真实消息。source_message_id
        为空时按旧行为计数（调用方 gate 已把"无法归属"的抽取挡在门外）。
        """
        key = _candidate_key(content)
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT id, hits, content, source_ids FROM memory_candidates "
                "WHERE session_id=? AND field=? AND promoted=0",
                (session_id, field),
            ).fetchall()
            for cand_id, hits, stored, source_ids_json in rows:
                if _candidate_key(stored) != key:
                    continue
                try:
                    source_ids = json.loads(source_ids_json or "[]")
                    if not isinstance(source_ids, list):
                        source_ids = []
                except ValueError:
                    source_ids = []
                if source_message_id and source_message_id in source_ids:
                    # R11b：同一来源消息的重复投票不算新证据
                    return hits, cand_id
                if source_message_id:
                    source_ids.append(source_message_id)
                self._conn.execute(
                    "UPDATE memory_candidates SET hits=hits+1, source_ids=? WHERE id=?",
                    (json.dumps(source_ids, ensure_ascii=False), cand_id),
                )
                return hits + 1, cand_id
            cur = self._conn.execute(
                "INSERT INTO memory_candidates "
                "(session_id, field, content, quote, hits, first_seen, source_ids) "
                "VALUES (?, ?, ?, ?, 1, ?, ?)",
                (session_id, field, content, quote[:200], now,
                 json.dumps([source_message_id] if source_message_id else [],
                            ensure_ascii=False)),
            )
            return 1, cur.lastrowid

    # ------------------- R11c：仲裁 keep 复用缓存 -------------------
    #
    # keep 是"仲裁员认定口误/玩笑"的裁决。E2 修复后被否决的候选会标晋升，但
    # 用户每把同一句话再说两遍，就会攒出一条新候选、再烧一次仲裁 LLM——同样的
    # 证据 + 同样的档案现态，答案不会变。缓存键 = (session, field, 值归一化)，
    # 命中还要求：档案现态没变 + 本次引文在已裁决引文集里。新的独立证据
    # （没见过的引文）或档案变化都会重新仲裁；update 裁决直接清缓存。
    # 这不是永久拉黑：用户以后真搬去被否过的城市，换个说法（新证据）即可重裁。

    _KEEP_CACHE_TTL_DAYS = 30

    @staticmethod
    def _keep_cache_key(session_id: str, field: str, value_key: str) -> str:
        return f"{session_id}|{field}|{value_key}"

    def keep_cache_lookup(self, session_id: str, field: str, value_key: str,
                          old_value: str, quote_key: str) -> bool:
        """查 keep 缓存。命中条件：键在、档案现态一致、引文在已裁决集合里。"""
        import json as _json

        key = self._keep_cache_key(session_id, field, value_key)
        cutoff = (datetime.now() - timedelta(days=self._KEEP_CACHE_TTL_DAYS)
                  ).isoformat(timespec="seconds")
        with self._lock, self._conn:
            # 懒清理：过期缓存顺手删（不挂巡检，代价最小的收口）
            self._conn.execute("DELETE FROM arbitration_keep WHERE created_at < ?", (cutoff,))
            row = self._conn.execute(
                "SELECT old_value, quotes FROM arbitration_keep WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None or row[0] != old_value:
            return False  # 档案变了（或没缓存过）：证据的语境不同了，重裁
        try:
            quotes = _json.loads(row[1] or "[]")
        except ValueError:
            quotes = []
        return quote_key in quotes

    def keep_cache_store(self, session_id: str, field: str, value_key: str,
                         old_value: str, quote_key: str) -> None:
        """记一次 keep 裁决（或往已裁决引文集里添一条新证据的裁决结果）。"""
        import json as _json

        key = self._keep_cache_key(session_id, field, value_key)
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT old_value, quotes FROM arbitration_keep WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is not None and row[0] == old_value:
                try:
                    quotes = _json.loads(row[1] or "[]")
                except ValueError:
                    quotes = []
                if quote_key not in quotes:
                    quotes.append(quote_key)
                self._conn.execute(
                    "UPDATE arbitration_keep SET quotes = ?, created_at = ? WHERE cache_key = ?",
                    (_json.dumps(quotes, ensure_ascii=False), now, key),
                )
            else:
                # 档案已变化（或首次）：旧的引文集不作数，按当前语境重开
                self._conn.execute(
                    "INSERT INTO arbitration_keep "
                    "(cache_key, session_id, field, value_key, old_value, quotes, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(cache_key) DO UPDATE SET old_value = excluded.old_value, "
                    "quotes = excluded.quotes, created_at = excluded.created_at",
                    (key, session_id, field, value_key, old_value,
                     _json.dumps([quote_key], ensure_ascii=False), now),
                )

    def keep_cache_drop(self, session_id: str, field: str, value_key: str) -> None:
        """update 裁决/档案变更后清缓存：同值再出现要重新仲裁。"""
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM arbitration_keep WHERE cache_key = ?",
                (self._keep_cache_key(session_id, field, value_key),),
            )

    # ------------------- R18b：派生任务完成标记的原子裁决 -------------------
    #
    # derived_task_done 的 task_key 主键就是原子闸：INSERT OR IGNORE 谁抢到
    # rowcount=1 谁就是本次的执行者——并发领取由存储裁决，不靠进程内约定。
    # 顺序语义：**先领取、再应用、失败退回**。应用成功前标记已在，进程死在
    # 中间只会丢一次学习任务（安全侧），绝不会"状态已改、标记没写"地重复
    # 改状态；跨重启的可靠排队与租约是 R19 pending_writes 的活，这里不越位。
    #
    # 应用失败退回标记的原因：领取不等于完成，失败了下次（重试/重放）还该能跑。

    def claim_task(self, task_key: str) -> bool:
        """原子领取：返回 True 表示本次调用拥有执行权，False=已被应用过。"""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO derived_task_done (task_key, done_at) VALUES (?, ?)",
                (task_key, datetime.now().isoformat(timespec="seconds")),
            )
            return cur.rowcount > 0

    def release_task(self, task_key: str) -> None:
        """退回领取标记（应用失败时）：下次重试/重放还能执行。"""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM derived_task_done WHERE task_key = ?", (task_key,))

    def mark_candidate_promoted(self, cand_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE memory_candidates SET promoted=1 WHERE id=?", (cand_id,)
            )

    def prune_candidates(self, days: int = 14) -> int:
        """入池超过 N 天还没晋升的候选：放弃（证据不足）。返回清理条数。

        只删 promoted=0 的行是有意的：已晋升的候选是"她为什么会有这条档案"的证据，
        留档不删；而 candidate_hit 只匹配 promoted=0，所以留着也不会再被命中，
        不影响正确性。代价是这张表仍会随晋升缓慢增长——真要控体积，
        应该另加一条"已晋升且超过 N 天"的清理规则，而不是把这句改成无条件 DELETE。
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM memory_candidates WHERE promoted=0 AND first_seen < ?",
                (cutoff,),
            )
            return cur.rowcount

    # ---------- 修剪（J9）：按 created_at 删旧行，只给方法，接线由调用方决定 ----------
    #
    # 有意做成"纯能力、不自带定时器"：什么时候删、删多久以前的，是产品决定
    # （配置项 + 巡检线程），不是存储层的决定。这里只保证调用即生效、返回删了几条。

    def prune_chat_log(self, keep_days: int = 90) -> int:
        """删掉 keep_days 天之前的聊天记录，返回删除条数。

        chat_log 是**原始对话**：蒸馏后的长期记忆在 Chroma 里，删旧行不等于她忘了，
        只是不再能逐字回放那么久以前的话。真正的消费方回看都不长
        （写日记取当天、小时分布取 14 天、恢复短期记忆取十来条），默认 90 天余量很足。
        但每句话一行、只增不减，不修剪就是慢性膨胀——库越大，启动和查询都越慢。
        """
        return self._prune_older_than("chat_log", keep_days)

    def prune_ledger(self, keep_days: int = 365) -> int:
        """删掉 keep_days 天之前的关系账本（affection_history），返回删除条数。

        账本是审计面不是数据面：亲密度/信任的**当前值**在 relationship 那一行里，
        删旧账只是丢掉"很久以前那次为什么变"的可追溯性，比删对话更能忍，
        所以默认留一整年（"最近关系升温/变僵"的聚合也只看近期）。
        """
        return self._prune_older_than("affection_history", keep_days)

    def _prune_older_than(self, table: str, keep_days: int) -> int:
        """按 created_at 删旧行的公共实现（两张表时间列同名，SQL 只差表名）。

        keep_days 钳到 >=1：0 或负数会让 cutoff 跑到未来、把整张表删空——
        那是"清空"不是"修剪"，接线方把配置读错成 0 时宁可少删，
        也别一把抹掉她的全部对话史。
        created_at 为空串的脏行也会被删掉：它本来就被任何时间段查询漏掉，留着没用。
        """
        days = max(1, int(keep_days))
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with self._lock, self._conn:
            cur = self._conn.execute(f"DELETE FROM {table} WHERE created_at < ?", (cutoff,))
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

    def get_recent_chat(self, session_id: str, n: int = 10,
                        learnable: bool = False) -> List[dict]:
        """最近 n 条对话，按时间正序返回（老 -> 新），给恢复短期记忆用。

        R14c：learnable=True 只取**可学习的已提交正常轮**（disposition='normal'，
        或旧记录分类字段为空——legacy 行按原样对待，不假装能自动辨认旧污染）。
        身份/画像/事实提炼的学习素材必须走这个口子；聊天展示与短期记忆恢复
        用默认全量（降级轮的兜底正文是真实发布过的）。
        """
        with self._lock:
            sql = ("SELECT id, role, content, intent, emotion, mode, created_at FROM chat_log "
                   "WHERE session_id = ?")
            params: list = [session_id]
            if learnable:
                sql += (" AND (disposition = 'normal' OR disposition IS NULL "
                        "OR disposition = '')")
            sql += " ORDER BY id DESC LIMIT ?"
            params.append(n)
            rows = self._conn.execute(sql, params).fetchall()
        # 倒序取的，翻回来才是对话本来的顺序
        return [
            {
                "id": r[0],  # R11b：chat_log 行 id——证据归属的真实来源键
                "role": r[1],
                "text": r[2],
                "intent": r[3],
                "emotion": r[4],
                "mode": r[5],
                "time": r[6],
            }
            for r in reversed(rows)
        ]

    def get_chats_between(
        self, session_id: str, start_iso: str, end_iso: str
    ) -> List[dict]:
        """捞某个时间段内的对话（含头不含尾），写日记时取"今天聊了啥"用。"""
        with self._lock:
            # R14c：带出 disposition——日记把降级轮当"回复失败"的客观记录，
            # 不能当成她真的说了一段有意义的话。
            rows = self._conn.execute(
                "SELECT role, content, intent, emotion, mode, created_at, disposition "
                "FROM chat_log "
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
                "disposition": r[6] or "",
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
    """懒加载单例：第一次用时建库 + 跑迁移（_migrate），之后一直用同一份连接。

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
