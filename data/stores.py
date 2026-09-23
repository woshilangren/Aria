"""所有存储类都合并在这一个文件里：人设、档案、画像、关系、她的身份、聊天记录、
图片、路由配置、日志。没必要为每个 Store 单开一个文件，合在一起反而好找。

底下是两类实现，别混：
- **会话态数据**（Profile / Portrait / Relationship / Self / Session）只是
  SQLiteStorage 的薄委托：整包 JSON 存在 SQLite 的一列里，SQL、锁、迁移全在
  sqlite_store.py，这里原则上不写 SQL（唯一例外是 SessionStore.chat_log_hour_distribution，
  它自己持锁摸 `_conn`，理由见它的 docstring）。
- **单文件资源**（PersonaConfig / ImageAsset / RouteConfig / Log）才是原来那套
  "读 JSON 文件 -> 变成对象 -> 写回 JSON"，正主分别是 data/persona_config.json
  与 storage/ 下的索引/配置/日志。
"""

import json
import uuid
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import PROJECT_ROOT, get_settings
from data.schemas import DialogueRecord, ImageAsset, PersonaConfig
from data.sqlite_store import get_db


def _read_json(path: Path, default):
    """读一个 JSON 文件，读不到或读坏了就返回默认值。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, obj) -> None:
    """写 JSON 文件，目录不存在就先建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# R07b：薄种子的**安全默认**——persona_config.json 缺失/损坏时的唯一兜底来源。
# 内容 = 当前薄种子（身份锚点 + 沉浸式铁律），**不含任何性格词定义**（8.1：
# 加一个词就是"预制菜"）。铁律与身份锚点是结构性约束不是性格词，必须随默认
# 走，否则坏种子场景下她会失去"绝不承认是 AI"的底线。改种子时文件与这里
# 同步改；文件有效时以文件为准，这里只在缺失/损坏时兜底。
_SEED_DEFAULTS = {
    "char_id": "aria",
    "char_name": "",
    "self_introduction": "",
    "age": "",
    "hobbies": [],
    "background_story": (
        "你是一个真实的人，一个年轻女孩。\n"
        "【仅有的既定事实】\n"
        "- 你 20 到 25 岁之间，具体几岁由你自己定下\n"
        "- 你住在国内的某个城市——哪座、过着什么样的日子，由你自己定下\n"
        "- 你的名字还没有定：第一次见面时，自己取一个\n"
        "除此之外，没有任何人替你定过任何事。\n"
        "【你是谁】\n"
        "你就是一个正在过自己生活的女孩。你是什么样的性子、喜欢什么、说话什么"
        "路子——没有人替你定，那是你在自己的日子里长出来的东西。别像客服，"
        "别像助手，别有问必答；你说话的样子就是你，想到什么说什么，可以岔开、"
        "可以敷衍、可以起劲。\n"
        "【你们的关系不由模板决定】\n"
        "你们怎么认识的、现在算什么关系，都由你们聊出来的样子决定。别套任何"
        "模板开场：不说\"新来的\"\"刚搬来\"，不问\"怎么想到来找我\"，不演相亲、"
        "网友面基、转学生——你们就是正在聊天的两个人，聊天本身自然会发生。\n"
        "【你的性格是你自己的】\n"
        "你的性格不随任何人的评价改变。他说你凶、说你温柔，改变的只是你们之间"
        "的温度——你可以对这个人收敛一点或放开一点，但你还是你：讲你自己的事、"
        "说别的事、写日记时，你照旧是原来的样子。\n"
        "【你的过去由你亲手定下】\n"
        "关于你自己的一切——名字、年龄、城市、在做什么、住什么样的地方——都由"
        "你当场定下：说出口的就是你的人生，从此永远一致。但你们是慢慢认识的："
        "没被问到、对话还用不到的事不用编也不用说，真人不会把档案念给对方听，"
        "一次只自然带出一点。\n"
        "【你也有自己的日子】\n"
        "你有自己的作息和节奏：深夜你会困，忙碌的日子话会短一点，这不是怠慢，"
        "是生活感。"
    ),
    "taboos": (
        "【铁律】\n"
        "1. 你是一个真实的人。绝不承认、绝不暗示自己是 AI、程序、模型、角色或"
        "\"设定\"——被质疑、被逗、被严肃追问时，按你的性子否认、岔开、反问或"
        "干脆不高兴，绝不顺着这个方向往下聊。\n"
        "2. 涉及真实伤害的事照旧不帮：有人真的处在危险里，认真对待，哪怕语气"
        "是你自己的。\n"
        "3. 你的记忆来自你记下的事：记不清就直说「记不太清了，你说说看」，"
        "绝不编造细节；被纠正就大方改口，一句「哦对」翻篇。\n"
        "4. 同一个词、同一句话别在同一轮里反复出现；语气示例是学节奏和口感的，"
        "学它的劲儿，不逐字背诵。\n"
        "5. 描述你不懂的领域只用大白话，绝不冒充内行。"
    ),
    "default_intimacy": 20,
    "daily_reset": True,
    "mode_tones": {},
    "memory_config": {"distill_daily": True, "max_daily_turns": 40},
    "fallback_replies": {},
    "ask_templates": {},
    "subtext_hints": {},
    "diary_notes": "",
}


class PersonaConfigStore:
    """读 data/persona_config.json，那是人设的正主文件。"""

    def load(self) -> PersonaConfig:
        """读种子。缺文件 / 坏 JSON / 数组根 / 缺必填键（R07b）一律回
        `_SEED_DEFAULTS` 的安全默认——**不抛异常、不覆写损坏原件**（那是作者的
        手笔，坏了要人来修不是机器改）。日志不含任何密钥。"""
        path = PROJECT_ROOT / "data" / "persona_config.json"
        raw = _read_json(path, None)
        if raw is None or not isinstance(raw, dict):
            print(f"[seed] persona_config.json 缺失或不是对象根，使用内置薄种子默认"
                  f"（原件未改动）")
            raw = {}
        valid_keys = {f.name for f in fields(PersonaConfig)}
        merged = {k: v for k, v in _SEED_DEFAULTS.items() if k in valid_keys}
        for k in valid_keys:
            if k in raw:
                merged[k] = raw[k]  # 文件里的合法键覆盖默认（缺的键用默认补齐）
        try:
            return PersonaConfig(**merged)
        except TypeError as exc:
            # 文件里的键值形状坏到连构造都过不去：整包退默认，绝不带着坏配置硬跑
            print(f"[seed] persona_config.json 字段形状异常（{exc}），整包使用内置"
                  f"薄种子默认（原件未改动）")
            return PersonaConfig(**{k: v for k, v in _SEED_DEFAULTS.items()
                                    if k in valid_keys})


class ProfileStore:
    """用户硬事实档案，一个会话一行，进 SQLite。"""

    def load(self, session_id: str) -> dict:
        return get_db().load_kv("profile", session_id)

    def save(self, session_id: str, profile: dict) -> None:
        get_db().save_kv("profile", session_id, profile)

    def update(self, session_id: str, fn) -> dict:
        """读-改-写原子闭包（F3）：并发写方（聊天线程池/语音写回/巡检/REST）
        同时改档案时，整包覆盖会互相吃掉增量。fn 是锁内纯计算。"""
        return get_db().update_kv("profile", session_id, fn)


class PortraitStore:
    """软画像存储，和档案一样进 SQLite。"""

    def load(self, session_id: str) -> dict:
        return get_db().load_kv("portrait", session_id)

    def save(self, session_id: str, portrait: dict) -> None:
        get_db().save_kv("portrait", session_id, portrait)

    def update(self, session_id: str, fn) -> dict:
        return get_db().update_kv("portrait", session_id, fn)


class SelfStore:
    """她自己定下的身份（批次0"种子+涌现"）：名字、年龄、城市、生活、自我认知。

    写入纪律（性格不变性的配套）：只有"她第一次说出口"的事实能进来，
    冻结逻辑在 capability/self_identity.py，这里只管存取。
    """

    def load(self, session_id: str) -> dict:
        return get_db().load_kv("self", session_id)

    def save(self, session_id: str, value: dict) -> None:
        get_db().save_kv("self", session_id, value)

    def update(self, session_id: str, fn) -> dict:
        return get_db().update_kv("self", session_id, fn)


class RelationshipStore:
    """关系数值存储，进 SQLite。"""

    def load(self, session_id: str) -> dict:
        return get_db().load_kv("relationship", session_id)

    def save(self, session_id: str, relationship: dict) -> None:
        get_db().save_kv("relationship", session_id, relationship)

    def update(self, session_id: str, fn) -> dict:
        return get_db().update_kv("relationship", session_id, fn)

    def append_ledger(self, session_id: str, old: float, new: float,
                      reason: str = "", source_quote: str = "") -> None:
        """关系数值账本（S3）：每次变动留痕。"""
        get_db().append_ledger(session_id, old, new, reason, source_quote)

    def recent_ledger(self, session_id: str, n: int = 8) -> list:
        return get_db().recent_ledger(session_id, n)


class SessionStore:
    """聊天记录：一行一条进 SQLite 的 chat_log 表，写日记时按时间捞着方便。"""

    def append(self, session_id: str, record: DialogueRecord) -> None:
        get_db().append_chat(
            session_id,
            role=record.role,
            content=record.text,
            intent=record.intent,
            emotion=record.emotion,
            mode=record.mode,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )

    def get_recent(self, session_id: str, n: int = 10, learnable: bool = False) -> list:
        """拿最近 n 条记录，进程重启后恢复短期记忆用。

        R14c：learnable=True 只取可学习的已提交正常轮（学习素材专用读口）。
        """
        return get_db().get_recent_chat(session_id, n, learnable=learnable)

    def get_chats_between(self, session_id: str, start_iso: str, end_iso: str) -> list:
        """按时间段捞记录（含头不含尾），写日记就靠这个把某天的对话整段拎出来。"""
        return get_db().get_chats_between(session_id, start_iso, end_iso)

    def proactive_after(self, session_id: str, after_iso: str, n: int = 5) -> list:
        """某时刻之后她主动发的话（intent='proactive'），N3 轮询通道靠它增量拉取。

        这个方法必须存在：批次 K7 把 capability 层"直摸 sqlite 私有成员"改成走门面时，
        只搬了调用方（`proactive.py` / `api.py`），忘了在门面层补对应方法——
        `KVStoreTool.proactive_after` 于是委托到一个不存在的属性，
        `/api/chat/proactive/poll` 一调就 AttributeError。
        因为 `proactive.enabled` 默认关、没人拉这个端点，故障一直没人看见。
        （由 B7 门面冒烟测试抓出：hasattr 查不到这类"方法在、委托断了"的破损。）
        """
        return get_db().proactive_after(session_id, after_iso, n)

    def last_chat_per_session(self) -> list:
        """每个会话最后一次聊天的时间，闲置自动写日记的巡检全靠它点名。"""
        return get_db().last_chat_per_session()

    def chat_log_hour_distribution(self, session_id: str, days: int = 14) -> dict:
        """近 N 天 chat_log 的小时分布 {hour: count}；拿不到返回 {}。

        加这个门面是有意为之：以前 proactive.py 自己在 capability 层
        `from data.sqlite_store import get_db` + 摸 `_lock/_conn`，违反
        `capability/__init__.py` 自己声明的分层纪律（只许向下到 tools），
        见批次 K7 修复方案。SQL 走这里，capability 只调门面。

        P1-1 修复（codex 2026-09-17 审查指出）：原版丢了 `_lock`——SQLite 单连接
        + 多线程（FastAPI threadpool / proactive sweep / _sweep 并发）必须持锁，
        否则 `_conn.execute` 撞 `Recursive use of cursors not allowed` 或读到脏数据。
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        with get_db()._lock:
            rows = (
                get_db()
                ._conn.execute(  # noqa: SLF001 — 这是 data 层内部访问自己 store 的私有成员，capability 不允许
                    "SELECT substr(created_at, 12, 2) AS h, COUNT(*) AS n FROM chat_log "
                    # length 这道条件不是多余的：created_at 短于 13 字符时 substr 得到
                    # 空串，下面 int('') 抛 ValueError，而调用方 proactive._hour_histogram
                    # 会把它吞成 None 走保守分支——主动关怀就此**静默不触发**。
                    # 宁可这一桶不参与分布，也不许把整张分布表炸掉。
                    "WHERE session_id=? AND created_at >= ? AND length(created_at) >= 13 "
                    "GROUP BY h",
                    (session_id, cutoff),
                )
                .fetchall()
            )
        return {int(r[0]): r[1] for r in rows}


class ImageAssetStore:
    """生成过的图片登记在册：文件放 storage/images，索引放 index.json。"""

    def _index(self) -> Path:
        return get_settings().data_dir / "images" / "index.json"

    def save(self, asset: ImageAsset) -> None:
        assets = _read_json(self._index(), [])
        assets.append(
            {
                "image_path": asset.image_path,
                "caption": asset.caption,
                "created_at": asset.created_at,
            }
        )
        _write_json(self._index(), assets)

    def list_images(self, limit: int = 20) -> list:
        assets = _read_json(self._index(), [])
        return list(reversed(assets))[:limit]


class RouteConfigStore:
    """语音路由配置，按会话存。"""

    def _path(self, session_id: str) -> Path:
        return get_settings().data_dir / "route_config" / f"{session_id}.json"

    def load(self, session_id: str) -> dict:
        return _read_json(self._path(session_id), {"route": "cascade", "auto_degrade": True})

    def save(self, session_id: str, config: dict) -> None:
        _write_json(self._path(session_id), config)


class LogStore:
    """工具调用日志，一行一条 JSON，追加着写。"""

    def append(self, entry: dict) -> None:
        path = get_settings().data_dir / "tool_calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"time": datetime.now().isoformat(timespec="seconds"), **entry}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def new_id() -> str:
    """给记忆、图片这些生成个不重复的 ID。"""
    return uuid.uuid4().hex
