"""所有存储类都合并在这一个文件里：人设、档案、画像、关系、聊天记录、图片、
路由配置、日志，全是"读 JSON 文件 -> 变成对象 / 对象 -> 写回 JSON"这一套。

没必要为每个 Store 单开一个文件，合在一起反而好找。
"""

import json
import uuid
from datetime import datetime
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


class PersonaConfigStore:
    """读 data/persona_config.json，那是人设的正主文件。"""

    def load(self) -> PersonaConfig:
        path = PROJECT_ROOT / "data" / "persona_config.json"
        raw = _read_json(path, {})
        if not raw:
            raise FileNotFoundError("persona_config.json 读不到，人设没法用")
        return PersonaConfig(
            char_id=raw["char_id"],
            char_name=raw["char_name"],
            self_introduction=raw["self_introduction"],
            age=raw["age"],
            hobbies=raw["hobbies"],
            background_story=raw["background_story"],
            taboos=raw.get("taboos", ""),
            default_intimacy=raw["default_intimacy"],
            daily_reset=raw["daily_reset"],
            mode_tones=raw["mode_tones"],
            memory_config=raw["memory_config"],
            fallback_replies=raw.get("fallback_replies", {}),
            ask_templates=raw.get("ask_templates", {}),
            diary_notes=raw.get("diary_notes", ""),
        )


class ProfileStore:
    """用户硬事实档案，一个会话一行，进 SQLite。"""

    def load(self, session_id: str) -> dict:
        return get_db().load_kv("profile", session_id)

    def save(self, session_id: str, profile: dict) -> None:
        get_db().save_kv("profile", session_id, profile)


class PortraitStore:
    """软画像存储，和档案一样进 SQLite。"""

    def load(self, session_id: str) -> dict:
        return get_db().load_kv("portrait", session_id)

    def save(self, session_id: str, portrait: dict) -> None:
        get_db().save_kv("portrait", session_id, portrait)


class RelationshipStore:
    """关系数值存储，进 SQLite。"""

    def load(self, session_id: str) -> dict:
        return get_db().load_kv("relationship", session_id)

    def save(self, session_id: str, relationship: dict) -> None:
        get_db().save_kv("relationship", session_id, relationship)


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

    def get_recent(self, session_id: str, n: int = 10) -> list:
        """拿最近 n 条记录，进程重启后恢复短期记忆用。"""
        return get_db().get_recent_chat(session_id, n)

    def get_chats_between(self, session_id: str, start_iso: str, end_iso: str) -> list:
        """按时间段捞记录（含头不含尾），写日记就靠这个把某天的对话整段拎出来。"""
        return get_db().get_chats_between(session_id, start_iso, end_iso)

    def last_chat_per_session(self) -> list:
        """每个会话最后一次聊天的时间，闲置自动写日记的巡检全靠它点名。"""
        return get_db().last_chat_per_session()


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
