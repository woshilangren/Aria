"""R07：坏配置的结构与值兜底（app config + 薄种子）。

- R07a：config.json 根必须是对象；已知字段按类型校验，坏字段**单独**回默认
  并告警（同 section 的好字段照常生效）；超时值有限且 >0，显式拒绝
  bool / NaN / Infinity / 字符串；
- R07b：persona_config.json 缺文件 / 坏 JSON / 数组根 / 缺必填键 → 内置薄种子
  安全默认；默认来源唯一（data/stores.py::_SEED_DEFAULTS），不含性格词；
  不覆写损坏原件。

全部走 tmp_path 的假 PROJECT_ROOT，不碰仓库真实 config.json / 种子。
"""

from __future__ import annotations

import json

import pytest

import config.settings as settings_mod
import data.stores as stores_mod
from config.settings import load_app_config, timeout_seconds


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    """把 PROJECT_ROOT 指到 tmp_path：config.json 与种子都落隔离目录。"""
    monkeypatch.setattr(settings_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(stores_mod, "PROJECT_ROOT", tmp_path)
    load_app_config.cache_clear()
    yield tmp_path
    load_app_config.cache_clear()


def _write_cfg(root, obj):
    (root / "config.json").write_text(json.dumps(obj, ensure_ascii=False),
                                      encoding="utf-8")


# ---------------- R07a：app config ----------------

def test_root_must_be_object(config_dir):
    (config_dir / "config.json").write_text("[]", encoding="utf-8")
    cfg = load_app_config()
    assert cfg["memory"]["max_turns_short_term"] == 12, "数组根整包回默认"
    assert cfg["llm"]["max_tokens"] == 1024


def test_bad_field_falls_back_alone(config_dir):
    """坏字段单独回默认，同 section 好字段与其他 section 全保留。"""
    _write_cfg(config_dir, {
        "memory": {"max_turns_short_term": "十二", "recall_top_k": 7},
        "proactive": {"enabled": True},
    })
    cfg = load_app_config()
    assert cfg["memory"]["max_turns_short_term"] == 12
    assert cfg["memory"]["recall_top_k"] == 7
    assert cfg["proactive"]["enabled"] is True


def test_bool_and_nonfinite_rejected(config_dir):
    """bool 是 int 的子类、Infinity 能溜过 `> 0`——都必须显式挡住。"""
    _write_cfg(config_dir, {"personality": {"quirk_rate": True},
                            "llm": {"temperature": float("nan")}})
    cfg = load_app_config()
    assert cfg["personality"]["quirk_rate"] == 0.12
    assert cfg["llm"]["temperature"] == 0.7


def test_timeout_bool_nan_inf_rejected(config_dir):
    _write_cfg(config_dir, {"timeouts": {"asr_seconds": True,
                                         "tts_seconds": float("inf"),
                                         "image_seconds": float("nan")}})
    assert timeout_seconds("asr_seconds", 60.0) == 60.0
    assert timeout_seconds("tts_seconds", 60.0) == 60.0
    assert timeout_seconds("image_seconds", 300.0) == 300.0
    assert timeout_seconds("not_there", 42.0) == 42.0


def test_timeout_zero_and_string_rejected(config_dir):
    _write_cfg(config_dir, {"timeouts": {"asr_seconds": 0}})
    assert timeout_seconds("asr_seconds", 60.0) == 60.0
    _write_cfg(config_dir, {"timeouts": {"asr_seconds": "60s"}})
    assert timeout_seconds("asr_seconds", 60.0) == 60.0


def test_valid_config_still_applies(config_dir):
    _write_cfg(config_dir, {"memory": {"max_turns_short_term": 20}})
    assert load_app_config()["memory"]["max_turns_short_term"] == 20


def test_missing_file_returns_defaults(config_dir):
    cfg = load_app_config()
    assert cfg["proactive"]["enabled"] is False
    assert cfg["personality"]["quirk_rate"] == 0.12


# ---------------- R07b：薄种子 ----------------

def test_seed_missing_file_returns_defaults(config_dir):
    pc = stores_mod.PersonaConfigStore().load()
    assert pc.char_id == "aria"
    assert pc.char_name == ""                      # 名字留白（8.1）
    assert pc.age == ""
    assert pc.hobbies == []
    assert "绝不承认" in pc.taboos                  # 铁律随默认走
    assert pc.default_intimacy == 20


def test_seed_corrupt_json_returns_defaults_and_keeps_original(config_dir):
    seed = config_dir / "data" / "persona_config.json"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text("{坏掉的JSON", encoding="utf-8")
    pc = stores_mod.PersonaConfigStore().load()
    assert pc.default_intimacy == 20
    assert "绝不承认" in pc.taboos
    assert seed.read_text(encoding="utf-8") == "{坏掉的JSON", "不得覆写损坏原件"


def test_seed_array_root_returns_defaults(config_dir):
    seed = config_dir / "data" / "persona_config.json"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text("[1,2,3]", encoding="utf-8")
    pc = stores_mod.PersonaConfigStore().load()
    assert pc.char_id == "aria"


def test_seed_missing_keys_filled_from_defaults(config_dir):
    seed = config_dir / "data" / "persona_config.json"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(json.dumps({"char_id": "aria", "default_intimacy": 30},
                               ensure_ascii=False), encoding="utf-8")
    pc = stores_mod.PersonaConfigStore().load()
    assert pc.default_intimacy == 30               # 文件里的键生效
    assert pc.char_name == ""                      # 缺的键用默认补齐


def test_seed_defaults_carry_no_personality_words():
    """默认种子不得携带性格词定义（8.1：加一个词就是"预制菜"）——
    身份锚点/铁律之外必须为空。"""
    d = stores_mod._SEED_DEFAULTS
    assert d["char_name"] == ""
    assert d["age"] == ""
    assert d["self_introduction"] == ""
    assert d["hobbies"] == []
    assert d["mode_tones"] == {}                   # 口吻不预置
