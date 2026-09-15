"""工具调用管线纯逻辑单测：`_strip_commands` / `_extract_city`
/ `_build_rule_args` / `ToolGuard` / `RULE_TOOL_MAP`。

ToolGuard 需要一个 tool_registry：这里用最小替身（只实现 get_meta），
避免把真实注册表背后的外部 SDK 拉进来。
"""

import pytest

from capability.toolcall import (
    RULE_TOOL_MAP,
    ToolGuard,
    _build_rule_args,
    _extract_city,
    _strip_commands,
)
from shared.types import ToolCallSpec


class _FakeRegistry:
    """最小注册表替身：只提供 ToolGuard 需要的 get_meta。"""

    _METAS = {
        "weather_query": {"parameters": {"required": ["city"]}},
        "web_search": {"parameters": {"required": ["query"]}},
        "image_gen": {"parameters": {"required": ["prompt"]}},
        "clock_now": {"parameters": {"required": []}},
        "diary_write": {"parameters": {"required": []}},
    }

    def get_meta(self, name):
        return self._METAS.get(name)


@pytest.fixture
def guard(register_services):
    register_services(tool_registry=_FakeRegistry())
    return ToolGuard()


# --------------------------- _strip_commands ------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("帮我查一下北京天气", "北京天气"),
        ("上海天气", "上海天气"),
        ("现在几点了", "几点了"),
        ("画一张猫", "猫"),
        ("", ""),
        (None, ""),
    ],
)
def test_strip_commands(raw, expected):
    assert _strip_commands(raw) == expected


# ---------------------------- _extract_city -------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("上海天气", "上海"),
        ("北京天气怎么样", "北京"),
        ("在东京玩", "东京"),
        ("随便聊聊", ""),
        ("", ""),
        (None, ""),
    ],
)
def test_extract_city(raw, expected):
    assert _extract_city(raw) == expected


# --------------------------- _build_rule_args -----------------------
def test_build_rule_args_weather_extract():
    assert _build_rule_args("weather", "北京天气", "广州") == {"city": "北京"}


def test_build_rule_args_weather_fallback_city():
    assert _build_rule_args("weather", "天气怎么样", "广州") == {"city": "广州"}


def test_build_rule_args_weather_default_city():
    # 抠不到又没 fallback -> 兜底"上海"
    assert _build_rule_args("weather", "今天天气", "") == {"city": "上海"}


def test_build_rule_args_search():
    assert _build_rule_args("search", "帮我查一下新闻", "") == {
        "query": "新闻",
        "top_k": 3,
    }


def test_build_rule_args_image():
    assert _build_rule_args("image", "画一张猫", "") == {"prompt": "猫"}


def test_build_rule_args_diary_needs_no_args():
    assert _build_rule_args("diary", "写日记", "") == {}


def test_build_rule_args_other_intent_empty():
    assert _build_rule_args("chat", "随便聊聊", "") == {}


# ------------------------------ ToolGuard ---------------------------
def test_guard_ok(guard):
    ok, reason = guard.check(
        ToolCallSpec(tool_name="weather_query", arguments={"city": "上海"})
    )
    assert ok is True
    assert reason == ""


def test_guard_unknown_tool(guard):
    ok, reason = guard.check(ToolCallSpec(tool_name="nope", arguments={}))
    assert ok is False
    assert "没有这个工具" in reason


def test_guard_missing_required_param(guard):
    ok, reason = guard.check(ToolCallSpec(tool_name="weather_query", arguments={}))
    assert ok is False
    assert "city" in reason


def test_guard_no_required_params_ok(guard):
    ok, reason = guard.check(ToolCallSpec(tool_name="diary_write", arguments={}))
    assert ok is True
    assert reason == ""


# ---------------------------- RULE_TOOL_MAP -------------------------
def test_rule_tool_map_covers_tool_intents():
    assert RULE_TOOL_MAP["weather"] == "weather_query"
    assert RULE_TOOL_MAP["search"] == "web_search"
    assert RULE_TOOL_MAP["image"] == "image_gen"
    assert RULE_TOOL_MAP["diary"] == "diary_write"
