"""管道纯逻辑单测：`_subst_count` / `_is_onechar_reply` / `_needs_thinking`
/ `_too_short_input` / `_clean_review_input` / `_has_city`。

不启应用、不打网络、不需要任何密钥——只测调度层里那些纯函数。
"""

import pytest

from orchestration.pipeline import (
    TurnState,
    _clean_review_input,
    _has_city,
    _is_onechar_reply,
    _needs_thinking,
    _subst_count,
    _too_short_input,
)
from shared.types import EmotionResult

# 「想」的判定用到的配置：固定成可知值，避免被仓库 config.json 改动带跑。
_THINKING_CFG = {
    "thinking": {
        "enabled": True,
        "char_threshold": 60,
        "never_on_comfort": True,
        "never_on_emotions": ["sad", "angry", "委屈"],
        "force_if_last_poor": True,
    }
}


@pytest.fixture
def thinking_config(monkeypatch):
    """把 pipeline 里 `_needs_thinking` 读的配置钉死，返回一个可改 enabled 的开关。"""
    state = {"cfg": _THINKING_CFG}
    monkeypatch.setattr(
        "orchestration.pipeline.load_app_config", lambda: state["cfg"]
    )
    return state


def _state(text, emotion=None, comfort=False):
    st = TurnState(user_text=text, session_id="t")
    st.comfort_mode = comfort
    st.emotion = emotion
    return st


# --------------------------- _subst_count ---------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("你好。", 2),          # 2 汉字，句号不算
        ("Hello!", 5),          # 5 字母
        ("a1,。 ！", 2),        # 1 字母 + 1 数字
        ("    ", 0),            # 全空白
        ("", 0),
        (None, 0),
        ("2026 年", 5),         # 4 数字 + 1 汉字
    ],
)
def test_subst_count(text, expected):
    assert _subst_count(text) == expected


# ------------------------- _is_onechar_reply ------------------------
@pytest.mark.parametrize(
    "reply,expected",
    [
        ("哼", True),           # 单字崩
        ("。！？", True),        # 0 实质字符 <= 1
        ("", True),
        ("   ", True),
        ("好的", False),         # 2 汉字
        ("哈！", True),          # 1 汉字 + 标点 -> 1 个实质字符，仍判敷衍
    ],
)
def test_is_onechar_reply(reply, expected):
    assert _is_onechar_reply(reply) is expected


def test_is_onechar_reply_boundary():
    # "哈！" 去掉标点只剩 1 个实质字符 -> 判为敷衍
    assert _is_onechar_reply("哈！") is True
    # 纯标点也是 <=1
    assert _is_onechar_reply("！！！") is True


# --------------------------- _needs_thinking ------------------------
def test_needs_thinking_disabled_globally(thinking_config):
    thinking_config["cfg"] = {"thinking": {**_THINKING_CFG["thinking"], "enabled": False}}
    long_text = "这是一段很长的输入内容用来触发阈值" * 4
    assert _needs_thinking(_state(long_text)) is False


def test_needs_thinking_comfort_never(thinking_config):
    long_text = "我好累啊今天什么都不想干" * 5
    assert _needs_thinking(_state(long_text, comfort=True)) is False


def test_needs_thinking_never_on_emotions(thinking_config):
    long_text = "我真的非常生气现在特别想找人说说话" * 3
    emo = EmotionResult(emotion="angry", intensity=0.8)
    assert _needs_thinking(_state(long_text, emotion=emo)) is False


def test_needs_thinking_short_sad_never(thinking_config):
    emo = EmotionResult(emotion="sad", intensity=0.6)
    assert _needs_thinking(_state("我好难过", emotion=emo)) is False


def test_needs_thinking_long_text_triggers(thinking_config):
    long_text = "这是一段很长的输入内容用来触发阈值" * 4  # 长度远超 60
    assert _needs_thinking(_state(long_text)) is True


def test_needs_thinking_regex_triggers(thinking_config):
    # 短输入但命中"追问原因"模式 -> 要想
    assert _needs_thinking(_state("这件事为什么会这样")) is True


def test_needs_thinking_noneed_thinking(thinking_config):
    assert _needs_thinking(_state("你好呀")) is False


# --------------------------- _too_short_input -----------------------
def test_too_short_input_mentions_original_and_rewrite():
    out = _too_short_input("哼")
    assert "哼" in out
    assert "只回了一个字" in out


# -------------------------- _clean_review_input ---------------------
def test_clean_review_input_mentions_reanswer():
    out = _clean_review_input("原话")
    assert "原话" in out
    assert "重新回答" in out


# ------------------------------ _has_city ---------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("上海天气", True),
        ("北京天气怎么样", True),
        ("今天天气不错", True),   # 正则只粗查"X天气"结构
        ("你好呀", False),
        ("", False),
        (None, False),
    ],
)
def test_has_city(text, expected):
    assert _has_city(text) is expected
