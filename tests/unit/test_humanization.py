"""新增拟人机制的纯逻辑单测：间隔感知（gap_perception）、反应前缀剥离
（_strip_reaction_tag）、关系氛围线（recent_arc 的聚合规则用假账本验证）。

不启应用、不打网络、不需要任何密钥。
"""

from datetime import datetime, timedelta

import pytest

from tools.misc import ClockTool
from capability.memory import _arc_from_entries  # noqa: E402


# ------------------------- gap_perception（N1） -------------------------
_NOW = datetime(2026, 9, 16, 20, 0, 0)


def _iso(minutes_ago: float) -> str:
    return (_NOW - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")


@pytest.mark.parametrize(
    "minutes_ago,phrase",
    [
        (1, "刚聊完没两句"),
        (4.9, "刚聊完没两句"),
        (30, "刚分开一会儿"),
        (100, "半天没见"),
        (11 * 60, "大半天没聊了"),
        (20 * 60, "昨天聊完到现在"),
        (2 * 24 * 60, "两三天没理我"),
        (10 * 24 * 60, "好一阵子没见了"),
    ],
)
def test_gap_perception_tiers(minutes_ago, phrase):
    out = ClockTool().gap_perception(_iso(minutes_ago), now=_NOW)
    assert phrase in out


def test_gap_perception_short_tier_has_no_reconnect_hint():
    # 刚聊完：不该出现"重新熟络"的分寸提示
    out = ClockTool().gap_perception(_iso(2), now=_NOW)
    assert "不用重新打招呼" in out
    assert "想念" not in out


def test_gap_perception_long_tier_has_reconnect_hint():
    # 2~3 天档：带"这才回来"的分寸提示
    out = ClockTool().gap_perception(_iso(2 * 24 * 60), now=_NOW)
    assert "这才回来" in out
    # 更久（10 天）：提示升级为"直说想念/不满"
    longer = ClockTool().gap_perception(_iso(10 * 24 * 60), now=_NOW)
    assert "想念" in longer


def test_gap_perception_bad_input_returns_empty():
    assert ClockTool().gap_perception("") == ""
    assert ClockTool().gap_perception(None) == ""
    assert ClockTool().gap_perception("不是时间") == ""


def test_gap_perception_future_timestamp_treated_as_just_now():
    # 时钟被拨回去之类的脏数据：当"刚聊完"处理最安全，不能炸
    future = (_NOW + timedelta(hours=3)).isoformat(timespec="seconds")
    out = ClockTool().gap_perception(future, now=_NOW)
    assert "刚聊完没两句" in out


# ---------------------- _strip_reaction_tag（F8） ----------------------
from orchestration.pipeline import _strip_reaction_tag  # noqa: E402


def test_strip_reaction_tag_leading():
    assert _strip_reaction_tag("@r 你说什么？") == "你说什么？"


def test_strip_reaction_tag_no_tag_untouched():
    assert _strip_reaction_tag("你好呀") == "你好呀"


def test_strip_reaction_tag_only_strips_leading():
    # 中间的 @r 是正文的一部分（比如邮箱），不动
    assert _strip_reaction_tag("发到 a@r.com") == "发到 a@r.com"


# ---------------- F：语音轮情绪标签 + 反应前缀的组合剥离 ----------------
from tools.speech import strip_emotion_marks  # noqa: E402


def test_reaction_tag_after_emotion_mark_stripped():
    """语音轮 "[开心]@r 你好"：先剥情绪标记再剥反应前缀，两个标记都要清掉。

    旧顺序（先剥 @r 再剥情绪）会因开头是 '[' 而放过 @r，落库残留 "@r 你好"。
    """
    cleaned = _strip_reaction_tag(strip_emotion_marks("[开心]@r 你好"))
    assert cleaned == "你好"


def test_plain_reaction_tag_still_stripped():
    # 纯文本轮 "@r 你好" 不能回归
    assert _strip_reaction_tag(strip_emotion_marks("@r 你好")) == "你好"
    assert _strip_reaction_tag("@r 你好") == "你好"


def test_email_reaction_tag_not_touched():
    # 中间/尾部的 @r 是正文，组合剥离也不能误伤
    assert _strip_reaction_tag(strip_emotion_marks("发到 a@r.com")) == "发到 a@r.com"


def test_emotion_prefix_survives_for_voice_synthesis():
    """语音轮必须在 final_reply 里给合成器留下情绪标签（从 clean 文本里拆不到了）。"""
    from tools.speech import split_emotion

    raw = "[开心]@r 你好"
    clean = _strip_reaction_tag(strip_emotion_marks(raw))
    _body, tags, descs = split_emotion(raw)
    prefix = f"[{tags[0]}]" if tags else ""
    prefix += f"（{descs[0]}）" if descs else ""
    rebuilt = prefix + clean
    # 合成路径 split_emotion 仍能拆出情绪词，且正文无 @r
    synth_body, synth_tags, _d = split_emotion(rebuilt)
    assert synth_tags == ["开心"]
    assert synth_body == "你好"
    assert "@r" not in synth_body


# ----------------------- recent_arc（S10）聚合规则 -----------------------
def test_recent_arc_rules():
    from capability.memory import _arc_from_entries

    # 空账本：不注入
    assert _arc_from_entries([]) == ""
    # 正常账本：拼叙事 + 趋势
    entries = [
        {"delta": 1.5, "reason": "聊得开心，关系热乎了一点", "time": _iso(60)},
        {"delta": -0.5, "reason": "闹了点不愉快，热度降了一点", "time": _iso(60 * 26)},
    ]
    arc = _arc_from_entries(entries, now=_NOW)
    assert "聊得开心" in arc
    assert "昨天" in arc
    # 正向趋势（+1.0）不够"明显升温"的阈值，不应出现该短语
    assert "明显在升温" not in arc
    # 趋势阈值：和 ≥ +2 出现"升温"
    hot = _arc_from_entries([
        {"delta": 1.5, "reason": "聊得开心", "time": _iso(10)},
        {"delta": 1.0, "reason": "又聊了一轮", "time": _iso(30)},
    ], now=_NOW)
    assert "明显在升温" in hot
    # 负向趋势 ≤ -2 出现"有点僵"
    cold = _arc_from_entries([
        {"delta": -1.5, "reason": "闹了点不愉快", "time": _iso(10)},
        {"delta": -1.0, "reason": "又闹了一场", "time": _iso(30)},
    ], now=_NOW)
    assert "有点僵" in cold


def test_recent_arc_same_day_dedupes_label_and_reason():
    """D：同一天连聊三轮、reason 相同 —— "今天"不出现、reason 只出现一次。"""
    same_reason = "聊得开心，关系热乎了一点"
    entries = [
        {"delta": 1, "reason": same_reason, "time": _iso(30)},
        {"delta": 1, "reason": same_reason, "time": _iso(20)},
        {"delta": 1, "reason": same_reason, "time": _iso(10)},
    ]
    arc = _arc_from_entries(entries, now=_NOW)
    assert arc.count("今天") == 0          # 单日不加日期标签
    assert arc.count(same_reason) == 1     # reason 去重后只出现一次
    assert "最近这几轮" in arc             # 单日措辞用"这轮"不是"这几天"
    assert "这几天" not in arc


def test_recent_arc_same_day_keeps_distinct_reasons():
    """D：同一天三条不同 reason —— 全保留，仍不加"今天"，趋势措辞为"最近这几轮"。"""
    entries = [
        {"delta": 1, "reason": "聊得开心，关系热乎了一点", "time": _iso(30)},
        {"delta": 1, "reason": "平平常常聊了一会儿", "time": _iso(20)},
        {"delta": 1, "reason": "他说了难过的事，信任多了一分", "time": _iso(10)},
    ]
    arc = _arc_from_entries(entries, now=_NOW)
    assert arc.count("今天") == 0
    assert "聊得开心，关系热乎了一点" in arc
    assert "平平常常聊了一会儿" in arc
    assert "他说了难过的事，信任多了一分" in arc
    assert "最近这几轮" in arc
    assert "这几天" not in arc


def test_recent_arc_cross_day_keeps_date_labels():
    """D：跨天三条 —— 每天标签各出现一次，趋势措辞为"这几天"。"""
    entries = [
        {"delta": 1, "reason": "聊得开心", "time": _iso(10)},
        {"delta": 1, "reason": "又聊了一轮", "time": _iso(60 * 26)},      # 昨天
        {"delta": 1, "reason": "平平常常聊了一会儿", "time": _iso(60 * 50)},  # 前天
    ]
    arc = _arc_from_entries(entries, now=_NOW)
    assert arc.count("今天") == 1
    assert arc.count("昨天") == 1
    assert arc.count("前天") == 1
    assert "这几天" in arc
    assert "最近这几轮" not in arc
