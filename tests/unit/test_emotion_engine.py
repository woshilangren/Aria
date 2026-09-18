"""批次5 情绪引擎单测：心情惯性（N2）、心事（S2）、关系收敛层。

不联网、不启应用；tracker 走真 KVStoreTool（conftest 已隔离 DATA_DIR）。
"""

import pytest

from capability.memory import RelationshipTracker
from capability.quirks import MoodEngine
from shared.types import PerceptionExtras
from tools.storage import KVStoreTool


@pytest.fixture()
def kv(register_services):
    kv = KVStoreTool()
    register_services(kv_store=kv)
    return kv


# ----------------------- MoodEngine 惯性（N2） -----------------------
def _rel():
    return {"mood": "平常"}


def test_negative_event_sets_mood_with_inertia():
    rel = _rel()
    mood = MoodEngine().update(rel, "angry", intensity=1.0)
    assert mood == "别扭"
    assert rel["mood_left"] == 5          # 2 + 1.0*3
    assert rel["mood_from"] == "angry"


def test_inertia_decays_over_neutral_turns(monkeypatch):
    # 钉死骰子（random 永不命中漂移），让衰减路径完全确定
    monkeypatch.setattr("capability.quirks.random.random", lambda: 0.99)
    eng = MoodEngine()
    rel = _rel()
    eng.update(rel, "sad", intensity=0.3)   # left = 2+round(0.9)=3
    left0 = rel["mood_left"]
    m1 = eng.update(rel, "neutral")
    assert m1 == "低落" and rel["mood_left"] == left0 - 1
    m2 = eng.update(rel, "neutral")
    m3 = eng.update(rel, "neutral")
    m4 = eng.update(rel, "neutral")
    # 衰减到 0 回平常（且不被漂移干扰）
    assert m4 == "平常" and rel["mood"] == "平常"


def test_comfort_halves_inertia_but_keeps_mood():
    rel = _rel()
    eng = MoodEngine()
    eng.update(rel, "angry", intensity=1.0)     # left=5
    mood = eng.update(rel, "neutral", comfort_mode=True)
    assert mood == "心软"
    assert rel["mood_left"] == 2                 # 5//2，没有立刻翻篇


def test_positive_mood_is_light():
    import random

    random.seed(7)  # 钉住骰子，让"命中正面心情"的分支稳定走到
    rel = _rel()
    MoodEngine().update(rel, "happy", intensity=0.8)
    # 正面心情最多留 1 轮（来得快去得快）；也可能这一掷没中（保持平常）
    assert rel.get("mood_left", 0) <= 1
    assert rel["mood"] in ("得意", "雀跃", "来劲", "心软", "平常")


# --------------------- 心事（S2）与收敛层 ---------------------
def test_concern_set_decay_and_clear(kv):
    t = RelationshipTracker()
    ex = PerceptionExtras(concern_text="他是不是烦我了", concern_delta=0.3)
    rel = t.update("s1", "neutral", extras=ex)
    assert rel["concern"]["text"] == "他是不是烦我了"
    # C7：新念头起点是 0.5+delta 再衰减 0.9 → (0.5+0.3)*0.9 = 0.72。
    # 以前是 0+delta（0.3*0.9=0.27），那个起点在 delta<0.35 时永远压在
    # persona_engine 的表达门槛之下——文本入了库、强度在衰减，她却从不把它说出口。
    assert rel["concern"]["intensity"] == pytest.approx(0.72)
    assert rel["concern"]["id"]                       # 实体有 id，不是一团自由文本

    # 连续多轮无事件：自动衰减到归零翻篇。0.72 * 0.9^n <= 0.05 需 n>=26，给到 30
    for _ in range(30):
        rel = t.update("s1", "neutral")
    assert rel["concern"]["text"] == ""
    assert rel["concern"]["intensity"] == 0.0
    assert not rel["concern"].get("id")               # 归零即翻篇，实体 id 一起清


def test_concern_continuation_keeps_entity_and_uses_inertia(kv):
    """C7：同一个念头的延续——实体不变（id/created_at 保留），强度走惯性不重新起算。"""
    t = RelationshipTracker()
    first = t.update("s4", "neutral", extras=PerceptionExtras(
        concern_text="他是不是烦我了", concern_delta=0.3))
    rel = t.update("s4", "neutral", extras=PerceptionExtras(
        concern_text="他是不是嫌我烦", concern_delta=0.3, same_concern=True))
    assert rel["concern"]["id"] == first["concern"]["id"]
    assert rel["concern"]["created_at"] == first["concern"]["created_at"]
    assert rel["concern"]["text"] == "他是不是嫌我烦"     # 措辞可以换
    # 惯性：0.72 + 0.3 = 1.02 钳到 1.0，再衰减 0.9 → 0.9（不是从 0.5 重新起算）
    assert rel["concern"]["intensity"] == pytest.approx(0.9)


def test_concern_new_entity_replaces_old(kv):
    """C7：same_concern=False 是关旧开新——换实体、换 id、强度从 0.5+delta 起算。"""
    t = RelationshipTracker()
    first = t.update("s5", "neutral", extras=PerceptionExtras(
        concern_text="他是不是烦我了", concern_delta=0.3))
    rel = t.update("s5", "neutral", extras=PerceptionExtras(
        concern_text="实习转正会不会没戏", concern_delta=0.1, same_concern=False))
    assert rel["concern"]["id"] != first["concern"]["id"]
    assert rel["concern"]["text"] == "实习转正会不会没戏"   # 文本不再漂到旧念头上
    assert rel["concern"]["intensity"] == pytest.approx(0.54)   # (0.5+0.1)*0.9


def test_confort_halves_concern(kv):
    t = RelationshipTracker()
    t.update("s2", "angry", extras=PerceptionExtras(
        concern_text="他真的会走吗", concern_delta=0.8))
    rel = t.update("s2", "neutral", comfort_mode=True)
    assert rel["concern"]["intensity"] < 0.5   # 被哄后显著回落


def test_calm_goes_up_on_tone_down_and_never_touches_core(kv):
    t = RelationshipTracker()
    rel = t.update("s3", "neutral", extras=PerceptionExtras(feedback="tone_down"))
    assert rel["calm"] == pytest.approx(0.15)
    rel = t.update("s3", "neutral", extras=PerceptionExtras(feedback="tone_down"))
    assert rel["calm"] == pytest.approx(0.3)
    # 哄让收敛回落（温度回升）
    rel = t.update("s3", "neutral", comfort_mode=True)
    assert rel["calm"] < 0.3
    # 无反馈轮收敛不变
    rel2 = t.update("s3", "neutral")
    assert rel2["calm"] == rel["calm"]


def test_extras_missing_is_safe(kv):
    """旧调用方（语音写回）不传 extras：不炸，行为退化。"""
    rel = RelationshipTracker().update("s4", "happy")
    assert rel["concern"]["intensity"] == 0.0
    assert rel["calm"] == 0.0
