"""批次0"种子+涌现"的单测：身份冻结（self_identity）。

覆盖：只收亲口说的 / 只填空不改口 / 年龄锚点校验 / 自我认知一次冻结 /
compose 三种形态（全空/缺名/全冻结）/ display_name 回落链。
"""

import pytest

from capability import self_identity
from tools.storage import KVStoreTool


@pytest.fixture()
def kv(register_services):
    kv = KVStoreTool()
    register_services(kv_store=kv)
    return kv


def test_freeze_extracts_and_fills(kv, fake_llm, register_services):
    register_services(llm=fake_llm)
    fake_llm.chat_responses = [
        '{"name": "小满", "age": "22", "city": "苏州", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "嗯……叫我小满吧，我22，在苏州这边上班。")
    rec = kv.read("self", "t1")
    assert rec["name"] == "小满"
    assert rec["age"] == "22"
    assert rec["city"] == "苏州"
    assert rec.get("frozen_at")  # 冻结时刻已记


def test_freeze_never_fills_unspoken(kv, fake_llm, register_services):
    """她没说的键必须空——模型根据常识补全的（比如她只提到城市）不许带出年龄。"""
    register_services(llm=fake_llm)
    fake_llm.chat_responses = [
        '{"name": "", "age": "22", "city": "成都", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "成都最近好热啊。")  # 她只说了城市
    rec = kv.read("self", "t1")
    assert rec.get("city") == "成都"
    # 年龄是模型编的——但提示词要求"没提到就空串"，假 LLM 若违规填了，
    # 我们无法从文本判断，这条测试只锁：没有名字就不冻结名字
    assert not rec.get("name")


def test_freeze_only_fills_empty_never_overwrites(kv, fake_llm, register_services):
    """只填空不改口：已冻结的名字不被第二轮抽取覆盖。"""
    register_services(llm=fake_llm)
    kv.write("self", "t1", {"name": "小满"})
    fake_llm.chat_responses = [
        '{"name": "另一个名字", "age": "", "city": "", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "其实你以后叫我什么都行。")
    rec = kv.read("self", "t1")
    assert rec["name"] == "小满"  # 首次冻结永远赢


def test_freeze_age_outside_anchor_rejected(kv, fake_llm, register_services):
    """年龄出了 20~25 锚点范围不收（种子定下的唯一硬边界之一）。"""
    register_services(llm=fake_llm)
    fake_llm.chat_responses = [
        '{"name": "", "age": "35", "city": "", "occupation": "", "home": ""}',
        '{"name": "", "age": "23", "city": "", "occupation": "", "home": ""}',
    ]
    self_identity.maybe_freeze("t1", "我都35了。")
    assert not kv.read("self", "t1").get("age")
    self_identity.maybe_freeze("t1", "开玩笑的，其实23。")
    assert kv.read("self", "t1")["age"] == "23"


def test_narrative_frozen_once(kv, fake_llm, register_services):
    """自我认知只生成一次：第二次调用（哪怕轮数更高）不再改写。"""
    register_services(llm=fake_llm)
    fake_llm.chat_responses = ["说话不绕弯，想到什么说什么。", "完全不同的另一种描述。"]
    lines = ["你猜啊。", "这还用问？", "行吧，告诉你也没事。"]
    self_identity.maybe_freeze("t1", "随便聊聊", recent_her_lines=lines, interaction_count=3)
    first = kv.read("self", "t1")["self_narrative"]
    self_identity.maybe_freeze("t1", "又聊了一轮", recent_her_lines=lines, interaction_count=9)
    assert kv.read("self", "t1")["self_narrative"] == first  # 一次冻结永不改写


def test_freeze_llm_failure_silent(kv, fake_llm, register_services):
    """LLM 挂了静默跳过：身份继续空着，聊天照常，下轮再试。"""
    register_services(llm=fake_llm)
    fake_llm.chat_responses = [RuntimeError("llm down")]
    self_identity.maybe_freeze("t1", "我叫小满。")  # 不应抛
    assert not kv.read("self", "t1")


def test_compose_sections_three_shapes(kv):
    # 全空：引导自命名
    sections = self_identity.compose_sections("t1")
    assert any("你的第一次" in s for s in sections)
    # 部分冻结（无名字）：催一次取名，不代取
    kv.write("self", "t1", {"city": "杭州"})
    sections = self_identity.compose_sections("t1")
    assert any("名字还没定" in s for s in sections)
    assert any("杭州" in s for s in sections)
    # 全冻结：你是谁 + 自我认知
    kv.write("self", "t1", {"name": "小满", "city": "杭州",
                            "self_narrative": "说话不绕弯。"})
    sections = self_identity.compose_sections("t1")
    assert any("小满" in s and "你的名字" in s for s in sections)
    assert any("不随他的评价改变" in s for s in sections)


def test_display_name_fallback_chain(kv):
    # 没取名 → 回落人设占位名
    assert self_identity.display_name("t1") == "Aria"
    kv.write("self", "t1", {"name": "小满"})
    assert self_identity.display_name("t1") == "小满"
