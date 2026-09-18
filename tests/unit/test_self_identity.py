"""批次0"种子+涌现" + 批次G"披露预算"的单测：身份冻结（self_identity）。

覆盖：只收亲口说的 / 披露预算（stage 闸门 + 一轮最多一槽 + 说过被拦的不回头补冻）/
改口一次即历史（第二次拒绝）/ 越界年龄落 age_rejected 不静默丢弃 /
自我认知慢冻结（≥20 句素材 + 两轮一致）/ compose 只递下一槽 / display_name 回落链。
"""

import pytest

from capability import self_identity
from tools.storage import KVStoreTool


@pytest.fixture()
def kv(register_services):
    kv = KVStoreTool()
    register_services(kv_store=kv)
    return kv


def _set_stage(kv, session_id, stage):
    """把关系阶段摆到指定档：披露预算按 stage 走（G2）。"""
    kv.write("relationship", session_id, {"stage": stage, "intimacy": 0})


def test_freeze_first_turn_only_name(kv, fake_llm, register_services):
    """G1-G4 核心场景：turn 1 她一口气交代名字+年龄+城市+职业，
    stage 初识、他只问过名字 → 最多冻结 1 个键，其余**丢弃**（不是延后入库）。"""
    register_services(llm=fake_llm)
    fake_llm.chat_responses = [
        '{"name": "小满", "age": "22", "city": "苏州", "occupation": "上班", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "叫我小满吧，我22，在苏州上班。",
                               user_text="你叫什么名字呀")
    rec = kv.read("self", "t1")
    filled = [k for k in ("name", "age", "city", "occupation", "home") if rec.get(k)]
    assert filled == ["name"]  # 旧版这里会冻出 4 个——"认识她"一轮花光
    assert rec["name"] == "小满"
    assert rec.get("frozen_at")  # 名字冻上了，冻结时刻已记
    # 预算前缀内说过但被拦下的键要记账（age）：以后不回头补冻，等阶段到了她亲口再说；
    # 前缀外的键（city/occupation）初识阶段根本不进考虑，不留账
    assert set(rec.get("disclosed") or []) >= {"name", "age"}


def test_freeze_name_when_asked(kv, fake_llm, register_services):
    """G1 反向：初识阶段名字槽的钥匙是"他上一句问过"。"""
    register_services(llm=fake_llm)
    fake_llm.chat_responses = [
        '{"name": "小满", "age": "", "city": "", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "叫我小满就好。", user_text="你叫什么名字呀")
    assert kv.read("self", "t1")["name"] == "小满"


def test_freeze_not_even_name_if_not_asked(kv, fake_llm, register_services):
    """初识 + 他没问名字 → 一个键都不冻，哪怕她主动自报家门（档案倾泻就是这么做成的）。"""
    register_services(llm=fake_llm)
    fake_llm.chat_responses = [
        '{"name": "小满", "age": "", "city": "", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "对了，我叫小满。", user_text="你好呀，第一次跟你聊天")
    rec = kv.read("self", "t1") or {}
    assert not rec.get("name")
    assert "name" in (rec.get("disclosed") or [])  # 说过的账照记


def test_stage_unlocks_next_slot_and_dropped_keys_not_backfilled(kv, fake_llm, register_services):
    """G2：熟悉阶段预算开到年龄；初识说过但被丢弃的城市**不回头补冻**——
    等亲近阶段她亲口再说，那才是"挣来的"披露。"""
    register_services(llm=fake_llm)
    _set_stage(kv, "t1", "初识")
    fake_llm.chat_responses = [
        '{"name": "小满", "age": "", "city": "苏州", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "叫我小满，我在苏州这边。", user_text="你叫什么")
    rec = kv.read("self", "t1")
    assert rec["name"] == "小满"
    assert not rec.get("city") and "city" in rec["disclosed"]

    _set_stage(kv, "t1", "熟悉")
    fake_llm.chat_responses = [
        '{"name": "", "age": "22", "city": "苏州", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "我22，苏州最近好热。", user_text="")
    rec = kv.read("self", "t1")
    assert rec["age"] == "22"   # 熟悉阶段的槽给了年龄
    assert not rec.get("city")  # 城市虽在预算前缀内，但没人问起——说过被拦的不主动回头补

    # 他直接问起 = "自然需要的时候"：被丢弃的槽重开，她才答得上来
    _set_stage(kv, "t1", "亲近")
    fake_llm.chat_responses = [
        '{"name": "", "age": "", "city": "苏州", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "我在苏州呀。", user_text="你在哪个城市来着")
    assert kv.read("self", "t1")["city"] == "苏州"


def test_revision_once_then_refused(kv, fake_llm, register_services):
    """G5：改口一次入账（revisions），第二次拒绝、冻结值原样保留。"""
    register_services(llm=fake_llm)
    kv.write("self", "t1", {"name": "小满"})
    fake_llm.chat_responses = [
        '{"name": "满满", "age": "", "city": "", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "还是叫我满满吧，小满是口误。", user_text="")
    rec = kv.read("self", "t1")
    assert rec["name"] == "满满"
    assert rec["revisions"][0]["key"] == "name"
    assert rec["revisions"][0]["old"] == "小满"
    assert rec["revisions"][0]["new"] == "满满"
    # 第二次改口：拒绝
    fake_llm.chat_responses = [
        '{"name": "第三个名字", "age": "", "city": "", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "算了还是叫第三个名字吧。", user_text="")
    rec = kv.read("self", "t1")
    assert rec["name"] == "满满"
    assert len(rec["revisions"]) == 1


def test_revision_lands_in_long_term_memory(kv, fake_llm, register_services, monkeypatch):
    """G5：改口不只覆盖字段，还要落一条长期记忆——被记录的改口比不可变的错值更像人。

    只覆盖字段的话，"她曾经说过另一个"在数据里毫无痕迹，三天后她自己无从提起。
    """
    register_services(llm=fake_llm)
    noted = []
    import capability.memory as memory_mod
    monkeypatch.setattr(
        memory_mod, "remember_note",
        lambda sid, content, kind="event", importance=3, **kw:
            noted.append((sid, content, kind, importance)),
    )
    kv.write("self", "t1", {"name": "小满"})
    fake_llm.chat_responses = [
        '{"name": "满满", "age": "", "city": "", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "还是叫我满满吧，小满是口误。", user_text="")
    assert len(noted) == 1
    sid, content, kind, importance = noted[0]
    assert sid == "t1"
    assert kind == "event" and importance == 4
    # 逐键的人话句式，不是把 old/new 原样倒出来
    assert content == "她原来说自己叫「小满」，后来改口叫「满满」。"

    # 第二次改口被拒（一生一次）：什么都没变，就不该再多一条记忆
    fake_llm.chat_responses = [
        '{"name": "第三个名字", "age": "", "city": "", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "算了还是叫第三个名字吧。", user_text="")
    assert len(noted) == 1


def test_freeze_only_fills_empty_same_value_not_revision(kv, fake_llm, register_services):
    """只填空：同值重述不算改口（模型把同一事实换个措辞再抽一遍是常态，
    不能让它白烧唯一一次额度）。"""
    register_services(llm=fake_llm)
    kv.write("self", "t1", {"name": "小满"})
    fake_llm.chat_responses = [
        '{"name": "小满", "age": "", "city": "", "occupation": "", "home": ""}'
    ]
    self_identity.maybe_freeze("t1", "我是小满呀。", user_text="")
    rec = kv.read("self", "t1")
    assert rec["name"] == "小满"
    assert not rec.get("revisions")


def test_age_out_of_range_marked_then_retracted(kv, fake_llm, register_services):
    """G6：越界年龄不再静默丢弃——落 age_rejected 标记，compose 引导改口一次，
    改口落地后标记清除并记入 revisions。"""
    register_services(llm=fake_llm)
    fake_llm.chat_responses = [
        '{"name": "", "age": "35", "city": "", "occupation": "", "home": ""}',
        '{"name": "", "age": "23", "city": "", "occupation": "", "home": ""}',
    ]
    self_identity.maybe_freeze("t1", "我都35了。", user_text="")
    rec = kv.read("self", "t1")
    assert not rec.get("age")
    assert rec.get("age_rejected") == "35"
    # compose 注入改口引导（而不是让短期上下文和档案各说各话）
    sections = self_identity.compose_sections("t1")
    assert any("口误要圆" in s for s in sections)
    self_identity.maybe_freeze("t1", "开玩笑的，其实23。", user_text="")
    rec = kv.read("self", "t1")
    assert rec["age"] == "23"
    assert not rec.get("age_rejected")
    assert any(r["key"] == "age" and r["old"] == "35" and r["new"] == "23"
               for r in rec["revisions"])


def test_narrative_needs_material_and_two_consistent_drafts(kv, fake_llm, register_services):
    """G7：素材不足 20 句不冻结；第一轮草稿只留档；两轮一致才冻结；
    冻结后不再改写（旧版第 3 轮 5 句话就安装人格，那是把社交面焊死成底色）。"""
    register_services(llm=fake_llm)
    lines = [f"第{i}句话。" for i in range(25)]
    empty_extract = '{"name":"","age":"","city":"","occupation":"","home":""}'
    # 素材不足 20 句：不开工
    fake_llm.chat_responses = [empty_extract, "说话不绕弯，想到什么说什么。"]
    self_identity.maybe_freeze("t1", "随便聊聊", recent_her_lines=lines[:10],
                               interaction_count=5)
    rec = kv.read("self", "t1") or {}
    assert not rec.get("self_narrative") and not rec.get("narrative_candidate")
    # 第一轮草稿：留档不冻结
    fake_llm.chat_responses = [empty_extract, "说话不绕弯，想到什么说什么。"]
    self_identity.maybe_freeze("t1", "随便聊聊", recent_her_lines=lines,
                               interaction_count=5)
    rec = kv.read("self", "t1")
    assert rec.get("narrative_candidate") and not rec.get("self_narrative")
    # 第二轮独立生成高度一致：冻结
    fake_llm.chat_responses = [empty_extract, "说话不绕弯，想到什么就说什么。"]
    self_identity.maybe_freeze("t1", "又聊一轮", recent_her_lines=lines,
                               interaction_count=6)
    first = kv.read("self", "t1")["self_narrative"]
    assert first
    # 冻结后哪怕轮数更高也不再改写。预置一条**全冻结**记录：五个身份键都非空 →
    # 身份抽取整段跳过，唯一一发响应只可能落在底色重议路径上——而重议要求
    # "明显矛盾"证据，不会误触。
    # 走 update() 不走 write()：E7 之后 write("self", …) 是"只填空键、冲突就整包
    # 拒绝"的冻结写入，拿它预置一条与已冻结值**不同**的记录会被整包拒掉（那道闸
    # 正是为此而设，见 tools/storage.py::_write_self_frozen）。改 self 表的正当
    # 路径是 update() 的原子闭包。
    frozen = "说话不绕弯，想到什么说什么。"
    kv.update("self", "t1", lambda rec: {
        **(rec or {}), "name": "小满", "age": "22", "city": "苏州",
        "occupation": "上班", "home": "老城区", "self_narrative": frozen})
    fake_llm.chat_responses = ["完全不同的另一种描述。"]
    self_identity.maybe_freeze("t1", "再聊一轮", recent_her_lines=lines,
                               interaction_count=9)
    assert kv.read("self", "t1")["self_narrative"] == frozen


def test_narrative_inconsistent_drafts_not_frozen(kv, fake_llm, register_services):
    """G7：两轮草稿对不上 → 都不是底色，弃旧留新再等，绝不冻结。"""
    register_services(llm=fake_llm)
    lines = [f"第{i}句话。" for i in range(25)]
    # 每次 maybe_freeze 先烧一发身份抽取（rec 空 + user_text 未接线 → 名字槽开），
    # 草稿响应要排在抽取响应之后
    fake_llm.chat_responses = ['{"name":"","age":"","city":"","occupation":"","home":""}',
                               "我说话直，想到什么说什么。"]
    self_identity.maybe_freeze("t1", "聊聊", recent_her_lines=lines, interaction_count=5)
    fake_llm.chat_responses = ['{"name":"","age":"","city":"","occupation":"","home":""}',
                               "我其实很内向，喜欢一个人待着。"]
    self_identity.maybe_freeze("t1", "再聊", recent_her_lines=lines, interaction_count=6)
    rec = kv.read("self", "t1")
    assert not rec.get("self_narrative")
    assert rec.get("narrative_candidate")  # 新草稿留档，等下一轮印证


def test_freeze_llm_failure_silent(kv, fake_llm, register_services):
    """LLM 挂了静默跳过：身份继续空着，聊天照常，下轮再试。"""
    register_services(llm=fake_llm)
    fake_llm.chat_responses = [RuntimeError("llm down")]
    self_identity.maybe_freeze("t1", "我叫小满。")  # 不应抛
    assert not kv.read("self", "t1")


def test_compose_sections_three_shapes(kv):
    # 全空：引导自命名（G3：只提名字这一个槽，不列五键清单）
    sections = self_identity.compose_sections("t1")
    assert any("你的第一次" in s for s in sections)
    # 部分冻结（无名字）：名字是下一个槽，催一次取名，不代取
    kv.write("self", "t1", {"city": "杭州"})
    sections = self_identity.compose_sections("t1")
    assert any("名字还没定" in s for s in sections)
    assert any("杭州" in s for s in sections)
    # 全冻结：你是谁 + 自我认知；没有新槽时明确"不用再说新的"
    kv.write("self", "t1", {"name": "小满", "city": "杭州",
                            "self_narrative": "说话不绕弯。"})
    sections = self_identity.compose_sections("t1")
    assert any("小满" in s and "你的名字" in s for s in sections)
    assert any("不随他的评价改变" in s for s in sections)


def test_compose_sections_only_next_slot(kv):
    """G3：初识预算只有名字——名字已定就没有新槽（明说"不用再说新的"）；
    到熟悉阶段下一槽是年龄，且只递这一个槽。"""
    kv.write("self", "t1", {"name": "小满"})
    sections = self_identity.compose_sections("t1", user_text="在干嘛呢")
    assert any("不用再说新的" in s for s in sections)
    _set_stage(kv, "t1", "熟悉")
    sections = self_identity.compose_sections("t1", user_text="在干嘛呢")
    assert any("只说【年龄】" in s for s in sections)
    # 别的槽不会以"只说【X】"的形式被递出去
    assert not any("只说【城市】" in s or "只说【职业】" in s or "只说【住处】" in s
                   for s in sections)


def test_display_name_fallback_chain(kv):
    # 没取名 → 回落人设占位名
    assert self_identity.display_name("t1") == "Aria"
    kv.write("self", "t1", {"name": "小满"})
    assert self_identity.display_name("t1") == "小满"
