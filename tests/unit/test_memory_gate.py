"""批次7 单测：S4 记忆闸门（候选晋升+仲裁）、C2 第一刀/N4 生活面（char_life）。"""

import pytest

from capability import char_life, self_identity
from capability.memory import MemoryGatekeeper
from tools.storage import KVStoreTool


@pytest.fixture()
def kv(register_services):
    kv = KVStoreTool()
    llm = _ScriptedLLM()
    register_services(kv_store=kv, llm=llm)
    kv.scripted_llm = llm  # 仲裁动作在 LLM 替身上，测试从这里拨开关
    return kv


class _ScriptedLLM:
    """最小 LLM 替身：按提示词里出现的关键字决定回应。"""

    def __init__(self):
        self.arbitrate_action = "update"

    def chat(self, messages, temperature=None, max_tokens=None):
        system = messages[0]["content"]
        if "档案仲裁员" in system:
            return f'{{"action":"{self.arbitrate_action}","reason":"测试"}}'
        if "提取其中她说出口的" in system or "关于她自己的事实" in system:
            return '{"name":"","age":"","city":"","occupation":"","home":""}'
        if "对自己的认知" in system:
            return "测试认知。"
        if "总基调" in system:
            return "懒散周末"
        if "在琢磨的小事" in system:
            return "在读一本讲记忆的书。"
        raise RuntimeError("unexpected prompt")


# --------------------------- S4 候选晋升与仲裁 ---------------------------
def test_candidate_needs_two_hits(kv):
    g = MemoryGatekeeper()
    # 第一次：入池，不入档（证据不足）
    g.process("t1", "city", "杭州", confidence=0.9)
    assert not (kv.read("profile", "t1") or {}).get("city")
    # 第二次：达到门槛 → 仲裁 update → 入档
    g.process("t1", "city", "杭州", confidence=0.9)
    assert (kv.read("profile", "t1") or {}).get("city") == "杭州"


def test_confidence_floor_rejects(kv):
    MemoryGatekeeper().process("t1", "city", "杭州", confidence=0.3)
    kv_arb = MemoryGatekeeper()
    kv_arb.process("t1", "city", "杭州", confidence=0.5)  # 两次都低置信
    assert not (kv.read("profile", "t1") or {}).get("city")


def test_arbitration_keep_preserves_old_value(kv):
    kv.write("profile", "t1", {"city": "上海"})
    g = MemoryGatekeeper()
    kv.scripted_llm.arbitrate_action = "keep"
    # 第一次入池 + 第二次达到门槛 → 仲裁 keep → 旧值保留
    g.process("t1", "city", "杭州", confidence=0.9)
    g.process("t1", "city", "杭州", confidence=0.9)
    assert (kv.read("profile", "t1") or {}).get("city") == "上海"


def test_arbitration_update_replaces_old(kv):
    kv.write("profile", "t1", {"city": "上海"})
    g = MemoryGatekeeper()
    kv.scripted_llm.arbitrate_action = "update"
    g.process("t1", "city", "杭州", confidence=0.9)
    g.process("t1", "city", "杭州", confidence=0.9)
    assert (kv.read("profile", "t1") or {}).get("city") == "杭州"


def test_arbitration_failure_retries(kv):
    """LLM 挂 → 仲裁失败 → 不入档不标晋升，下次重试。"""
    kv.write("profile", "t1", {"city": "上海"})
    g = MemoryGatekeeper()

    def boom(messages, temperature=None, max_tokens=None):
        raise RuntimeError("llm down")

    g_arb = g
    g_arb._arbitrate = lambda *a, **k: None
    g_arb.process("t1", "city", "杭州", confidence=0.9)
    g_arb.process("t1", "city", "杭州", confidence=0.9)
    assert (kv.read("profile", "t1") or {}).get("city") == "上海"  # 旧值没动
    # 候选仍在池里（下次出现可重试）
    from data.sqlite_store import get_db

    with get_db()._lock:
        n = get_db()._conn.execute(
            "SELECT COUNT(*) FROM memory_candidates WHERE session_id='t1' AND promoted=0"
        ).fetchone()[0]
    assert n == 1


# ---------------------- C2 第一刀 / N4 生活面 ----------------------
def test_char_life_ensure_generates(kv):
    char_life.ensure("t1")
    rec = char_life.get("t1")
    assert rec.get("shape") == "懒散周末"
    assert len(rec.get("items") or []) == 1
    assert "记忆的书" in rec["items"][0]["detail"]


def test_consume_topic_retires_after_two_uses(kv):
    char_life.ensure("t1")
    first = char_life.consume_topic("t1")
    assert "记忆的书" in first
    char_life.consume_topic("t1")
    # 用满两次 → 退休，没有可用的了
    assert char_life.consume_topic("t1") == ""


def test_shape_line_only_today(kv):
    assert char_life.shape_line("t1") == ""      # 没生成过
    char_life.ensure("t1")
    assert "懒散周末" in char_life.shape_line("t1")
    # 过期（day 不是今天）→ 不注入
    kv.update("self", "t1", lambda r: {**r, "life": {**(r.get("life") or {}), "day": "2000-01-01"}})
    assert char_life.shape_line("t1") == ""


def test_self_identity_unaffected_by_life(kv):
    """生活面写进 self 记录，但不该污染身份冻结的键。"""
    char_life.ensure("t1")
    rec = kv.read("self", "t1")
    assert not rec.get("name")
    assert char_life.get("t1").get("shape")
