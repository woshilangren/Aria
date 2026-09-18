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


def test_peek_does_not_burn_but_commit_does(kv):
    """I1：挑与烧分离——peek 不烧计数，commit 才烧，用满两次即退休。

    以前 quirks 掷骰子命中就调 consume_topic（挑+烧一体），而六个动作里只有
    off_topic / recall 真用素材，其余四种白烧一次 → 素材被提前掏空。
    """
    char_life.ensure("t1")
    first = char_life.peek_topic("t1")
    assert "记忆的书" in first
    # 只挑不烧：连挑三次还是同一条，计数一格没动（这条就是 I1 的回归闸）
    assert char_life.peek_topic("t1") == first
    assert char_life.peek_topic("t1") == first
    char_life.commit_topic("t1", first)
    assert char_life.peek_topic("t1") == first      # 烧了一次，还可用
    char_life.commit_topic("t1", first)
    assert char_life.peek_topic("t1") == ""         # 用满两次 → 退休


def test_commit_unknown_topic_is_noop(kv):
    """烧一条已经不在池里的素材：什么都不该动，更不能烧到别人头上。"""
    char_life.ensure("t1")
    real = char_life.peek_topic("t1")
    char_life.commit_topic("t1", "早就被挤掉的一条")
    assert char_life.peek_topic("t1") == real       # 真素材一格没烧


class _StubRandom:
    """钉死 quirks 的随机源：骰子必过闸，且落在指定的那个动作上。"""

    def __init__(self, uniform_value: float):
        self._u = uniform_value

    def random(self) -> float:
        return 0.0                    # 必过 quirk_rate 闸（0.0 >= rate 恒假）

    def uniform(self, a, b) -> float:
        return self._u

    def choice(self, seq):
        return seq[0]


def _used_count(session_id: str) -> int:
    items = (char_life.get(session_id).get("items") or [])
    return sum(int(it.get("used_count") or 0) for it in items)


# 无 flaws / 无 distilled / stage=初识 时桌面是 snark3 off_topic2 pride2 recall2，
# total=9，累计边界 3 / 5 / 7 / 9（有 life_topic 才上得了 recall 这张桌）
@pytest.mark.parametrize("pick,action,burns", [
    (1.0, "snark", False),
    (4.0, "off_topic", True),
    (6.0, "pride", False),
    (8.0, "recall", True),
])
def test_quirk_roll_only_burns_topic_when_used(kv, monkeypatch, pick, action, burns):
    """I1 调用方侧：只有 off_topic / recall 真嵌了素材才烧计数。

    改之前 roll 一命中就 consume_topic，snark / pride 这两种命中白烧一次——
    约半数素材被白白消耗，used_count>=2 就退休，她的生活面被提前掏空。
    """
    from capability import quirks
    from shared.types import MemoryBundle

    char_life.ensure("t1")
    topic = char_life.peek_topic("t1")
    assert topic                                    # 前置：确实有素材可用
    monkeypatch.setattr(quirks, "random", _StubRandom(pick))
    monkeypatch.setattr(quirks, "load_app_config",
                        lambda: {"personality": {"quirk_rate": 1.0}})

    directive = quirks.QuirkDirector().roll(MemoryBundle(), stage="初识",
                                            mood="平常", session_id="t1")
    assert directive                                # 骰子命中就不许返回空串
    assert (topic in directive) is burns
    # 断言**确切计数**，不用 `(_used_count()==1) is burns`：后者在多烧一次时
    # used_count 变 2，`2==1` 为假正好和 burns=False 撞上，snark/pride 会假绿
    # （反证跑出来过这个坑）。
    assert _used_count("t1") == (1 if burns else 0)


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
