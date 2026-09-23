"""批次7 单测：S4 记忆闸门（候选晋升+仲裁）、C2 第一刀/N4 生活面（char_life）。"""

from datetime import datetime, timedelta, timezone

import pytest

from capability import char_life, self_identity
from capability.memory import MemoryGatekeeper, PortraitBuilder
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
        # R05b：仲裁故障开关。置 True 时仲裁调用真的抛错，走真实 _arbitrate
        # 的失败分支——以前那条测试是把被测方法本身换成空函数，测的是替身。
        self.arbitration_down = False

    def chat(self, messages, temperature=None, max_tokens=None):
        system = messages[0]["content"]
        if "档案仲裁员" in system:
            if self.arbitration_down:
                raise RuntimeError("llm down")
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
    g.process("t1", "city", "杭州", confidence=0.9, source_message_id="m1")
    assert not (kv.read("profile", "t1") or {}).get("city")
    # 第二次：达到门槛 → 仲裁 update → 入档
    g.process("t1", "city", "杭州", confidence=0.9, source_message_id="m2")
    assert (kv.read("profile", "t1") or {}).get("city") == "杭州"


def test_confidence_floor_rejects(kv):
    MemoryGatekeeper().process("t1", "city", "杭州", confidence=0.3, source_message_id="m1")
    kv_arb = MemoryGatekeeper()
    kv_arb.process("t1", "city", "杭州", confidence=0.5, source_message_id="m2")  # 两次都低置信
    assert not (kv.read("profile", "t1") or {}).get("city")


def test_arbitration_keep_preserves_old_value(kv):
    kv.write("profile", "t1", {"city": "上海"})
    g = MemoryGatekeeper()
    kv.scripted_llm.arbitrate_action = "keep"
    # 第一次入池 + 第二次达到门槛 → 仲裁 keep → 旧值保留
    g.process("t1", "city", "杭州", confidence=0.9, source_message_id="m1")
    g.process("t1", "city", "杭州", confidence=0.9, source_message_id="m2")
    assert (kv.read("profile", "t1") or {}).get("city") == "上海"


def test_arbitration_update_replaces_old(kv):
    kv.write("profile", "t1", {"city": "上海"})
    g = MemoryGatekeeper()
    kv.scripted_llm.arbitrate_action = "update"
    g.process("t1", "city", "杭州", confidence=0.9, source_message_id="m1")
    g.process("t1", "city", "杭州", confidence=0.9, source_message_id="m2")
    assert (kv.read("profile", "t1") or {}).get("city") == "杭州"


def test_arbitration_failure_retries(kv):
    """LLM 挂 → 真实 _arbitrate 吃到故障返回 None → 不入档不标晋升，下次重试。

    R05b：以前这条测试把被测方法本身替换成空函数（`_arbitrate = lambda: None`），
    断言的其实是替身的行为；现在给替身 LLM 加故障开关，让**真实**仲裁路径
    （chat 调用抛错 → _arbitrate 捕获 → 返回 None → process 放弃晋升）跑全程。
    """
    kv.write("profile", "t1", {"city": "上海"})
    kv.scripted_llm.arbitration_down = True
    g = MemoryGatekeeper()
    g.process("t1", "city", "杭州", confidence=0.9, source_message_id="m1")
    g.process("t1", "city", "杭州", confidence=0.9, source_message_id="m2")
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


def test_quirk_roll_deferred_burn_waits_for_commit(kv, monkeypatch):
    """R17d：deferred_burns 传入时烧计数不当场落账。

    取消/降级轮的指令已经拼出去了，但她的生活素材不该被没提交的轮白白
    消耗——挑与烧分离（I1）延伸到提交边界：烧不烧等写回协调器裁决。
    """
    from capability import quirks
    from shared.types import MemoryBundle

    char_life.ensure("t1")
    topic = char_life.peek_topic("t1")
    assert topic
    monkeypatch.setattr(quirks, "random", _StubRandom(4.0))  # 命中 off_topic（嵌素材）
    monkeypatch.setattr(quirks, "load_app_config",
                        lambda: {"personality": {"quirk_rate": 1.0}})

    burns: list = []
    directive = quirks.QuirkDirector().roll(
        MemoryBundle(), stage="初识", mood="平常", session_id="t1",
        deferred_burns=burns,
    )
    assert directive and topic in directive
    assert burns == [("t1", topic)], "该把待烧素材交给调用方"
    assert _used_count("t1") == 0, "延后烧不许当场落账"


def test_search_memory_touch_flag_controls_heat(kv, monkeypatch):
    """R17d：touch=False 召回不记热度（轮内路径，提交后补记）；默认保持原行为。"""
    from data.sqlite_store import get_db

    rows = [("m1", "甲", 0.3, "2026-09-23T10:00:00")]
    tool = _make_search_tool(monkeypatch, rows)
    got = tool.search_memory("随便搜搜", top_k=5, touch=False)
    assert got and got[0]["id"] == "m1"
    stats = get_db().memory_stats_bulk(["m1"])
    assert not stats.get("m1"), "touch=False 不许记热度"

    tool.search_memory("随便搜搜", top_k=5)  # 默认 touch=True：turn 外的独立检索口子
    stats = get_db().memory_stats_bulk(["m1"])
    assert (stats.get("m1") or {}).get("count") == 1


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


# ------------------- R04：向量写入失败结果必须被消费 -------------------

class _VectorStub:
    """向量库替身：可脚本化返回值/异常。"""

    def __init__(self, result=True, exc=None):
        self.result = result
        self.exc = exc
        self.calls = 0

    def upsert_memory(self, item):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


def _remember_with(monkeypatch, vector):
    """注册替身向量库并拦截 _enqueue_retry，返回 (记录列表, 记录器调用数)。"""
    from capability import memory as mem_mod
    from shared.singletons import services

    registered = []
    monkeypatch.setattr(mem_mod, "_enqueue_retry", lambda kind, payload: registered.append((kind, payload)))
    services.register("vector_store", vector)
    mem_mod.remember_note("r04-session", "一条值得记住的事", kind="event")
    return registered, vector


def test_remember_note_false_is_consumed(monkeypatch):
    """upsert 返回 False（集合未装配/内部失败）→ 走失败入口登记一次。"""
    registered, vector = _remember_with(monkeypatch, _VectorStub(result=False))
    assert vector.calls == 1
    assert len(registered) == 1, f"False 必须消费且只登记一次，实测 {registered}"
    assert registered[0][0] == "vector_memory"
    assert "一条值得记住的事" in registered[0][1].get("content", "")


def test_remember_note_exception_is_consumed(monkeypatch):
    """upsert 抛异常 → 同一个失败入口登记一次（不许无声蒸发）。"""
    registered, vector = _remember_with(monkeypatch, _VectorStub(exc=RuntimeError("chroma down")))
    assert vector.calls == 1
    assert len(registered) == 1


def test_remember_note_true_enqueues_nothing(monkeypatch):
    """明确 True 才算成功：成功路径零补偿登记。"""
    registered, _v = _remember_with(monkeypatch, _VectorStub(result=True))
    assert registered == [], "成功不许排重试"


def test_search_timestamp_offset_equivalence(monkeypatch):
    """R09b：同一条记忆的 timestamp 写成 naive / Z / +08:00 → 衰减天数一致、
    分数相同（任何机器时区下都成立）。"""
    now = datetime.now().astimezone()
    ts_base = (now - timedelta(days=3)).replace(microsecond=0)
    rows = [
        ("a_naive", "甲", 0.3, ts_base.replace(tzinfo=None).isoformat()),
        ("b_z", "乙", 0.3, ts_base.astimezone(timezone.utc).isoformat()),
        ("c_off", "丙", 0.3, ts_base.isoformat()),
        ("d_future", "丁", 0.3, (now + timedelta(days=2)).isoformat()),  # 未来：钳制成刚发生
    ]
    tool = _make_search_tool(monkeypatch, rows)
    got = {g["id"]: g for g in tool.search_memory("随便搜搜", top_k=5)}
    assert got["a_naive"]["score"] == pytest.approx(got["b_z"]["score"])
    assert got["a_naive"]["score"] == pytest.approx(got["c_off"]["score"])
    # 未来时间戳：负间隔钳制成 0 天 → 衰减=1（当作刚发生，旧行为）
    assert got["d_future"]["score"] >= got["a_naive"]["score"]


# ------------------- R08：召回负分反转（语义置信分/排序分分离） -------------------
# 补记（附二十七）：上一批的追加命令被工具层错误打断后未重试，这 4 条测试
# 当时没有落盘（R08 提交信息声称有它们）——本文件补齐，教训写进日志。

def _make_search_tool(monkeypatch, rows):
    """搭 search_memory 的替身环境：rows = [(id, doc, dist, timestamp)]。

    query 返回固定形状；embedding 替身；热度统计走真 tmp 库（conftest 隔离）。
    """
    from tools.storage import VectorStoreTool

    ids = [r[0] for r in rows]
    docs = [r[1] for r in rows]
    dists = [r[2] for r in rows]
    metas = [{"timestamp": r[3], "valence": 0.0, "arousal": 0.0} for r in rows]

    class _FakeCol:
        def query(self, **kwargs):
            return {"ids": [ids], "documents": [docs],
                    "metadatas": [metas], "distances": [dists]}

    tool = VectorStoreTool()
    tool._cols = {"distilled_memory": _FakeCol()}

    class _FakeEmbedder:
        def embed_query(self, q):
            return [0.0]

    tool._embedder = _FakeEmbedder()
    return tool


def test_search_distance_over_one_stays_positive(monkeypatch):
    """d>1：旧 1-d 为负、负分乘衰减/热度会让旧记忆排前——新公式必须非负。"""
    now = datetime.now()
    ts_now = now.isoformat(timespec="seconds")
    ts_old = (now - timedelta(days=40)).isoformat(timespec="seconds")
    tool = _make_search_tool(monkeypatch, [
        ("m1", "近的那条", 1.5, ts_now),
        ("m2", "远的那条", 1.5, ts_old),
    ])
    got = tool.search_memory("随便搜搜", top_k=5)
    assert got, "d>1 的结果不该被丢光"
    sims = {g["id"]: g["similarity"] for g in got}
    assert sims["m1"] == pytest.approx(1.0 / 2.5)
    scores = {g["id"]: g["score"] for g in got}
    assert scores["m1"] > scores["m2"] > 0, f"分数必须非负且新的更靠前: {scores}"


def test_search_bad_distances_never_rank_first(monkeypatch):
    """NaN / Infinity / 负距离：异常数据跳过，绝不进榜首冒充最相关。"""
    now = datetime.now().isoformat(timespec="seconds")
    tool = _make_search_tool(monkeypatch, [
        ("bad_nan", "nan 文档", "nan", now),
        ("bad_inf", "inf 文档", float("inf"), now),
        ("bad_neg", "neg 文档", -0.5, now),
        ("good", "正常的那条", 0.5, now),
    ])
    got = tool.search_memory("随便搜搜", top_k=5)
    ids = [g["id"] for g in got]
    assert ids == ["good"], f"异常距离必须全部跳过，实测 {ids}"


def test_search_similarity_and_score_are_separate(monkeypatch):
    """语义置信分（similarity）与排序分（score）必须分开：后者=前者×衰减×热度。"""
    now = datetime.now().isoformat(timespec="seconds")
    tool = _make_search_tool(monkeypatch, [("m1", "一条记忆", 0.5, now)])
    got = tool.search_memory("随便搜搜", top_k=5)
    g = got[0]
    assert g["similarity"] == pytest.approx(1.0 / 1.5)
    assert g["score"] < g["similarity"], "排序分乘了衰减（<1），必须低于语义置信分"
    assert g["score"] > 0


def test_s7_uncertain_hedge_fixed_samples():
    """S7 低置信边界的固定样本：校准语义分 <0.6 含糊；legacy 排序分 <0.35 兼容。"""
    from capability.persona_engine import _needs_uncertain_hedge

    assert _needs_uncertain_hedge({"similarity": 0.59}) is True
    assert _needs_uncertain_hedge({"similarity": 0.61}) is False
    assert _needs_uncertain_hedge({"similarity": float("nan")}) is False
    assert _needs_uncertain_hedge({"score": 0.3}) is True     # legacy 兼容
    assert _needs_uncertain_hedge({"score": 0.4}) is False
    assert _needs_uncertain_hedge({}) is False                # 双缺：宁少勿滥

# ------------------- R14c：消费端统一（学习素材只取可学习的已提交正常轮） -------------------

def _insert_chat_row(session_id, role, text, disposition, mode="text"):
    from data.sqlite_store import get_db

    db = get_db()
    with db._lock, db._conn:
        db._conn.execute(
            "INSERT INTO chat_log (session_id, role, content, intent, emotion, mode, "
            "created_at, turn_id, disposition, source_review_status, reason_code) "
            "VALUES (?, ?, ?, '', '', ?, ?, ?, ?, 'accepted', '')",
            (session_id, role, text, mode,
             datetime.now().isoformat(timespec="seconds"),
             "t-" + text[:6], disposition),
        )


def test_learnable_recent_chat_filters_degraded(kv):
    """R14c：learnable 读口只出 normal + legacy（分类为空的旧行）；降级轮的
    兜底正文不许变成画像/身份的学习素材。"""
    _insert_chat_row("r14c", "assistant", "正常轮一", "normal")
    _insert_chat_row("r14c", "assistant", "降级兜底话", "degraded")
    _insert_chat_row("r14c", "assistant", "危机轮", "crisis")
    _insert_chat_row("r14c", "assistant", "legacy 旧记录", "")

    rows = kv.recent_chat("r14c", 100, learnable=True)
    texts = [r["text"] for r in rows]
    assert "正常轮一" in texts and "legacy 旧记录" in texts
    assert "降级兜底话" not in texts and "危机轮" not in texts
    # 默认全量（聊天展示/短期记忆恢复）：降级轮真实发布过，照样在
    all_rows = kv.recent_chat("r14c", 100)
    assert "降级兜底话" in [r["text"] for r in all_rows]


def test_diary_compose_marks_degraded_rows(kv):
    """R14c：日记把降级轮当"回复失败"的客观记录——transcript 里明确标注，
    不许把模板兜底话当成她说了段有意义的话来写日记。"""
    from capability.diary import DiaryWriter

    captured = {}

    class _CaptureLLM:
        def chat(self, messages, temperature=None, max_tokens=None):
            captured["content"] = messages[-1]["content"]
            return '{"items": [], "diary": "今天没聊什么。"}'

    kv.scripted_llm_orig = None
    import shared.singletons as singles

    old_llm = singles.services._services.get("llm")
    singles.services.register("llm", _CaptureLLM())
    try:
        chats = [
            {"role": "user", "text": "今天天气怎么样", "time": "2026-09-23T10:00:00",
             "disposition": "normal"},
            {"role": "assistant", "text": "这轮没接上，稍后再试试", "time": "2026-09-23T10:00:05",
             "disposition": "degraded"},
            {"role": "assistant", "text": "晴天呢，出门带伞也别忘了防晒", "time": "2026-09-23T10:00:10",
             "disposition": "normal"},
        ]
        diary = DiaryWriter()._compose(chats, session_id="r14c-diary")
        assert diary == "今天没聊什么。"
    finally:
        singles.services.register("llm", old_llm)

    t = captured["content"]
    assert "（这轮回复失败）：这轮没接上" in t, "降级轮必须带客观标注"
    assert "晴天呢" in t

# ------------------- R11a：同批去重止损 -------------------

def test_same_batch_duplicate_counts_once(kv):
    """R11a：同一次抽取里同字段同值最多命中一次——同批重复不许凑次数。

    以前一个窗口里模型把"杭州"提两遍（两条 fact / 列表里两项）就 hits=2
    直达晋升门槛，"≥2 次独立出现"被同批重复架空。杭州/杭州市按 candidate_hit
    同一套归一化算同一件。"""
    from data.sqlite_store import get_db

    pool = [("101", "我住在杭州，气候还行"), ("102", "我只喝咖啡")]
    PortraitBuilder()._emit_facts("t11a", [
        {"field": "city", "value": "杭州", "confidence": 0.9, "quote": "我住在杭州"},
        {"field": "city", "value": "杭州市", "confidence": 0.9, "quote": "我住在杭州"},
        {"field": "city", "value": "杭州", "confidence": 0.9, "quote": "我住在杭州"},
        {"field": "likes", "value": ["咖啡", "咖啡"], "confidence": 0.9, "quote": "我只喝咖啡"},
    ], source_pool=pool)
    rows = get_db()._conn.execute(
        "SELECT field, content, hits FROM memory_candidates WHERE session_id='t11a' ORDER BY id"
    ).fetchall()
    assert len(rows) == 2, f"同批去重后只应剩 city+likes 两条候选，实测 {rows}"
    assert all(r[2] == 1 for r in rows), f"每条候选 hits 必须=1，实测 {rows}"


def test_cross_batch_still_accumulates(kv):
    """R11a 只砍同批重复：跨批次（两次独立抽取）照样正常累积，不误伤晋升闸。"""
    from data.sqlite_store import get_db

    pool = [("201", "我住在杭州"), ("202", "我住杭州市，习惯了")]
    PortraitBuilder()._emit_facts("t11b", [
        {"field": "city", "value": "杭州", "confidence": 0.9, "quote": "我住在杭州"},
    ], source_pool=pool)
    PortraitBuilder()._emit_facts("t11b", [
        {"field": "city", "value": "杭州市", "confidence": 0.9, "quote": "我住杭州市"},
    ], source_pool=pool)
    rows = get_db()._conn.execute(
        "SELECT hits FROM memory_candidates WHERE session_id='t11b'"
    ).fetchall()
    assert rows == [(2,)], f"跨批两次独立出现应累积 hits=2，实测 {rows}"

# ------------------- R11b：稳定来源证据 -------------------

def test_same_source_message_does_not_double_count(kv):
    """R11b：同一来源消息重复命中（窗口重叠/重试/重放）不增加 hits。

    证据键 = (session, field, normalized_value, source_message_id)——同一条
    消息对同一候选只有一票。"""
    from data.sqlite_store import get_db

    db = get_db()
    g = MemoryGatekeeper()
    g.process("t11c", "city", "杭州", quote="我住在杭州", confidence=0.9,
              source_message_id="m1")
    hits1, _ = db.candidate_hit("t11c", "city", "杭州", source_message_id="m1")
    hits2, _ = db.candidate_hit("t11c", "city", "杭州市", source_message_id="m1")
    assert hits1 == 1 and hits2 == 1, f"同源重复不许涨 hits，实测 {hits1}/{hits2}"
    # 换一条真实新消息：正常累积
    hits3, _ = db.candidate_hit("t11c", "city", "杭州", source_message_id="m2")
    assert hits3 == 2
    # 来源 id 留痕可审计
    row = db._conn.execute(
        "SELECT source_ids FROM memory_candidates WHERE session_id='t11c'"
    ).fetchone()
    import json as _json

    assert sorted(_json.loads(row[0])) == ["m1", "m2"]


def test_unattributable_extraction_is_dropped(kv):
    """R11b：引文对应不上真实用户消息的抽取不算新增证据——丢弃，不入池。"""
    from data.sqlite_store import get_db

    accepted = PortraitBuilder()._emit_facts("t11d", [
        {"field": "city", "value": "杭州", "confidence": 0.9, "quote": "编造的原话"},
        {"field": "city", "value": "上海", "confidence": 0.9, "quote": ""},
    ], source_pool=[("301", "我住在杭州")])
    assert accepted == [], f"对不上来源的抽取必须全弃，实测 {accepted}"
    rows = get_db()._conn.execute(
        "SELECT * FROM memory_candidates WHERE session_id='t11d'"
    ).fetchall()
    assert rows == [], "不许入池"


def test_gate_rejects_missing_source_id(kv):
    """R11b：闸门拒绝无来源消息 id 的抽取（守门不靠调用方自觉）。"""
    from data.sqlite_store import get_db

    MemoryGatekeeper().process("t11e", "city", "杭州", confidence=0.9)
    rows = get_db()._conn.execute(
        "SELECT * FROM memory_candidates WHERE session_id='t11e'"
    ).fetchall()
    assert rows == []


def test_recent_chat_rows_carry_message_id(kv):
    """R11b：历史读口带 chat_log 行 id——证据归属的真实来源键。"""
    _insert_chat_row("t11f", "user", "我住在杭州", "normal")
    rows = kv.recent_chat("t11f", 10)
    assert rows and isinstance(rows[-1]["id"], int), f"行必须带 id，实测 {rows[-1]}"

# ------------------- R11c：keep 结果可复用而不永久拉黑 -------------------

def _counting_arb(kv, calls):
    """给仲裁 LLM 计数：数 chat 被调几次 = 烧了几次仲裁。"""
    orig = kv.scripted_llm.chat

    def _wrap(messages, temperature=None, max_tokens=None):
        calls["n"] += 1
        return orig(messages, temperature=temperature, max_tokens=max_tokens)

    kv.scripted_llm.chat = _wrap


def test_keep_reused_for_same_evidence_and_profile(kv):
    """R11c：相同证据 + 相同档案现态复用 keep——第二次不烧仲裁 LLM。"""
    kv.write("profile", "t11g", {"city": "上海"})
    kv.scripted_llm.arbitrate_action = "keep"
    calls = {"n": 0}
    _counting_arb(kv, calls)
    g = MemoryGatekeeper()
    # 第一轮候选攒到门槛：烧 1 次仲裁 → keep
    g.process("t11g", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m1")
    g.process("t11g", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m2")
    assert calls["n"] == 1
    assert (kv.read("profile", "t11g") or {}).get("city") == "上海"
    # 用户把同一句话再说两遍：新候选再次到门槛 → **复用 keep，0 次仲裁**
    g.process("t11g", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m3")
    g.process("t11g", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m4")
    assert calls["n"] == 1, f"相同证据不许再次付费仲裁，实测 {calls['n']} 次"
    assert (kv.read("profile", "t11g") or {}).get("city") == "上海"


def test_new_evidence_reenters_arbitration(kv):
    """R11c：新的独立证据（没见过的引文）可重新仲裁——不是永久拉黑。"""
    kv.write("profile", "t11h", {"city": "上海"})
    kv.scripted_llm.arbitrate_action = "keep"
    calls = {"n": 0}
    _counting_arb(kv, calls)
    g = MemoryGatekeeper()
    g.process("t11h", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m1")
    g.process("t11h", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m2")
    assert calls["n"] == 1
    # 用户换了说法、说了新的理由：新证据 → 重新仲裁
    kv.scripted_llm.arbitrate_action = "update"
    g.process("t11h", "city", "杭州", quote="工作定在杭州了，下周就搬", confidence=0.9,
              source_message_id="m3")
    g.process("t11h", "city", "杭州", quote="房子都租好了，就等搬家", confidence=0.9,
              source_message_id="m4")
    assert calls["n"] > 1, "新证据必须重新仲裁"
    assert (kv.read("profile", "t11h") or {}).get("city") == "杭州"


def test_profile_change_reenters_arbitration(kv):
    """R11c：档案现态变了（旧值不再一致）→ 同样的证据也要重裁。"""
    kv.write("profile", "t11i", {"city": "上海"})
    kv.scripted_llm.arbitrate_action = "keep"
    calls = {"n": 0}
    _counting_arb(kv, calls)
    g = MemoryGatekeeper()
    g.process("t11i", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m1")
    g.process("t11i", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m2")
    assert calls["n"] == 1
    # 档案被别的路径改掉（如手动修正），再出现同样的话：语境不同了，重裁
    kv.write("profile", "t11i", {"city": "苏州"})
    kv.scripted_llm.arbitrate_action = "update"
    g.process("t11i", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m5")
    g.process("t11i", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m6")
    assert calls["n"] == 2, "档案变化必须重裁"
    assert (kv.read("profile", "t11i") or {}).get("city") == "杭州"


def test_update_verdict_drops_keep_cache(kv):
    """R11c：update 裁决清掉同值 keep 缓存，避免陈旧裁决残留。"""
    from data.sqlite_store import get_db

    kv.write("profile", "t11j", {"city": "上海"})
    kv.scripted_llm.arbitrate_action = "keep"
    g = MemoryGatekeeper()
    g.process("t11j", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m1")
    g.process("t11j", "city", "杭州", quote="我要搬去杭州", confidence=0.9,
              source_message_id="m2")
    assert get_db()._conn.execute("SELECT COUNT(*) FROM arbitration_keep").fetchone()[0] == 1
    kv.scripted_llm.arbitrate_action = "update"
    g.process("t11j", "city", "杭州", quote="真搬了，杭州见", confidence=0.9,
              source_message_id="m3")
    g.process("t11j", "city", "杭州", quote="户口都迁过去了", confidence=0.9,
              source_message_id="m4")
    assert get_db()._conn.execute("SELECT COUNT(*) FROM arbitration_keep").fetchone()[0] == 0
