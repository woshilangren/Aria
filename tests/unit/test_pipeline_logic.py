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


# --------------- F→R17b：正文剥净后的提交落库（PreparedTurn 直测） ---------------
from orchestration.pipeline import DialoguePipeline  # noqa: E402
from orchestration.writeback import WritebackCoordinator  # noqa: E402
from shared.types import PreparedTurn  # noqa: E402
from tools.storage import KVStoreTool  # noqa: E402


class _FakeKeeper:
    """post_commit 里的 KEEPER 替身：追加与上下文都进内存。"""

    def __init__(self):
        self.msgs = []

    def append_turn(self, sid, user, assistant):
        self.msgs += [{"role": "user", "content": user},
                      {"role": "assistant", "content": assistant}]

    def get_context(self, sid):
        return list(self.msgs)

    def has_pending_summary(self, sid):
        return False

    def run_pending_summary(self, sid):
        return True


def _commit_prepared(monkeypatch, turn):
    """把协调器的外部依赖打桩，只验证"canonical 原样落库"。"""
    import orchestration.writeback as wb

    kv = KVStoreTool()
    monkeypatch.setattr(wb.services, "get", lambda name: kv)

    monkeypatch.setattr(wb, "ConversationDistiller", lambda: type(
        "D", (), {"distill_turn": lambda *a, **k: None}
    )())
    monkeypatch.setattr(wb, "PortraitBuilder", lambda: type(
        "P", (), {"refresh": lambda *a, **k: None}
    )())
    monkeypatch.setattr(wb, "RelationshipTracker", lambda: type(
        "T", (), {"update": lambda *a, **k: {"interaction_count": 1, "intimacy": 0},
                  "update_bookkeeping_only": lambda *a, **k: {}}
    )())
    coord = WritebackCoordinator(keeper=_FakeKeeper())
    receipt = coord.commit_prepared(turn)
    assert receipt["status"] == "committed", f"提交没成功: {receipt}"
    coord.post_commit(turn, receipt)
    # 副作用落点已从逐键 write 换成 commit_turn 事务（chat_log）——验证方式
    # 相应改为提交后从正式记录回读。
    return kv.read("session", turn.session_id)


def test_commit_prepared_voice_text_stored_clean(monkeypatch):
    """语音轮 [开心]@r 你好：剥离责任在流式装配器（push 前）；提交落库的是
    已剥净的 canonical——PreparedTurn 原样入 chat_log，无 @r 无情绪标记。"""
    turn = PreparedTurn(
        session_id="f-writeback", turn_id="t-f1", request_id="req-f1",
        request_digest="d", user_text="hi", assistant_text="你好",
        disposition="normal", emotion="happy", mode="voice",
    )
    written = _commit_prepared(monkeypatch, turn)
    assistant_records = [r for r in written if r.get("role") == "assistant"]
    assert assistant_records, "assistant 记录没落库"
    assert "@r" not in assistant_records[0]["text"]
    assert assistant_records[0]["text"] == "你好"


def test_commit_prepared_text_mode_strips_upstream(monkeypatch):
    """纯文本轮：canonical 由 astream 事件流拼出（已剥 @r）——提交只认
    canonical，任何带标记的原稿到了这里都说明上游漏剥（红）。"""
    turn = PreparedTurn(
        session_id="f-writeback-text", turn_id="t-f2", request_id="req-f2",
        request_digest="d", user_text="hi", assistant_text="你好",
        disposition="normal", mode="text",
    )
    written = _commit_prepared(monkeypatch, turn)
    assistant_records = [r for r in written if r.get("role") == "assistant"]
    assert assistant_records[0]["text"] == "你好"


# --------- F 补丁：<voice> 内 @r 逃过 _tag_checked 一次性开关 ---------
from orchestration.pipeline import _ReplyStreamer  # noqa: E402


def _frags(text, voice_mode=False):
    s = _ReplyStreamer(voice_mode)
    return s.feed(text) + s.finish(), s


def test_voice_inner_reaction_tag_stripped_after_body_sentence():
    """正文先跑掉 _tag_checked 后，<voice> 内的 @r 仍要被剥掉。

    形态 "你好。<voice>@r 你好呀</voice>"：_push_sentence 先消耗唯一一次
    _tag_checked，旧实现会让 <voice> 里的 @r 直接进 voice_text 被朗读。
    """
    frags, streamer = _frags("你好。<voice>@r 你好呀</voice>", voice_mode=True)
    voices = [f for f in frags if f[0] == "voice"]
    assert voices, "没有产出 voice 片段"
    assert "@r" not in voices[0][1]
    assert voices[0][1] == "你好呀"
    # 剥到 @r 要登记：这是"模型自标了本能反应"的信号，上层要用
    assert streamer.reaction_tagged is True


def test_voice_inner_no_reaction_tag_unaffected():
    """对照：正文/ <voice> 都不含 @r 时，voice_text 原样保留、reaction_tagged 为 False。"""
    frags, streamer = _frags("你好。<voice>你好呀</voice>", voice_mode=True)
    voices = [f for f in frags if f[0] == "voice"]
    assert voices, "没有产出 voice 片段"
    assert voices[0][1] == "你好呀"
    assert streamer.reaction_tagged is False


def test_voice_inner_email_not_touched():
    """<voice> 内中/尾部的 @r（邮箱之类）不能被误伤。"""
    frags, _s = _frags("好的。<voice>发到 a@r.com</voice>", voice_mode=True)
    voices = [f for f in frags if f[0] == "voice"]
    assert voices[0][1] == "发到 a@r.com"


# ------------------- R03：危机深度确认结果接线 -------------------

from shared.types import IntentResult, MemoryBundle, PerceptionExtras  # noqa: E402


class _FixedPerception:
    """固定返回初判结果的感知替身（不调 LLM）。"""

    def __init__(self, intent, emotion):
        self._intent = intent
        self._emotion = emotion

    def run(self, text, recent_context=None, session_id=""):  # noqa: ARG002
        return self._intent, self._emotion, "", PerceptionExtras()


class _EmptyRecall:
    def recall(self, query, session_id):  # noqa: ARG002
        return MemoryBundle()


class _CountingLLM:
    """深度确认调用的替身：记录调用次数并返回固定应答或抛错。"""

    def __init__(self, reply=None, exc=None):
        self.reply = reply
        self.exc = exc
        self.calls = 0

    def chat(self, messages, temperature=None, max_tokens=None):  # noqa: ARG002
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.reply


def _run_perceive(monkeypatch, llm):
    """搭好替身并真跑 _perceive：初判 = sad 强情绪（非危机，满足二次确认条件）。"""
    from shared.singletons import services

    emo = EmotionResult(emotion="sad", intensity=0.9)
    intent = IntentResult(intent="chat")
    from orchestration import pipeline as pl

    monkeypatch.setattr(pl, "PerceptionPipeline",
                        lambda: _FixedPerception(intent, emo))
    monkeypatch.setattr(pl, "MemoryRecaller", _EmptyRecall)
    monkeypatch.setattr(pl.KEEPER, "get_context", lambda sid: [])
    services.register("llm", llm)
    state = TurnState(user_text="我最近撑得很辛苦", session_id="r03")
    pl.DialoguePipeline()._perceive(state)
    return state, llm


def test_deep_confirm_crisis_sets_comfort_mode(monkeypatch):
    """初判非危机、二次确认危机 → is_crisis 与 comfort_mode 必须同时生效。

    缺陷（R03）：确认分支写的是 state.emotion，comfort_mode 却读旧局部
    emotion.is_crisis——确认结果被下游（compose 的 comfort 语气、writeback 的
    危机不沉淀边界）看到，安抚模式却没跟上。
    """
    state, llm = _run_perceive(monkeypatch, _CountingLLM(reply="yes"))
    assert state.emotion.is_crisis is True, "二次确认 yes 必须置危机"
    assert state.comfort_mode is True, "comfort_mode 必须读取确认后的最终结果"
    assert llm.calls == 1


def test_deep_confirm_failure_keeps_original_judgement(monkeypatch):
    """确认 LLM 挂 → 保留现有降级：不凭空变危机、不切安抚模式。"""
    state, llm = _run_perceive(monkeypatch, _CountingLLM(exc=RuntimeError("llm down")))
    assert state.emotion.is_crisis is False
    assert state.comfort_mode is False
    assert llm.calls == 1


def test_deep_confirm_no_keeps_original_judgement(monkeypatch):
    """确认回答 no → 保留原判定（sad，非危机）。"""
    state, _llm = _run_perceive(monkeypatch, _CountingLLM(reply="no"))
    assert state.emotion.is_crisis is False
    assert state.emotion.emotion == "sad"
    assert state.comfort_mode is False


def test_preconfirmed_crisis_skips_deep_confirm(monkeypatch):
    """初判已是危机（词表直判）→ 不烧确认调用，comfort_mode 直接生效。"""
    from shared.singletons import services

    emo = EmotionResult(emotion="crisis", intensity=1.0, is_crisis=True)
    intent = IntentResult(intent="comfort")
    from orchestration import pipeline as pl

    monkeypatch.setattr(pl, "PerceptionPipeline",
                        lambda: _FixedPerception(intent, emo))
    monkeypatch.setattr(pl, "MemoryRecaller", _EmptyRecall)
    monkeypatch.setattr(pl.KEEPER, "get_context", lambda sid: [])
    llm = _CountingLLM(reply="yes")
    services.register("llm", llm)
    state = TurnState(user_text="不想活了", session_id="r03b")
    pl.DialoguePipeline()._perceive(state)
    assert state.emotion.is_crisis is True
    assert state.comfort_mode is True
    assert llm.calls == 0, "初判危机走词表直判，不需要二次确认调用"
