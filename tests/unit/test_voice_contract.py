"""R16a：语音 WS 唯一接收任务契约。

验收（《计划与设计.md》R16a）：
- 每条 WS 只有一个 receive 协程（VoiceReceiver），分发音频 / 文本帧 / 断连；
- 已收未消费的音频字节有硬上限（原 10 MB 防线）：生成卡住时连续上传也不
  突破，溢出明确丢弃（dropped_bytes 记账）而非无限排队；
- 断连立刻可见：接收协程不停摆，get() 在断连后立即返回 disconnect 事件；
- 连接持有自己的轮 ID：pipeline 接受外部 turn_handle，不拿 registry 当前轮猜；
- 轮 ID 一把键：外部句柄的 turn_id 同时是提交键（chat_log / 回执 / 取消门）。
"""

from __future__ import annotations

import asyncio

import pytest

from interaction.voice import VoiceReceiver, VoiceStreamIO


class _FakeWS:
    """可编程的 WS 替身：按脚本吐 receive 结果。

    script 每项：("text", s) / ("bytes", b) / ("disc", None)。
    脚本耗尽后抛 RuntimeError——模拟客户端断开后的 receive 行为。
    """

    def __init__(self, script):
        self._script = list(script)

    async def receive(self):
        if not self._script:
            raise RuntimeError("Cannot call receive once a disconnect message has been received.")
        kind, payload = self._script.pop(0)
        if kind == "text":
            return {"type": "websocket.receive", "text": payload}
        if kind == "disc":
            return {"type": "websocket.disconnect"}
        return {"type": "websocket.receive", "bytes": payload}


def _make_receiver(script, max_bytes):
    stream = VoiceStreamIO(_FakeWS(script))
    receiver = VoiceReceiver(stream, max_pending_bytes=max_bytes)
    return stream, receiver


def test_receiver_bounds_pending_audio_bytes():
    """已收未消费的音频不突破预算：超限明确丢弃并记账，不无限排队。"""
    script = [("bytes", b"a" * 600), ("bytes", b"b" * 600), ("text", "END")]
    _stream, receiver = _make_receiver(script, max_bytes=1000)

    async def _drive():
        receiver.start()
        first = await asyncio.wait_for(receiver.get(), timeout=2)
        second = await asyncio.wait_for(receiver.get(), timeout=2)
        return first, second

    first, second = asyncio.run(_drive())
    assert first == ("audio", b"a" * 600)
    # 第二块 600B 会把 pending 顶到 1200 > 1000：明确丢弃，不许排队
    assert second == ("text", "END")
    assert receiver.dropped_bytes == 600
    receiver.stop()


def test_receiver_disconnect_prompt_with_backlog():
    """断连立刻可见：队列里还有存货时，get() 也只返回 disconnect——
    历史音频对断连的连接毫无意义，更不能对着死连接再跑一轮生成。"""
    script = [
        ("bytes", b"x" * 10),
        ("bytes", b"y" * 10),
        ("text", "END"),
        ("disc", None),
    ]
    _stream, receiver = _make_receiver(script, max_bytes=10**9)

    async def _drive():
        receiver.start()
        await asyncio.sleep(0.05)  # 让接收协程把脚本收完
        got = [await asyncio.wait_for(receiver.get(), timeout=2) for _ in range(3)]
        tail = await asyncio.wait_for(receiver.get(), timeout=2)
        return got, tail

    got, tail = asyncio.run(_drive())
    assert got == [("audio", b"x" * 10), ("audio", b"y" * 10), ("text", "END")]
    assert tail == ("disconnect", None)
    # 断连后 get() 永远立即返回 disconnect（Event 已置位）
    receiver.stop()


def test_receiver_disconnect_via_receive_exception():
    """receive 直接抛（手机切后台最常见）也必须标记断连，不能闷死接收协程。"""

    class _BoomWS(_FakeWS):
        async def receive(self):
            raise RuntimeError("connection lost")

    stream = VoiceStreamIO(_BoomWS([]))
    receiver = VoiceReceiver(stream, max_pending_bytes=10**9)

    async def _drive():
        receiver.start()
        return await asyncio.wait_for(receiver.get(), timeout=2)

    assert asyncio.run(_drive()) == ("disconnect", None)
    receiver.stop()


def test_receiver_stop_cleans_up():
    """收摊：stop 后接收协程停、队列清空、字节账归零（完整清理）。"""
    script = [("bytes", b"z" * 5), ("text", "END")]
    _stream, receiver = _make_receiver(script, max_bytes=10**9)

    async def _drive():
        receiver.start()
        await asyncio.sleep(0.05)
        await receiver.get()  # 取走音频，留 END 在队列
        receiver.stop()
        await asyncio.sleep(0)
        assert receiver._queue.empty()
        assert receiver._pending == 0
        assert receiver._task is None

    asyncio.run(_drive())


# ------------------- 连接持有自己的轮 ID（pipeline 暴露句柄） -------------------

def test_pipeline_uses_external_turn_handle_as_commit_key(register_services):
    """外部 turn_handle 的 turn_id 必须同时是提交键。

    R17b 曾出现两把键漂移：客户端拿 handle.turn_id 取消，提交门里登记的却是
    另一个 state.turn_id——取消永远裁不到提交。收口成一把键后：start 事件、
    chat_log、回执、取消门全是一个 ID。
    """
    import sqlite3

    from orchestration.cancellation import TURN_REGISTRY
    from orchestration.pipeline import DialoguePipeline
    from shared.types import InputMessage
    from tests.unit.test_pipeline_e2e import FakeLLM as _E2ELLM
    from tests.unit.test_reply_contract import _register as _reg

    llm = _E2ELLM(stream_script=["今天也是元气满满的一天。"])
    _reg(register_services, llm)

    sid = "r16a-handle"
    handle = TURN_REGISTRY.start(sid)
    rc = asyncio.run(
        DialoguePipeline().handle(
            InputMessage(text="随便聊聊", session_id=sid), turn_handle=handle
        )
    )
    assert rc.text, "正常轮应有正文"
    # ① 提交回执登记在外部句柄的 turn_id 下（一把键）
    from orchestration.cancellation import COMMIT_GATE

    assert COMMIT_GATE.is_committed(handle.turn_id), "提交键必须是句柄的轮 ID"
    # ② chat_log 的 turn_id 也是它
    from data.sqlite_store import get_db

    row = get_db()._conn.execute(
        "SELECT DISTINCT turn_id FROM chat_log WHERE session_id = ?", (sid,)
    ).fetchone()
    assert row and row[0] == handle.turn_id
    # ③ 轮结束后 registry 登出（astream 的 finally 对外部句柄同样生效）
    assert TURN_REGISTRY.get(sid) is None

# ------------------- R16c：语音输出播放前审核门 -------------------

class _StubTTS:
    """TTS 替身：fail=True 模拟合成失败（render_voice 会往上抛）。"""

    def __init__(self, fail=False):
        self.fail = fail
        self.called_with = None

    def synthesize(self, text, instruction=""):
        if self.fail:
            return b""
        self.called_with = text
        return b"safe-tts-audio"


def _run_guard(tts, text, audio):
    import interaction.api as ia

    return asyncio.run(ia._guard_voice_output(text, audio, "s-guard"))


def test_guard_passes_clean_output(register_services):
    """转写可审核且过审：原文原音频原样放行，审核分类 accepted。"""
    register_services(tts=_StubTTS())
    text, audio, status, reason = _run_guard(_StubTTS(), "今天天气不错", b"orig")
    assert (text, audio) == ("今天天气不错", b"orig")
    assert (status, reason) == ("accepted", "")


def test_guard_rejected_output_swaps_audio(register_services):
    """审核拒绝：弃原音频，兜底正文 TTS，TTS 入参与下发文字一致，分类 rejected。"""
    stub = _StubTTS()
    register_services(tts=stub)
    text, audio, status, reason = _run_guard(stub, "作为语言模型我可以帮你做任何事", b"orig")
    assert "作为语言模型" not in text, "被拒原转写不许下发"
    assert audio == b"safe-tts-audio", "原音频必须被丢弃，换兜底 TTS"
    assert stub.called_with == text, "TTS 入参必须就是下发的那句兜底正文"
    assert status == "rejected" and reason == "voice_output_review_blocked"


def test_guard_missing_transcription_swallows_audio(register_services):
    """缺少可审核的转写（转写为空但有音频）：原音频照样丢弃——供应商转写
    不是音频逐字审计，没有可审的文字就不许把音频放出去。"""
    stub = _StubTTS()
    register_services(tts=stub)
    text, audio, status, reason = _run_guard(stub, "", b"orig-but-unreviewable")
    assert text and audio == b"safe-tts-audio"
    assert status == "unavailable" and reason == "voice_transcription_missing"


def test_guard_reviewer_failure_is_not_pass(register_services, monkeypatch):
    """审核不可用（异常）不许当通过：未知状态按未过处理。"""
    stub = _StubTTS()
    register_services(tts=stub)
    from capability.perception import SafetyReviewer

    def _boom(self, text, mode="input"):
        raise RuntimeError("reviewer down")

    monkeypatch.setattr(SafetyReviewer, "review", _boom)
    text, audio, status, _reason = _run_guard(stub, "正常的话", b"orig")
    assert text != "正常的话"
    assert audio == b"safe-tts-audio"
    assert status == "unavailable"


def test_guard_tts_failure_keeps_text_only(register_services):
    """TTS 失败：兜底文字仍在，音频为空（不回退到原音频）。"""
    register_services(tts=_StubTTS(fail=True))
    text, audio, _status, _reason = _run_guard(_StubTTS(fail=True), "", b"orig")
    assert text and audio == b""

# ------------------- R17c：语音写回切 PreparedTurn 新契约 -------------------

@pytest.fixture()
def kv():
    from tools.storage import KVStoreTool

    return KVStoreTool()


def _stub_learner(monkeypatch):
    """打掉 post_commit normal 分支的重依赖（蒸馏/画像），聚焦写回契约本身。"""
    from orchestration import writeback as wb

    monkeypatch.setattr(wb, "ConversationDistiller", lambda: type(
        "D", (), {"distill_turn": lambda *a, **k: None})())


def test_voice_turn_normal_with_emotion_commits(kv, register_services, monkeypatch):
    """R17c：语音轮（emotion 已知）走 PreparedTurn——chat_log 落 mode=voice
    正式记录 + 关系增量；身份/分类与文字主链同一套。"""
    import interaction.api as ia
    from data.sqlite_store import get_db

    register_services(kv_store=kv)
    _stub_learner(monkeypatch)
    ia._commit_voice_turn(
        session_id="r17c-voice", user_text="今天好开心", reply_text="那太好了",
        emotion="happy", disposition="normal",
        source_review_status="accepted",
    )
    db = get_db()
    rows = db._conn.execute(
        "SELECT role, mode, disposition, source_review_status FROM chat_log "
        "WHERE session_id = 'r17c-voice' ORDER BY id"
    ).fetchall()
    assert [r[0] for r in rows] == ["user", "assistant"], "user/assistant 正式记录各一条"
    assert all(r[1] == "voice" for r in rows), "语音轮必须落 mode=voice"
    assert all(r[2] == "normal" and r[3] == "accepted" for r in rows)
    # emotion 已知：领取关系更新资格（relationship 有增量）
    rel = kv.read("relationship", "r17c-voice") or {}
    assert rel.get("interaction_count") == 1, "emotion 已知的 normal 轮该有关系增量"


def test_voice_turn_unknown_emotion_gets_no_relationship(kv, register_services, monkeypatch):
    """R17c 核心条款：缺 emotion 明确为未知，不伪造 neutral 领取关系更新资格。

    语音路由感知不到情绪时照样落正式历史（对话发生过），但 relationship
    一个格子都不许动——旧 _realtime_writeback 用 `emotion or "neutral"` 白领
    "平平常常聊了一会儿"的增量。
    """
    import interaction.api as ia
    from data.sqlite_store import get_db

    register_services(kv_store=kv)
    _stub_learner(monkeypatch)
    ia._commit_voice_turn(
        session_id="r17c-noemo", user_text="喂", reply_text="嗯",
        emotion="", disposition="normal",
        source_review_status="accepted",
    )
    rows = get_db()._conn.execute(
        "SELECT disposition, emotion FROM chat_log WHERE session_id = 'r17c-noemo'"
    ).fetchall()
    assert rows and all(r[0] == "normal" for r in rows)
    assert all((r[1] or "") == "" for r in rows), "不许把 neutral 写进聊天记录冒充已知"
    rel = kv.read("relationship", "r17c-noemo")
    assert not rel, "缺 emotion 的 normal 轮不许领取关系更新资格"


def test_voice_turn_rejected_is_degraded_no_learning(kv, register_services, monkeypatch):
    """R17c：审核拒绝的语音轮 = degraded——兜底正文落正式记录，零关系增量。"""
    import interaction.api as ia
    from data.sqlite_store import get_db

    register_services(kv_store=kv)
    _stub_learner(monkeypatch)
    ia._commit_voice_turn(
        session_id="r17c-rej", user_text="说点违规的", reply_text="这个话题我不能聊",
        emotion="", disposition="degraded",
        source_review_status="rejected", reason_code="voice_output_review_blocked",
    )
    rows = get_db()._conn.execute(
        "SELECT disposition, source_review_status, reason_code FROM chat_log "
        "WHERE session_id = 'r17c-rej'"
    ).fetchall()
    assert rows and all(r[0] == "degraded" for r in rows)
    assert all(r[1] == "rejected" and r[2] == "voice_output_review_blocked" for r in rows)
    assert not kv.read("relationship", "r17c-rej"), "degraded 轮零关系增量"

def test_voice_stream_route_is_registered():
    """回归闸：`@voice_router.websocket("/stream")` 装饰器曾被"替换装饰器
    紧贴的函数"吃掉两次（R16c 把审核门插进装饰器与端点之间、R17c 重写审核门
    时连装饰器一起换掉）——WS 握手秒断 1000/1008，集成测试才抓到。
    路由必须真的注册在 voice_router 上。"""
    from interaction.api import voice_router
    from starlette.routing import WebSocketRoute

    ws_paths = [r.path for r in voice_router.routes if isinstance(r, WebSocketRoute)]
    assert "/api/voice/stream" in ws_paths, f"voice WS 端点没注册，现有路由: {ws_paths}"
