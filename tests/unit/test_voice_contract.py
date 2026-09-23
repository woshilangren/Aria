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
