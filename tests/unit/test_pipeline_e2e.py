"""B3 · pipeline 端到端测试一条：FakeLLM 驱动 handle 真跑一轮。

目标不是覆盖全部路径（那是批次后续的活），而是确保 `DialoguePipeline.handle()`
在 FakeLLM + 真 KV store（conftest 隔离）的最小子集上**不崩**，且
handle 链路走完后返回非空 reply（不管是来自 FakeLLM 还是 FallbackController）。

为什么这条重要：pipeline 是全项目最长的同步链路（perception → 记忆召回 →
组装 → 生成 → 流式 → 写回），任何子步骤偷偷抛了都被 astream 吞掉——这条
测试至少守住"handle 主路径能跑完"。

不在本条范围：语音/TTS/工具调用/主动开口/persona_config 多场景——
都留给后续批次。
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest
from shared.types import InputMessage


class FakeLLM:
    """最小可用的 LLM 替身：chat/astream_chat 都返回固定回复。

    R05b：astream_chat 可注入自定义脚本（stream_script），确定成功链路测试
    用它验证"脚本应答原样出现在正文与存储"；不传时保持旧行为（降级冒烟用）。
    """

    def __init__(self, stream_script: "list[str] | None" = None) -> None:
        self.chat_calls: list = []
        self.stream_calls: list = []
        self._stream_script = list(stream_script or ["FakeLLM 流式回复。"])

    def chat(self, messages, temperature=None, max_tokens=None) -> str:  # noqa: ARG002
        self.chat_calls.append(list(messages))
        return "FakeLLM 占位回复"

    def chat_with_tools(self, messages, tools_catalog):  # noqa: ARG002
        return {"content": "", "tool_calls": []}

    async def astream_chat(self, messages, **kwargs) -> AsyncIterator[str]:  # noqa: ARG002
        self.stream_calls.append(list(messages))
        for chunk in self._stream_script:
            yield chunk


class FakeTTS:
    def synthesize(self, text, **kwargs):  # noqa: ARG002
        return b"fake-mp3"


@pytest.fixture
def fake_services(register_services):
    """FakeLLM + FakeTTS 注册到全局 services。"""
    fake_llm = FakeLLM()
    fake_tts = FakeTTS()
    register_services(llm=fake_llm, tts=fake_tts)
    return fake_llm, fake_tts


def test_pipeline_handle_does_not_crash(tmp_path, monkeypatch, fake_services):
    """handle() 主路径能跑完 + 返回非空 reply。

    容忍 FallbackController 路径：FakeLLM 注册了但 pipeline 仍可能因为
    其他子系统（persona_config 缺失、memory 召回失败等）走 fallback。
    只要 handle 不崩 + reply 非空，就证明主链路串联通过——这正是 B3 想要的保证。
    """
    # 重置 Settings cache 让 DATA_DIR 改动生效
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from config.settings import get_settings, load_app_config

    get_settings.cache_clear()
    load_app_config.cache_clear()

    # 重置 KEEPER 防止跨用例串味
    from orchestration.pipeline import KEEPER, DialoguePipeline

    KEEPER.restore("b3-e2e-session")

    import asyncio

    pipeline = DialoguePipeline()
    msg = InputMessage(text="你好", session_id="b3-e2e-session")
    reply = asyncio.run(pipeline.handle(msg))

    assert reply is not None, "handle 必须返回 FinalReply"
    assert reply.text != "", f"reply.text 必须非空，实测 {reply.text!r}"

    # 降级冒烟到此为止（保留）。R05b：原来这里还有一条
    # `assert total_llm_calls >= 0`——恒真的死断言，什么都不证明，已删。
    # "FakeLLM 必须被真正调用、其应答原样出现在正文与存储"的确定成功链路
    # 由下一条测试负责：那条不允许兜底，reply 必须一字不差等于脚本应答。


def test_fake_llm_registers_and_handle_uses_services(monkeypatch, fake_services):
    """轻量版：只验 FakeLLM 被 services 注册后 services.get('llm') 能取回。"""
    from shared.singletons import services

    fake_llm, _ = fake_services
    assert services.get("llm") is fake_llm, "FakeLLM 必须注册到全局 services"


# ------------------- R05b：确定成功链路（与降级冒烟互补） -------------------

_SCRIPTED_REPLY = "轨道今晚从东南方升起来，记得抬头看看。"


def test_fake_llm_scripted_reply_reaches_reply_and_storage(register_services):
    """脚本化 FakeLLM 的应答必须**原样**出现在接受正文与存储里（R05b）。

    上一条是降级冒烟：兜底也算过。这条不许兜底——
    - reply.text 一字不差等于脚本应答（不含 error/refuse/兜底痕迹）；
    - KEEPER 短期记忆与 chat_log（session 表）各有一份同样的 assistant 正文；
    - 生成阶段（astream_chat）确实被调用恰好一次。
    会话故意沿用降级冒烟的 session id：R05a 的跨用例隔离一旦失效，
    上一条测试写进模块单例 KEEPER 的占位回复会串进本条的断言，立刻红。
    """
    import asyncio

    from orchestration.pipeline import KEEPER, DialoguePipeline
    from shared.singletons import services
    from tools.storage import KVStoreTool

    llm = FakeLLM(stream_script=[_SCRIPTED_REPLY])
    register_services(kv_store=KVStoreTool(), llm=llm, tts=FakeTTS())

    sid = "b3-e2e-session"  # 故意与降级冒烟同 ID：隔离失效在这条上现形
    reply = asyncio.run(
        DialoguePipeline().handle(
            InputMessage(text="今晚有流星雨吗", session_id=sid)
        )
    )

    assert reply.text == _SCRIPTED_REPLY, f"正文必须是脚本应答原样，实测 {reply.text!r}"

    assistant_ctx = [
        m["content"] for m in KEEPER.get_context(sid) if m.get("role") == "assistant"
    ]
    assert assistant_ctx == [_SCRIPTED_REPLY], (
        f"KEEPER 应恰好含本轮应答一份（多了就是跨用例串味），实测 {assistant_ctx!r}"
    )

    rows = services.get("kv_store").read("session", sid) or []
    assistant_rows = [r.get("text") for r in rows if r.get("role") == "assistant"]
    assert assistant_rows == [_SCRIPTED_REPLY], (
        f"chat_log 应恰好含本轮应答一份，实测 {assistant_rows!r}"
    )

    assert len(llm.stream_calls) == 1, (
        f"生成阶段应恰好调用一次 astream_chat，实测 {len(llm.stream_calls)}"
    )
