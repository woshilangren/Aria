"""R14a：正文契约载体测试（canonical / voice_text / 审核结果）。

契约（8.11.1 + R14a）：
- 模型原始串里的 <voice> 协议**只在入口解析一次**（_ReplyStreamer）；
- canonical（唯一正式正文）与 voice_text（语音派生）是两个载体，
  **两段不拼回正文**——handle/FinalReply/存储里都不该再见到 <voice>；
- 审核结果（accepted / rejected / unavailable）与有限原因码随载体走；
- draft / 工具结果不具备持久化资格（进存储的只能是 canonical）。
"""

from __future__ import annotations

import asyncio

import pytest

from shared.singletons import services
from shared.types import InputMessage
from tests.unit.test_pipeline_e2e import FakeLLM, FakeTTS
from tools.storage import KVStoreTool

_SENTENCE = "今天也在好好过日子。"
_VOICE = "今天啊，就这么过着呢，还行。"


def _register(register_services, llm):
    register_services(kv_store=KVStoreTool(), llm=llm, tts=FakeTTS())


def _run_handle(register_services, llm, text, sid):
    from orchestration.pipeline import DialoguePipeline

    _register(register_services, llm)
    return asyncio.run(
        DialoguePipeline().handle(InputMessage(text=text, session_id=sid))
    )


# ---------------- 入口解析一次：_ReplyStreamer 直接验证 ----------------

def test_streamer_parses_voice_protocol_once():
    """<voice> 在流式装配器里解析一次：句子归句子、语音归语音，标签零残留。"""
    from orchestration.pipeline import _ReplyStreamer

    s = _ReplyStreamer(voice_mode=False)
    frags = s.feed(f"{_SENTENCE}<voice>{_VOICE}</voice>") + s.finish()
    sentences = [f[1] for f in frags if f[0] == "sentence"]
    voices = [f[1] for f in frags if f[0] == "voice"]
    assert _SENTENCE in "".join(sentences)
    assert _VOICE in "".join(voices)
    for piece in sentences + voices:
        assert "<voice" not in piece and "</voice" not in piece


def test_streamer_unclosed_voice_treated_as_voice():
    """模型忘闭合 </voice>：攒下的当语音处理（既有取舍），不进正文。"""
    from orchestration.pipeline import _ReplyStreamer

    s = _ReplyStreamer(voice_mode=False)
    frags = s.feed(f"{_SENTENCE}<voice>{_VOICE}") + s.finish()
    sentences = "".join(f[1] for f in frags if f[0] == "sentence")
    voices = "".join(f[1] for f in frags if f[0] == "voice")
    assert _SENTENCE in sentences
    assert _VOICE in voices
    assert "<voice" not in sentences + voices


# ---------------- handle 级：载体不再两段拼接 ----------------

def test_finalreply_carries_canonical_and_voice_separately(register_services):
    """handle 的 FinalReply：text=canonical（无标签）、voice_text 独立承载。"""
    llm = FakeLLM(stream_script=[f"{_SENTENCE}<voice>{_VOICE}</voice>"])
    reply = _run_handle(register_services, llm, "今天过得怎么样", "r14a-split")

    assert reply.text == _SENTENCE, f"正文必须是 canonical，实测 {reply.text!r}"
    assert reply.voice_text == _VOICE, f"语音派生必须独立承载，实测 {reply.voice_text!r}"
    assert "<voice" not in reply.text + reply.voice_text
    assert reply.review_status == "accepted"
    # 存储侧（draft/工具结果无持久化资格 → 只落 canonical）
    rows = services.get("kv_store").read("session", "r14a-split") or []
    assistant_rows = [r.get("text", "") for r in rows if r.get("role") == "assistant"]
    assert assistant_rows == [_SENTENCE], f"存储只能有 canonical，实测 {assistant_rows!r}"


def test_tool_reply_with_voice_protocol_also_split(register_services, monkeypatch):
    """工具链路同样遵守：转述稿里带 <voice> → 正文/语音分离入载体与存储。"""
    from capability.toolcall import ToolCallOrchestrator
    from orchestration.pipeline import DialoguePipeline
    from shared.types import ToolCallResult

    def fake_run(self, user_text, intent, fallback_city="", session_id="", should_cancel=None):
        return {"results": [ToolCallResult(tool_name="search", status="ok", data="x")],
                "final_text": ""}

    monkeypatch.setattr(ToolCallOrchestrator, "run", fake_run)
    llm = FakeLLM(chat_script=f"查到了。<voice>帮你查到啦。</voice>")
    _register(register_services, llm)
    sid = "r14a-tool"

    reply = asyncio.run(
        DialoguePipeline().handle(InputMessage(text="搜索一下附近的猫咖", session_id=sid))
    )
    assert reply.text == "查到了。"
    assert reply.voice_text == "帮你查到啦。"
    rows = services.get("kv_store").read("session", sid) or []
    assert not any("<voice" in str(r.get("text", "")) for r in rows)


def test_refuse_carries_rejected_review_status(register_services):
    """审核拒绝：FinalReply.review_status=rejected + 有限原因码，正文是安全兜底。"""
    llm = FakeLLM(stream_scripts=[
        ["作为一个AI不该说QW7X。"],
        ["重写后还是带QW7X，作为一个AI。"],
        ["第三次仍有QW7X，作为一个AI。"],
    ])
    reply = _run_handle(register_services, llm, "随便聊聊天", "r14a-refuse")

    assert reply.review_status == "rejected"
    assert reply.reason_code == "review_output_blocked"
    assert "QW7X" not in reply.text


# ------------------- R14b：发布内容汇总 -------------------

def test_refuse_after_published_prefix_composes_degraded_canonical(register_services):
    """已发布合格前缀后再拒绝：存储 = 前缀 + 兜底（R14b 核心），不是完整原稿
    也不是空。前端仍收到整轮替换的兜底文案（既有语义不变）。"""
    from orchestration.pipeline import KEEPER, DialoguePipeline

    llm = FakeLLM(stream_scripts=[
        # 同一次生成内：第一句已发布（推给前端），第二句被审核拒绝 → 已推过
        # 内容 → 整轮降级（_MAX_REWRITE 的重写只在"没推过"时发生）
        ["这句先说给你听了。后半句冒出作为一个AI的QW7X。"],
    ])
    _register(register_services, llm)
    sid = "r14b-degraded"

    reply = asyncio.run(
        DialoguePipeline().handle(InputMessage(text="随便聊聊天", session_id=sid))
    )
    # 前端：整轮替换成安全兜底
    assert reply.review_status == "rejected"
    assert "QW7X" not in reply.text
    # 存储：降级 canonical = 已发布前缀 + 兜底（不是完整原稿、不是空）
    expected_prefix = "这句先说给你听了。"
    rows = services.get("kv_store").read("session", sid) or []
    assistant_rows = [r.get("text", "") for r in rows if r.get("role") == "assistant"]
    assert assistant_rows, "降级轮必须落库（前缀+兜底）"
    stored = assistant_rows[-1]
    assert stored.startswith(expected_prefix), f"降级正文必须以已发布前缀开头，实测 {stored!r}"
    assert "QW7X" not in stored and "作为一个AI" not in stored
    assert stored != expected_prefix, "前缀之后应接兜底话"
    # KEEPER 同步
    ctx = [m["content"] for m in KEEPER.get_context(sid) if m.get("role") == "assistant"]
    assert expected_prefix in ctx[-1]


def test_refuse_without_published_prefix_stores_nothing(register_services):
    """没推出任何内容就拒绝：维持 refuse-only（什么都没说过），不造假正文。"""
    from orchestration.pipeline import KEEPER, DialoguePipeline
    from shared.singletons import services

    llm = FakeLLM(stream_scripts=[
        ["作为一个AI不该说QW7X。"],
        ["重写后还是带QW7X，作为一个AI。"],
        ["第三次仍有QW7X，作为一个AI。"],
    ])
    _register(register_services, llm)
    sid = "r14b-noprefix"

    reply = asyncio.run(
        DialoguePipeline().handle(InputMessage(text="随便聊聊天", session_id=sid))
    )
    assert reply.review_status == "rejected"
    rows = services.get("kv_store").read("session", sid) or []
    assert not any(r.get("role") == "assistant" for r in rows), "零发布前缀的拒绝轮不许有正文"
    ctx = [m for m in KEEPER.get_context(sid) if m.get("role") == "assistant"]
    assert ctx == []


def test_gateway_tts_source_is_canonical(register_services, monkeypatch):
    """R14b：voice_text 同义性不可核验 → TTS 源 = canonical 正文，不是语音版。"""
    from interaction.gateway import OutputRenderer

    spoken_inputs = []

    class _RecordingTTS:
        def synthesize(self, text, **kwargs):
            spoken_inputs.append(text)
            return b"fake-audio"

    monkeypatch.setitem(services._services, "tts", _RecordingTTS())
    from shared.types import FinalReply

    renderer = OutputRenderer()
    data = renderer.render(
        FinalReply(text="正文内容。", voice_text="口语版内容。"), "r14b-tts"
    )
    assert spoken_inputs == ["正文内容。"], f"TTS 源必须是 canonical，实测 {spoken_inputs}"
    assert data["voice_audio"]
    assert data["voice_text"] == "正文内容。", "语音条标注必须与实际合成内容一致"


def test_finalreply_delivery_defaults_unknown():
    """发布确认（8.11.1 规则 6）：handle 攒出的 FinalReply 不算发布——
    delivery 默认 unknown，只有交互适配器才能置 sent。"""
    from shared.types import FinalReply

    assert FinalReply(text="x").delivery == "unknown"
