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

    def __init__(self, stream_script: "list[str] | None" = None,
                 stream_scripts: "list[list[str]] | None" = None,
                 chat_script: str = "FakeLLM 占位回复") -> None:
        self.chat_calls: list = []
        self.stream_calls: list = []
        # R27a：stream_scripts 支持多次调用各用一段脚本（审核重写每次是一次
        # 新的 astream_chat 调用）；只传 stream_script 时退化为单脚本旧行为。
        if stream_scripts is not None:
            self._stream_queue = [list(s) for s in stream_scripts]
        elif stream_script is not None:
            self._stream_queue = [list(stream_script)]
        else:
            self._stream_queue = [["FakeLLM 流式回复。"]]
        # R02：chat 固定应答——工具链的转述调用（persona_wrap）也脚本化，
        # 返回值就是 _respond 写进 final_reply 的内容
        self._chat_script = chat_script

    def chat(self, messages, temperature=None, max_tokens=None) -> str:  # noqa: ARG002
        self.chat_calls.append(list(messages))
        return self._chat_script

    def chat_with_tools(self, messages, tools_catalog):  # noqa: ARG002
        return {"content": "", "tool_calls": []}

    async def astream_chat(self, messages, **kwargs) -> AsyncIterator[str]:  # noqa: ARG002
        self.stream_calls.append(list(messages))
        if not self._stream_queue:
            return
        for chunk in self._stream_queue.pop(0):
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


def test_pipeline_handle_does_not_crash(tmp_path, monkeypatch, register_services):
    """降级冒烟（B3，R27b 语义更新）：主备 LLM 全挂（窄载体）→ handle 萰兜底话。

    以前这条"容忍任何原因的非空 reply"——R27b 之后不允许那么含糊：
    - 外部服务失败（ExternalServiceError）→ 允许的降级，handle 返回 llm 兜底话；
    - 本地缺陷（内部错误）→ handle 抛 PipelineInternalError（见
      test_failure_contract.py），**不**算降级成功。
    这条钉住前一半：LLM 抛窄载体，handle 仍返回非空兜底、不崩、不冒泡。
    """
    # 重置 Settings cache 让 DATA_DIR 改动生效
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from config.settings import get_settings, load_app_config

    get_settings.cache_clear()
    load_app_config.cache_clear()

    # 重置 KEEPER 防止跨用例串味
    from orchestration.pipeline import KEEPER, DialoguePipeline
    from shared.singletons import services
    from shared.types import ExternalServiceError
    from tools.storage import KVStoreTool

    KEEPER.restore("b3-e2e-session")

    class _AllProvidersDownLLM:
        """主备全挂（窄载体）：astream/chat 全部抛 ExternalServiceError。"""

        async def astream_chat(self, messages, **kwargs):  # noqa: ARG002
            raise ExternalServiceError("llm", "all_providers_failed")
            yield  # pragma: no cover —— 仅为成为 async 生成器

        def chat(self, messages, temperature=None, max_tokens=None):  # noqa: ARG002
            raise ExternalServiceError("llm", "all_providers_failed")

        def chat_with_tools(self, messages, tools_catalog):  # noqa: ARG002
            raise ExternalServiceError("llm", "all_providers_failed")

    register_services(kv_store=KVStoreTool(), llm=_AllProvidersDownLLM(),
                      tts=FakeTTS())

    import asyncio

    pipeline = DialoguePipeline()
    msg = InputMessage(text="你好", session_id="b3-e2e-session")
    reply = asyncio.run(pipeline.handle(msg))

    assert reply is not None, "handle 必须返回 FinalReply"
    assert reply.text != "", f"降级后 reply 必须非空（llm 兜底话），实测 {reply.text!r}"

    # 降级冒烟到此为止（保留）。R05b：原来这里还有一条
    # `assert total_llm_calls >= 0`——恒真的死断言，什么都不证明，已删。
    # "FakeLLM 必须被真正调用、其应答原样出现在正文与存储"的确定成功链路
    # 由下一条测试负责：那条不允许兜底，reply 必须一字不差等于脚本应答。


def test_all_providers_down_commits_degraded_no_learning(register_services):
    """R17b 补漏（真机记录04 实锤）：外部失败的本地模板兜底轮必须按 **degraded**
    提交——历史落正式记录但零关系增量、零账本；不许冒充 normal 领更新资格。

    修复前：_astream_chat 的 ExternalServiceError 处理器用本地 generate() 顶上
    时没打降级标记，兜底轮按 normal 提交且 intimacy +1（真机 storage 复现）。
    """
    import asyncio
    import sqlite3

    from data.sqlite_store import get_db
    from orchestration.pipeline import DialoguePipeline
    from shared.types import ExternalServiceError
    from tools.storage import KVStoreTool

    class _AllProvidersDownLLM:
        async def astream_chat(self, messages, **kwargs):  # noqa: ARG002
            raise ExternalServiceError("llm", "all_providers_failed")
            yield  # pragma: no cover

        def chat(self, messages, temperature=None, max_tokens=None):  # noqa: ARG002
            raise ExternalServiceError("llm", "all_providers_failed")

        def chat_with_tools(self, messages, tools_catalog):  # noqa: ARG002
            raise ExternalServiceError("llm", "all_providers_failed")

    register_services(kv_store=KVStoreTool(), llm=_AllProvidersDownLLM())
    sid = "b3-degraded-contract"
    reply = asyncio.run(
        DialoguePipeline().handle(InputMessage(text="你好", session_id=sid))
    )
    assert reply.text != "", "降级轮必须有人设兜底正文"

    db = get_db()
    rows = db._conn.execute(
        "SELECT role, disposition, source_review_status, reason_code FROM chat_log "
        "WHERE session_id = ?", (sid,)
    ).fetchall()
    assert rows, "降级轮也要落正式历史（对话发生过该留痕）"
    assert all(r[1] == "degraded" for r in rows), f"必须是 degraded，实测 {rows}"
    assert all(r[2] == "unavailable" for r in rows)
    assert all(r[3] == "all_providers_failed" for r in rows)
    rel = db._conn.execute(
        "SELECT value FROM relationship WHERE session_id = ?", (sid,)
    ).fetchone()
    assert rel is None, "降级轮零关系增量（不许预创建/预增）"
    ledger = db._conn.execute(
        "SELECT * FROM affection_history WHERE session_id = ?", (sid,)
    ).fetchall()
    assert ledger == [], "降级轮零账本"


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


# ------------------- R02：工具拒绝原稿不许写回 -------------------

_TOOL_MARK = "独特工具标记QW7X"
# 含 "作为一个AI"（_META_PATTERNS）→ output 审核必拒，不依赖 mock 审核器
_REJECTED_REPLY = f"{_TOOL_MARK}，作为一个AI我只说真话。"
_NORMAL_REPLY = "巷口那家面馆晚上人多，去早点。"


def _register_tool_pipeline(register_services, monkeypatch, chat_reply: str) -> dict:
    """搭一条工具链路：search 意图 + 替身工具结果 + FakeLLM 转述。

    返回 {"calls": 计数器}——已执行的副作用工具不许重跑（R02 验收项）。
    """
    from capability.toolcall import ToolCallOrchestrator
    from shared.types import ToolCallResult
    from tools.storage import KVStoreTool

    calls = {"n": 0}

    def fake_run(self, user_text, intent, fallback_city="", session_id="", should_cancel=None):
        calls["n"] += 1
        return {
            "results": [ToolCallResult(tool_name="search", status="ok",
                                       data=f"search结果：{_TOOL_MARK}")],
            "final_text": "",
        }

    monkeypatch.setattr(ToolCallOrchestrator, "run", fake_run)
    register_services(kv_store=KVStoreTool(), llm=FakeLLM(chat_script=chat_reply),
                      tts=FakeTTS())
    return calls


def test_tool_review_reject_draft_never_persisted(register_services, monkeypatch):
    """工具审核拒绝：被拒原稿不许混进 KEEPER / chat_log / 画像输入（R02）。

    缺陷机理：_respond 在审核**之前**就把转述稿塞进 state.final_reply，而
    _writeback 靠"final_reply 非空"决定可保存——审核一拒，_RefuseTurn 走到
    写回时 final_reply 还非空，被拒原稿就这么落进 KEEPER 和 chat_log。
    修复后本用例必须绿：前端收到安全兜底、被拒标记零残留、副作用工具不重跑。
    """
    import asyncio

    from orchestration.pipeline import KEEPER, DialoguePipeline
    from shared.singletons import services
    from shared.types import InputMessage

    sid = "r02-tool-reject"
    calls = _register_tool_pipeline(register_services, monkeypatch, _REJECTED_REPLY)

    reply = asyncio.run(
        DialoguePipeline().handle(InputMessage(text="搜索一下附近好吃的", session_id=sid))
    )

    # 前端收到的是安全兜底（refuse 文案），被拒标记一个字都不许出现
    assert reply.text and _TOOL_MARK not in reply.text, f"reply 泄漏被拒原稿: {reply.text!r}"
    assert "作为一个AI" not in reply.text
    # KEEPER：被拒标记零残留（refuse-only 分支连 assistant 正文都不该有）
    ctx = KEEPER.get_context(sid)
    assert not any(_TOOL_MARK in m.get("content", "") for m in ctx), f"KEEPER 混入被拒原稿: {ctx!r}"
    # chat_log：被拒标记零残留，且不产生 assistant 行
    rows = services.get("kv_store").read("session", sid) or []
    assert not any(_TOOL_MARK in str(r.get("text", "")) for r in rows), f"chat_log 混入被拒原稿: {rows!r}"
    assert not any(r.get("role") == "assistant" for r in rows)
    # 已执行的副作用工具不重跑（审核拒绝不触发任何重新执行）
    assert calls["n"] == 1, f"工具应恰好执行一次，实测 {calls['n']}"


def test_tool_normal_reply_still_persisted(register_services, monkeypatch):
    """对照组：正常工具回复仍然原样入正文与存储（R02 不许误伤正常路径）。"""
    import asyncio

    from orchestration.pipeline import KEEPER, DialoguePipeline
    from shared.singletons import services
    from shared.types import InputMessage

    sid = "r02-tool-normal"
    _register_tool_pipeline(register_services, monkeypatch, _NORMAL_REPLY)

    reply = asyncio.run(
        DialoguePipeline().handle(InputMessage(text="搜索一下附近好吃的", session_id=sid))
    )

    assert reply.text == _NORMAL_REPLY, f"正文必须是转述稿原样，实测 {reply.text!r}"
    ctx = [m["content"] for m in KEEPER.get_context(sid) if m.get("role") == "assistant"]
    assert ctx == [_NORMAL_REPLY], f"KEEPER 应有正常回复，实测 {ctx!r}"
    rows = services.get("kv_store").read("session", sid) or []
    assistant_rows = [r.get("text") for r in rows if r.get("role") == "assistant"]
    assert assistant_rows == [_NORMAL_REPLY], f"chat_log 应有正常回复，实测 {assistant_rows!r}"


# ------------------- R27a：审核原文不外泄（reason 输出边界） -------------------

def _collect_stream(pipeline, msg) -> list:
    """直接消费 astream 事件流——SSE 的完整载荷就是这些事件的 JSON 序列，
    检查它们而不只看最终 UI（R27a 验收要求）。"""
    import asyncio

    async def _run():
        evs = []
        async for ev in pipeline.astream(msg):
            evs.append(ev)
        return evs

    return asyncio.run(_run())


def _assert_no_leak(evs, *secrets):
    """哨兵不得出现在任何对外字段的任何事件里（含 reason/message/debug）。"""
    import json

    blob = json.dumps(evs, ensure_ascii=False, default=str)
    for s in secrets:
        assert s not in blob, f"对外事件载荷泄漏 {s!r}：{blob[:400]}"


def test_review_reject_reason_code_text_path(register_services):
    """文字拒绝 + 重写耗尽：refuse.reason 只带原因码，被拦原文一个字不外泄。"""
    from orchestration.pipeline import DialoguePipeline
    from shared.types import InputMessage
    from tools.storage import KVStoreTool

    register_services(kv_store=KVStoreTool(), llm=FakeLLM(stream_scripts=[
        ["作为一个AI不该说QW7X。"],
        ["重写后还是带QW7X，作为一个AI。"],
        ["第三次仍有QW7X，作为一个AI口吻。"],
    ]), tts=FakeTTS())

    evs = _collect_stream(
        DialoguePipeline(),
        InputMessage(text="随便聊聊天", session_id="r27a-text"),
    )
    _assert_no_leak(evs, "QW7X", "作为一个AI")
    refuse = [e for e in evs if e.get("type") == "refuse"]
    assert refuse, f"重写耗尽必须整轮降级，实测事件 {[(e.get('type')) for e in evs]}"
    assert refuse[0]["reason"] == "review_output_blocked", (
        f"reason 必须是有限原因码，实测 {refuse[0]['reason']!r}"
    )
    assert "QW7X" not in refuse[0].get("text", "")


def test_review_reject_reason_code_voice_path(register_services):
    """voice 拒绝：<voice> 被拦原文同样只出原因码。"""
    from orchestration.pipeline import DialoguePipeline
    from shared.types import InputMessage
    from tools.storage import KVStoreTool

    register_services(kv_store=KVStoreTool(), llm=FakeLLM(stream_scripts=[
        ["<voice>语音带QW7X，作为一个AI。</voice>"],
        ["<voice>重说仍有QW7X，作为一个AI。</voice>"],
        ["<voice>第三次QW7X，作为一个AI。</voice>"],
    ]), tts=FakeTTS())

    evs = _collect_stream(
        DialoguePipeline(),
        InputMessage(text="在吗", session_id="r27a-voice", input_mode="voice"),
    )
    _assert_no_leak(evs, "QW7X", "作为一个AI")
    refuse = [e for e in evs if e.get("type") == "refuse"]
    assert refuse and refuse[0]["reason"] == "review_output_blocked"


def test_reject_after_pushed_prefix_keeps_prefix_hides_rest(register_services):
    """已发布合格前缀后拒绝：前缀保留（合法已发布内容不在禁传范围），
    未发布的被拒片段不外泄。"""
    from orchestration.pipeline import DialoguePipeline
    from shared.types import InputMessage
    from tools.storage import KVStoreTool

    register_services(kv_store=KVStoreTool(), llm=FakeLLM(stream_scripts=[
        ["这句完全没问题，先说着。第二句冒出作为一个AI的QW7X。"],
        ["换说法还是有作为一个AI的QW7X。"],
        ["依旧有QW7X，作为一个AI。"],
    ]), tts=FakeTTS())

    evs = _collect_stream(
        DialoguePipeline(),
        InputMessage(text="随便聊聊天", session_id="r27a-prefix"),
    )
    # 哨兵/被拦片段零外泄
    _assert_no_leak(evs, "QW7X", "作为一个AI")
    # 合法已发布前缀仍推给了前端（不能把"说出去的话"也藏了）
    assert any(
        e.get("type") == "sentence" and "这句完全没问题" in e.get("text", "")
        for e in evs
    ), f"已发布前缀不该被隐藏，实测 {[(e.get('type'), e.get('text', '')) for e in evs]}"
    refuse = [e for e in evs if e.get("type") == "refuse"]
    assert refuse and refuse[0]["reason"] == "review_output_blocked"


def test_tool_reject_refuse_reason_no_draft(register_services, monkeypatch):
    """工具拒绝：refuse.reason 同样不含被拒工具原稿（R27a 与 R02 同一条链）。"""
    from orchestration.pipeline import DialoguePipeline
    from shared.types import InputMessage

    calls = _register_tool_pipeline(register_services, monkeypatch, _REJECTED_REPLY)
    evs = _collect_stream(
        DialoguePipeline(),
        InputMessage(text="搜索一下附近好吃的", session_id="r27a-tool"),
    )
    _assert_no_leak(evs, _TOOL_MARK, "作为一个AI")
    refuse = [e for e in evs if e.get("type") == "refuse"]
    assert refuse and refuse[0].get("reason", "") == ""
    assert calls["n"] == 1


def test_pipeline_error_event_whitelisted(register_services, monkeypatch):
    """外部服务失败：error 事件只给固定安全文案 + 有限错误码，不回 str(exc)。
    原因码区分：审核拒绝=refuse.review_output_blocked，外部失败=external_degraded，
    程序错误=internal_error（详见 test_failure_contract.py）。"""
    from orchestration.pipeline import DialoguePipeline
    from shared.types import ExternalServiceError, InputMessage
    from tools.storage import KVStoreTool

    register_services(kv_store=KVStoreTool(), llm=FakeLLM(stream_script=["正常一句话。"]),
                      tts=FakeTTS())

    def boom(self, state):  # noqa: ARG001
        raise ExternalServiceError("llm", "timeout", detail="内部细节SECRET-ERR")

    monkeypatch.setattr(DialoguePipeline, "_compose", boom)
    evs = _collect_stream(
        DialoguePipeline(),
        InputMessage(text="随便聊聊", session_id="r27a-err"),
    )
    _assert_no_leak(evs, "SECRET-ERR")
    errs = [e for e in evs if e.get("type") == "error"]
    assert errs, f"外部失败必须出 error 事件，实测 {[(e.get('type')) for e in evs]}"
    assert errs[0]["code"] == "external_degraded"
    assert errs[0]["message"] == "这轮没接上，稍后再试试"
