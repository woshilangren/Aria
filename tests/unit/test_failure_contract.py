"""R27b：外部故障与程序错误分流的失败契约测试。

契约（《计划与设计.md》R27b / 8.8 分层纪律）：
- openai SDK 的连接/超时/API 响应异常在 llm_client 最窄边界转成
  ExternalServiceError（来源/原因码/可重试），并计入主备熔断计数；
- TypeError/NameError 等本地装配缺陷原样上抛：**不**转窄载体、**不**计入
  熔断（程序错误不是供应商故障）、**不**烧备用供应商；
- 管道里本地缺陷冒泡到 handle() → PipelineInternalError：零重跑、零业务提交；
- 外部失败走允许的降级（error 事件 external_degraded → handle 落 llm 兜底话）；
- 原因码可区分：external_degraded / internal_error / review_output_blocked。

不触网：openai 异常用可构造的基类模拟，LLMClient 用 __new__ 绕过构造。
"""

from __future__ import annotations

import asyncio
import threading
import types

import openai
import pytest

from tests.unit.test_pipeline_e2e import FakeLLM, FakeTTS  # noqa: F401  (FakeTTS 供 _register)
from shared.types import ExternalServiceError, PipelineInternalError


# --------------------------------------------------------------------------
# 1. llm_client 同步/流式边界：SDK 异常转窄载体并计数；本地缺陷原样上抛不计数
# --------------------------------------------------------------------------

class _Completions:
    """替身 completions：create() 按脚本抛异常或返回固定响应，并计数。"""

    def __init__(self, exc, counter: dict, key: str, chunks=None):
        self._exc = exc
        self._counter = counter
        self._key = key
        self._chunks = chunks

    async def acreate(self, **kwargs):
        return await asyncio.sleep(0) or self._create(**kwargs)

    def create(self, **kwargs):
        self._counter[self._key] += 1
        if self._exc is not None:
            raise self._exc
        if self._chunks is not None:
            return self._agen()
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="正常回复"))])

    async def _agen(self):
        for c in self._chunks:
            if isinstance(c, Exception):
                raise c
            yield c


def _wire(client_obj, attr, exc=None, chunks=None, counter=None, key="main"):
    setattr(client_obj, attr, types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions(exc, counter, key, chunks))))


def _make_client(main_exc=None, fallback_exc=None):
    """绕过 __init__（要真 key）手工装配 LLMClient 的熔断/客户端内部状态。"""
    from tools.llm_client import LLMClient

    calls = {"main": 0, "fallback": 0}
    c = LLMClient.__new__(LLMClient)
    c._model = "test-model"
    _wire(c, "_client", main_exc, counter=calls, key="main")
    _wire(c, "_fallback", fallback_exc, counter=calls, key="fallback")
    c._afallback = None
    c._fallback_family = "claude"
    c._fallback_model = "test-fallback-model"
    c._family = "claude"
    c._cb_lock = threading.Lock()
    c._opened_at = None
    c._cooldown = 120.0
    c._error_count = 0
    c._max_errors = 2
    c._probe_in_flight = False
    c._probe_fail_streak = 0
    return c, calls


def test_sync_sdk_error_converts_to_external_and_counts():
    """SDK 异常（超时/连接类）→ 主失败计数一次；备用也挂 → ExternalServiceError。"""
    c, calls = _make_client(main_exc=openai.OpenAIError("模拟超时/连接失败"),
                            fallback_exc=openai.OpenAIError("备用也挂"))
    with pytest.raises(ExternalServiceError) as ei:
        c._complete(model="test-model", messages=[{"role": "user", "content": "hi"}])
    assert ei.value.source == "llm"
    assert ei.value.reason_code == "all_providers_failed"
    assert ei.value.retryable is True
    assert c._error_count == 1                    # 供应商失败必须计数（熔断证据）
    assert calls["fallback"] == 1                 # 备用被真实尝试


def test_sync_sdk_error_with_healthy_fallback_succeeds():
    """主挂、备用健康：正常返回，不抛（熔断计数后衰减一格，属 R26 语义）。"""
    c, calls = _make_client(main_exc=openai.OpenAIError("主挂"))
    resp = c._complete(model="test-model", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "正常回复"
    assert c._error_count == 0  # 主失败 +1 后被备用成功衰减回 0（J13 取舍保留）
    assert calls["fallback"] == 1


def test_sync_local_defect_propagates_and_not_counted():
    """TypeError（本地装配缺陷）不转窄载体、不计熔断、不烧备用——原样上抛。"""
    c, calls = _make_client(main_exc=TypeError("本地装配缺陷"))
    with pytest.raises(TypeError, match="本地装配缺陷"):
        c._complete(model="test-model", messages=[{"role": "user", "content": "hi"}])
    assert c._error_count == 0
    assert calls["fallback"] == 0


def test_stream_build_fail_both_providers_raises_external():
    """流式建流失败 + 备用也挂 → ExternalServiceError（不再是裸 RuntimeError）。"""
    from tools.llm_client import LLMClient

    c, _calls = _make_client(main_exc=openai.OpenAIError("主挂"),
                             fallback_exc=openai.OpenAIError("备挂"))
    counter = {"a": 0}

    async def _acreate(**kwargs):
        counter["a"] += 1
        raise openai.OpenAIError("主挂")

    c._aclient = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_acreate)))

    afails = {"n": 0}

    async def _afcreate(**kwargs):
        afails["n"] += 1
        raise openai.OpenAIError("备挂")

    c._afallback = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_afcreate)))

    async def _run():
        return [s async for s in c.astream_chat([{"role": "user", "content": "hi"}])]

    with pytest.raises(ExternalServiceError) as ei:
        asyncio.run(_run())
    assert ei.value.reason_code == "all_providers_failed"
    assert c._error_count == 1
    assert afails["n"] == 1


def test_stream_interrupted_after_tokens_raises_external():
    """已吐 token 后流中断：转 stream_interrupted（retryable=False），
    管道据此按外部降级收尾；主模型失败计一次。"""
    from tools.llm_client import LLMClient

    c, _calls = _make_client(main_exc=None)

    async def _agen():
        # chunk 必须是 SDK 形状（带 choices.delta.content）：否则 _delta_of 取不到
        # 文本，emitted 一直 False，会被当成"没吐过 token"走备用而不是测中断转换
        yield types.SimpleNamespace(choices=[types.SimpleNamespace(
            delta=types.SimpleNamespace(content="正常半句"))])
        raise openai.OpenAIError("流中断")

    async def _acreate(**kwargs):
        return _agen()

    c._aclient = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_acreate)))
    c._afallback = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_acreate)))  # 备用不该被走到

    out = []

    async def _run():
        with pytest.raises(ExternalServiceError) as ei:
            async for s in c.astream_chat([{"role": "user", "content": "hi"}]):
                out.append(s)
        return ei

    ei = asyncio.run(_run())
    assert out == ["正常半句"]
    assert ei.value.reason_code == "stream_interrupted"
    assert ei.value.retryable is False
    assert c._error_count == 1


# --------------------------------------------------------------------------
# 2. 管道级：本地缺陷 → internal_error 冒泡，零重跑零提交；外部失败 → 允许降级
# --------------------------------------------------------------------------

def _register(register_services, llm):
    from tools.storage import KVStoreTool

    register_services(kv_store=KVStoreTool(), llm=llm, tts=FakeTTS())


def test_internal_error_bubbles_to_handle_with_zero_commit(register_services):
    """生成中途注入 NameError（本地缺陷）：astream 出 internal_error 事件，
    handle 抛 PipelineInternalError；零 LLM 重试、零业务提交。"""
    from orchestration.pipeline import KEEPER, DialoguePipeline
    from shared.singletons import services
    from shared.types import InputMessage

    class _LocalDefectLLM(FakeLLM):
        """吐出一段后抛本地缺陷（NameError），模拟生成路径上的编程错误。"""

        async def astream_chat(self, messages, **kwargs):  # noqa: ARG002
            self.stream_calls.append(list(messages))
            yield "正常半句。"
            raise NameError("生成中途的本地缺陷")

    llm = _LocalDefectLLM()
    _register(register_services, llm)
    sid = "r27b-internal"

    # astream 层：error 事件带 internal_error，且没有任何兜底正文被推出
    evs = _collect_stream(DialoguePipeline(), InputMessage(text="随便聊聊", session_id=sid))
    errs = [e for e in evs if e.get("type") == "error"]
    assert errs and errs[0]["code"] == "internal_error"
    assert all(e.get("type") != "sentence" for e in evs), "内部错误不许有兜底正文"

    # 零重试：collect 那轮生成恰好一次（internal_error 不触发任何重新生成）
    assert len(llm.stream_calls) == 1

    # handle 层：internal_error 必须冒泡，不能被包成成功的 FinalReply；
    # 再完整跑一轮也只 +1 次生成（每轮各一次，绝无轮内重试/备用重发）
    with pytest.raises(PipelineInternalError):
        asyncio.run(
            DialoguePipeline().handle(InputMessage(text="随便聊聊", session_id=sid))
        )
    assert len(llm.stream_calls) == 2

    # 零业务提交：KEEPER 无本轮内容、chat_log 无任何行
    assert not any("随便聊聊" in m.get("content", "") for m in KEEPER.get_context(sid))
    rows = services.get("kv_store").read("session", sid) or []
    assert rows == []


def _collect_stream(pipeline, msg) -> list:
    """消费 astream 事件流（与 test_pipeline_e2e 同款，这里独立成小函数避免跨文件耦合）。"""
    async def _run():
        evs = []
        async for ev in pipeline.astream(msg):
            evs.append(ev)
        return evs

    return asyncio.run(_run())


def test_external_error_degrades_via_handle(register_services):
    """外部失败（窄载体）→ astream 出 external_degraded → handle 落 llm 兜底话
    （允许的降级，聊天不断），不抛 PipelineInternalError。"""
    from orchestration.pipeline import DialoguePipeline
    from shared.types import InputMessage

    class _ExternalFailLLM:
        """astream 与 chat 都抛窄载体（模拟主备全挂）。

        astream_chat 必须是 async **生成器**（管道用 async for 消费）：
        `raise` 前放一个不可达 yield，否则它是普通协程，async for 直接
        TypeError，会被分流成 internal_error 而不是被测的外部降级路径。
        """

        def __init__(self):
            self.stream_calls = []
            self.chat_calls = []

        def chat(self, messages, temperature=None, max_tokens=None):  # noqa: ARG002
            self.chat_calls.append(messages)
            raise ExternalServiceError("llm", "all_providers_failed")

        def chat_with_tools(self, messages, tools_catalog):  # noqa: ARG002
            raise ExternalServiceError("llm", "all_providers_failed")

        async def astream_chat(self, messages, **kwargs):  # noqa: ARG002
            self.stream_calls.append(list(messages))
            raise ExternalServiceError("llm", "all_providers_failed")
            yield  # pragma: no cover —— 仅为把它变成 async 生成器

    llm = _ExternalFailLLM()
    _register(register_services, llm)
    sid = "r27b-external"

    evs = _collect_stream(DialoguePipeline(), InputMessage(text="随便聊聊", session_id=sid))
    # 生成失败 → 管道内降级（generate 的兜底话）成功消化：无 error 事件、
    # 兜底正文正常推出；绝无 internal_error（那是程序错误的专属信号）
    assert not any(e.get("type") == "error" and e.get("code") == "internal_error" for e in evs)
    assert any(e.get("type") == "done" for e in evs), f"外部失败必须降级收尾，实测 {[e.get('type') for e in evs]}"
    _assert_no_leak_local(evs, "all_providers_failed")

    reply = asyncio.run(
        DialoguePipeline().handle(InputMessage(text="随便聊聊", session_id=sid))
    )
    # 允许的降级：返回兜底话（非空），不抛 PipelineInternalError
    assert reply.text, "外部失败必须降级成兜底话而不是空回复"


def _assert_no_leak_local(evs, *secrets):
    import json

    blob = json.dumps(evs, ensure_ascii=False, default=str)
    for s in secrets:
        assert s not in blob, f"对外事件载荷泄漏 {s!r}：{blob[:300]}"
