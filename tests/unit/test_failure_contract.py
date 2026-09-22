"""R27b/R27c：失败契约测试。

R27b 契约（《计划与设计.md》/ 8.8 分层纪律）：
- openai SDK 的连接/超时/API 响应异常在 llm_client 最窄边界转成
  ExternalServiceError（来源/原因码/可重试），并计入主备熔断计数；
- TypeError/NameError 等本地装配缺陷原样上抛：**不**转窄载体、**不**计入
  熔断（程序错误不是供应商故障）、**不**烧备用供应商；
- 管道里本地缺陷冒泡到 handle() → PipelineInternalError：零重跑、零业务提交；
- 外部失败走允许的降级（error 事件 external_degraded → handle 落 llm 兜底话）。

R27c 契约：ASR/TTS/realtime/weather 等适配器在同一窄边界把网络/SDK/协议失败
转成带来源的 ExternalServiceError；本地缺陷原样上抛。

不触网：SDK 异常用替身/可构造异常模拟，LLMClient 用 __new__ 绕过构造。
"""

from __future__ import annotations

import asyncio
import threading
import types

import openai
import pytest

from shared.types import ExternalServiceError, PipelineInternalError
from tests.unit.test_pipeline_e2e import FakeLLM, FakeTTS  # noqa: F401


# --------------------------------------------------------------------------
# 1. llm_client 同步/流式边界（R27b）
# --------------------------------------------------------------------------

class _Completions:
    """替身 completions：create() 按脚本抛异常或返回固定响应，并计数。"""

    def __init__(self, exc, counter: dict, key: str):
        self._exc = exc
        self._counter = counter
        self._key = key

    def create(self, **kwargs):
        self._counter[self._key] += 1
        if self._exc is not None:
            raise self._exc
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="正常回复"))])


def _make_client(main_exc=None, fallback_exc=None):
    """绕过 __init__（要真 key）手工装配 LLMClient 的熔断/客户端内部状态。"""
    from tools.llm_client import LLMClient

    calls = {"main": 0, "fallback": 0}
    c = LLMClient.__new__(LLMClient)
    c._model = "test-model"
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=_Completions(main_exc, calls, "main")))
    c._fallback = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=_Completions(fallback_exc, calls, "fallback")))
    c._afallback = None
    c._fallback_family = "claude"
    c._fallback_model = "test-fallback-model"
    c._family = "claude"
    c._cb_lock = threading.Lock()
    c._opened_at = None
    c._cooldown = 120.0
    c._error_count = 0
    c._max_errors = 2
    c._main_fail_streak = 0  # R26a：主连续失败连击（开熔断的唯一依据）
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
    assert c._error_count == 1
    assert calls["fallback"] == 1


def test_sync_sdk_error_with_healthy_fallback_succeeds():
    """主挂、备用健康：正常返回（主失败 +1 后被备用成功衰减回 0，J13 取舍保留）。"""
    c, calls = _make_client(main_exc=openai.OpenAIError("主挂"))
    resp = c._complete(model="test-model", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "正常回复"
    assert c._error_count == 0
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

    async def _acreate(**kwargs):
        raise openai.OpenAIError("主挂")

    async def _afcreate(**kwargs):
        raise openai.OpenAIError("备挂")

    c._aclient = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_acreate)))
    c._afallback = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_afcreate)))

    async def _run():
        return [s async for s in c.astream_chat([{"role": "user", "content": "hi"}])]

    with pytest.raises(ExternalServiceError) as ei:
        asyncio.run(_run())
    assert ei.value.reason_code == "all_providers_failed"
    assert c._error_count == 1


def test_stream_interrupted_after_tokens_raises_external():
    """已吐 token 后流中断：转 stream_interrupted（retryable=False）；
    主模型失败计一次；备用不该被走到。"""
    from tools.llm_client import LLMClient

    c, _calls = _make_client(main_exc=None)

    async def _agen():
        # chunk 必须是 SDK 形状（带 choices.delta.content）：否则 _delta_of 取
        # 不到文本，emitted 恒 False，会被当成"没吐过 token"走备用
        yield types.SimpleNamespace(choices=[types.SimpleNamespace(
            delta=types.SimpleNamespace(content="正常半句"))])
        raise openai.OpenAIError("流中断")

    async def _acreate(**kwargs):
        return _agen()

    c._aclient = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_acreate)))
    c._afallback = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_acreate)))

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


def _collect_stream(pipeline, msg) -> list:
    """消费 astream 事件流（与 test_pipeline_e2e 同款，独立小函数避免跨文件耦合）。"""
    async def _run():
        evs = []
        async for ev in pipeline.astream(msg):
            evs.append(ev)
        return evs

    return asyncio.run(_run())


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


def test_external_error_degrades_via_handle(register_services):
    """外部失败（窄载体）→ 管道内降级消化（generate 兜底话）→ done 收尾；
    无 internal_error（那是程序错误的专属信号）；handle 返回兜底话不抛异常。"""
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
    assert not any(e.get("type") == "error" and e.get("code") == "internal_error" for e in evs)
    assert any(e.get("type") == "done" for e in evs), f"外部失败必须降级收尾，实测 {[e.get('type') for e in evs]}"

    reply = asyncio.run(
        DialoguePipeline().handle(InputMessage(text="随便聊聊", session_id=sid))
    )
    assert reply.text, "外部失败必须降级成兜底话而不是空回复"


# --------------------------------------------------------------------------
# 3. R27c：各外部适配器接同一错误契约——外部失败→窄载体可降级；本地缺陷→原样上抛
# --------------------------------------------------------------------------

def test_embedding_external_failure_converts(monkeypatch):
    """embedding：dashscope 网络失败 → ExternalServiceError("embedding")。"""
    import data.embedding_client as emb_mod

    def boom(**kwargs):
        raise RuntimeError("模拟网络中断")

    monkeypatch.setattr(emb_mod.dashscope.MultiModalEmbedding, "call", boom)
    emb = emb_mod.QwenEmbedding()
    with pytest.raises(ExternalServiceError) as ei:
        emb.embed_query("一句话")
    assert ei.value.source == "embedding"
    assert ei.value.reason_code == "request_failed"


def test_embedding_protocol_bad_data_is_external_protocol_error(monkeypatch):
    """embedding：返回体结构坏 → bad_protocol（外部协议错误，不可重试）。"""
    import data.embedding_client as emb_mod

    class _Rsp:
        status_code = 200
        output = {"embeddings": [{"nope": 1}]}  # 缺 "embedding" 键

    monkeypatch.setattr(emb_mod.dashscope.MultiModalEmbedding, "call",
                        lambda **kw: _Rsp())
    emb = emb_mod.QwenEmbedding()
    with pytest.raises(ExternalServiceError) as ei:
        emb.embed_query("一句话")
    assert ei.value.reason_code == "bad_protocol"
    assert ei.value.retryable is False


def test_embedding_local_defect_propagates():
    """embedding：本地装配缺陷（texts=None）原样上抛 TypeError——不转窄载体。"""
    import data.embedding_client as emb_mod

    emb = emb_mod.QwenEmbedding()
    with pytest.raises(TypeError):
        emb.embed_documents(None)


def test_asr_external_and_local(monkeypatch):
    """ASR：_run_bounded 超时（RuntimeError）→ ExternalServiceError；
    本地缺陷（audio=None 走格式装配）原样上抛。"""
    import tools.speech as speech_mod

    monkeypatch.setenv("ASR_API_KEY", "test-asr-key")
    from config.settings import get_settings
    get_settings.cache_clear()

    def fake_run_bounded(fn, timeout_s, what):
        raise RuntimeError(f"{what}超时（{timeout_s:.0f}s）")

    monkeypatch.setattr(speech_mod, "_run_bounded", fake_run_bounded)
    monkeypatch.setattr(speech_mod, "timeout_seconds", lambda key, default: 5.0)

    with pytest.raises(ExternalServiceError) as ei:
        speech_mod.ASRTool().transcribe(b"abc", fmt="wav", sample_rate=16000)
    assert ei.value.source == "asr"

    # 本地缺陷：audio=None → 嗅探/装配炸，原样上抛（不是窄载体）
    with pytest.raises((TypeError, AttributeError)):
        speech_mod.ASRTool().transcribe(None, fmt="pcm", sample_rate=16000)


def test_tts_external_and_local(monkeypatch):
    """TTS：SDK 失败 → ExternalServiceError("tts")；本地缺陷（text=None 进
    情绪拆分）原样上抛。SDK 边界用替身合成器注入。"""
    from tools.speech import TTSTool

    import tools.speech as speech_mod

    monkeypatch.setenv("TTS_API_KEY", "test-tts-key")
    from config.settings import get_settings
    get_settings.cache_clear()

    class _Synth:
        def __init__(self, **kwargs):
            pass

        def call(self, text, timeout_millis=None):
            raise OSError("模拟 SDK 网络失败")

    monkeypatch.setattr(speech_mod, "SpeechSynthesizer", _Synth)
    with pytest.raises(ExternalServiceError) as ei:
        TTSTool().synthesize("你好呀")
    assert ei.value.source == "tts"
    assert ei.value.reason_code == "request_failed"

    # 本地缺陷：text=123（非字符串）→ 情绪拆分装配炸，原样上抛（不是窄载体）
    with pytest.raises((TypeError, AttributeError)):
        TTSTool().synthesize(123)


def test_weather_external_and_local(monkeypatch):
    """天气：httpx 网络失败 → ExternalServiceError("weather")；
    响应缺字段（本地装配路径 KeyError）原样上抛。"""
    import tools.external as ext_mod

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"name": "杭州"}]}  # 故意缺 latitude

    class _Client:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            if "geocoding" in url:
                return _Resp()
            raise KeyError("latitude（本地装配路径）")

    monkeypatch.setattr(ext_mod.httpx, "Client", _Client)
    with pytest.raises(KeyError):
        ext_mod.WeatherTool().query("杭州")

    class _ClientNetFail(_Client):
        def get(self, url, params=None):
            raise ext_mod.httpx.ConnectError("模拟 DNS 失败")

    monkeypatch.setattr(ext_mod.httpx, "Client", _ClientNetFail)
    with pytest.raises(ExternalServiceError) as ei:
        ext_mod.WeatherTool().query("杭州")
    assert ei.value.source == "weather"


def test_realtime_connect_failure_converts():
    """realtime：websockets 握手失败 → ExternalServiceError("realtime")。"""
    import tools.realtime as rt_mod

    async def _run():
        c = rt_mod.RealtimeDialogClient("sess-x", instructions="i")
        await c.connect()

    orig = rt_mod.websockets.connect

    async def _boom(*a, **kw):
        raise OSError("模拟拒绝连接")

    rt_mod.websockets.connect = _boom
    try:
        with pytest.raises(ExternalServiceError) as ei:
            asyncio.run(_run())
        assert ei.value.source == "realtime"
        assert ei.value.reason_code == "connect_failed"
    finally:
        rt_mod.websockets.connect = orig
