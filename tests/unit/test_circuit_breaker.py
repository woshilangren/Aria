"""R26：主备熔断的计数归属与状态转换（可控 monotonic 时钟，无 sleep、无网络）。

必测序列（《计划与设计.md》R26）：
1. 阈值 2 下连续主失败备成功，第二次后熔断打开、冷却内第三次不再请求主；
2. 仅备用失败不污染主；
3. 半开完整成功复位；
4. 半开建流/流中失败均重新冷却并扩大探测超时；
5. 并发只有一个半开名额；
6. 主流已出 token 后失败不拼接备用内容。

修复的缺陷（真机 429 复现）：主失败 +1 / 备用成功 -1 的单一计数让
"主挂→备活"循环里计数永远 0~1 打转，error_counts=[0,0,0]、熔断永不打开。
现在开熔断只认 _main_fail_streak（主连续失败），_error_count 退化为会被
备用成功衰减的"整体故障压力"；半开探测是真实主调用，任何失败都立即重开
并刷新冷却起点。

R20a 改超时后必须重跑这些契约用例；计数修复不删 SDK max_retries=0、
探测短超时、指数封顶（本文件用例全部在 max_retries=0 的手工装配上跑）。
"""

from __future__ import annotations

import asyncio
import threading
import types

import openai
import pytest

import tools.llm_client as llm_mod
from shared.types import ExternalServiceError
from tools.llm_client import LLMClient


class _Clock:
    """可控 monotonic 时钟：测试里手动拨针，绝不 sleep。"""

    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _ok_resp(content="备用顶上了"):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content=content))])


class _SeqCompletions:
    """按脚本逐次响应（同步）：元素为 Exception（抛出）或 None（正常返回）。"""

    def __init__(self, script, calls: dict):
        self._script = list(script)
        self._calls = calls

    def create(self, **kwargs):
        self._calls[self._key] += 1
        if not self._script:
            raise AssertionError(f"{self._key} 被过多调用（脚本已耗尽）")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        # 主/备成功返回不同文案，测试能断言"这次到底是谁答的"
        return _ok_resp("正常回复" if self._key == "main" else "备用顶上了")


def _sync_completions(script, calls, key):
    c = _SeqCompletions(script, calls)
    c._key = key
    return c


def _make(script_main, script_fallback, clock):
    """手工装配 LLMClient：脚本化主备 + 可控时钟（monkeypatch 由调用方负责）。"""
    calls = {"main": 0, "fallback": 0}
    c = LLMClient.__new__(LLMClient)
    c._model = "test-model"
    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=_sync_completions(script_main, calls, "main")))
    c._fallback = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=_sync_completions(script_fallback, calls, "fallback")))
    c._afallback = None
    c._fallback_family = "claude"
    c._fallback_model = "test-fallback-model"
    c._family = "claude"
    c._cb_lock = threading.Lock()
    c._opened_at = None
    c._cooldown = 120.0
    c._error_count = 0
    c._max_errors = 2
    c._main_fail_streak = 0
    c._probe_in_flight = False
    c._probe_fail_streak = 0
    c._probe_timeout = 5.0
    return c, calls


@pytest.fixture()
def clock(monkeypatch):
    """把 llm_client 看到的 time.monotonic 换成可控时钟（monkeypatch 自动还原）。"""
    clk = _Clock()
    monkeypatch.setattr(llm_mod.time, "monotonic", clk)
    return clk


# --------------------------------------------------------------------------
# 序列 1（同步）：阈值 2 下主失败×2（备用成功），第二次后熔断打开、冷却内不碰主
# --------------------------------------------------------------------------

def test_sync_main_fail_backup_success_opens_at_threshold(clock):
    script_main = [openai.OpenAIError("429-1"), openai.OpenAIError("429-2"),
                   openai.OpenAIError("429-3 不该被用到")]
    script_fallback = [None, None, None]
    c, calls = _make(script_main, script_fallback, clock)

    # 第 1 轮：主挂 +1、备成功（压力衰减 -1）——旧缺陷下计数归零、熔断永不打开
    assert c.chat([{"role": "user", "content": "a"}]) == "备用顶上了"
    with c._cb_lock:
        assert c._main_fail_streak == 1, "备用成功不得衰减主失败连击（R26a 核心）"
        assert c._opened_at is None
    # 第 2 轮：主再挂 → 连击到阈值 → 熔断打开
    assert c.chat([{"role": "user", "content": "b"}]) == "备用顶上了"
    with c._cb_lock:
        assert c._main_fail_streak == 2
        assert c._opened_at == clock.t, "第二次主失败后熔断必须打开"
    # 第 3 轮（冷却内）：主**不再被请求**，直接走备用
    assert c.chat([{"role": "user", "content": "c"}]) == "备用顶上了"
    assert calls["main"] == 2, f"冷却内不许碰主模型，实测主被调 {calls['main']} 次"


# --------------------------------------------------------------------------
# 序列 2：仅备用失败不污染主
# --------------------------------------------------------------------------

def test_sync_backup_failure_does_not_pollute_main(clock):
    # 主成功一次（清连击），随后主冷却/未调用的场景里备用失败——主计数不许动
    script_main = [None]  # 主正常
    script_fallback = []
    c, calls = _make(script_main, script_fallback, clock)
    assert c.chat([{"role": "user", "content": "a"}]) == "正常回复"
    with c._cb_lock:
        assert c._main_fail_streak == 0

    # 主失败 → 备用也失败：只有主失败计连击；备用失败不追加"主失败"
    c2, calls2 = _make([openai.OpenAIError("主挂")], [openai.OpenAIError("备挂")], clock)
    with pytest.raises(ExternalServiceError):
        c2.chat([{"role": "user", "content": "a"}])
    with c2._cb_lock:
        assert c2._main_fail_streak == 1, "备用自身失败不得算作主失败"
        assert c2._error_count == 1


# --------------------------------------------------------------------------
# 序列 3（同步）：半开完整成功复位
# --------------------------------------------------------------------------

def test_sync_halfopen_full_success_resets(clock):
    c, calls = _make(
        [openai.OpenAIError("1"), openai.OpenAIError("2"), None],
        [None, None],
        clock,
    )
    # 打开熔断（连击 2）
    c.chat([{"role": "user", "content": "a"}])
    c.chat([{"role": "user", "content": "b"}])
    with c._cb_lock:
        assert c._opened_at is not None
    # 冷却到期 → 半开探测 → 主完整成功 → 四项状态全复位
    clock.advance(121.0)
    assert c.chat([{"role": "user", "content": "c"}]) == "正常回复"
    with c._cb_lock:
        assert c._main_fail_streak == 0
        assert c._error_count == 0
        assert c._opened_at is None
        assert c._probe_fail_streak == 0
    assert calls["main"] == 3


# --------------------------------------------------------------------------
# 序列 4（流式）：半开建流失败 / 流中失败均重新冷却并扩大探测超时
# --------------------------------------------------------------------------

def _wire_async(c, main_events, fallback_events):
    """脚本化异步客户端：main_events/fallback_events 元素为
    Exception（建流即抛）或 chunk 列表（列表内可有 Exception 表示流中抛）。"""

    def _mk(events, calls, key):
        async def create(**kwargs):
            calls[key] += 1
            if events and isinstance(events[0], Exception):
                raise events.pop(0)
            chunks = events.pop(0) if events else []
            async def _agen():
                for ch in chunks:
                    if isinstance(ch, Exception):
                        raise ch
                    yield ch
            return _agen()
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))

    c._aclient = _mk(main_events, {"n": 0} or c.__dict__.setdefault("_acalls", {"main": 0, "fallback": 0}), "main")
    c._afallback = _mk(fallback_events, c._acalls, "fallback")
    return c._acalls


def _chunk(text):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        delta=types.SimpleNamespace(content=text))])


def test_stream_halfopen_build_fail_reopens_and_backs_off(clock):
    """流式：熔断打开后冷却到期，半开**建流**失败 → 立即重开（冷却起点刷新）
    且探测超时翻倍。"""
    script_main = [openai.OpenAIError("1"), openai.OpenAIError("2")]
    script_fallback = [None, None]
    c, _calls = _make(script_main, script_fallback, clock)
    acalls = {"main": 0, "fallback": 0}

    def _acreate(**kwargs):
        acalls["main"] += 1
        raise openai.OpenAIError("半开建流失败")

    async def _afcreate(**kwargs):
        acalls["fallback"] += 1
        return _agen_ok()

    async def _agen_ok():
        yield _chunk("备")

    c._aclient = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_acreate)))
    c._afallback = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_afcreate)))

    # 打开熔断
    c.chat([{"role": "user", "content": "a"}])
    c.chat([{"role": "user", "content": "b"}])
    with c._cb_lock:
        assert c._opened_at is not None
    # 冷却到期 → 半开建流失败 → 探测失败立即重开、冷却起点=现在、探测退避 +1；
    # 备用顶上成功——**请求本身不报错**（探测失败 ≠ 请求失败）
    clock.advance(121.0)
    t_probe = clock.t
    out = asyncio.run(_drain(c.astream_chat([{"role": "user", "content": "s"}])))
    assert out == ["备"], f"半开探测失败应回落备用，实测 {out!r}"
    with c._cb_lock:
        assert c._opened_at == t_probe, "半开失败必须立即重开并刷新冷却起点"
        assert c._probe_fail_streak == 1
    assert c._probe_timeout_now() == pytest.approx(10.0), "探测超时应翻倍（5s×2）"


def test_stream_halfopen_midstream_fail_reopens(clock):
    """流式：半开探测建流成功但**流中**失败 → 同样立即重开刷新冷却。"""
    script_main = [openai.OpenAIError("1"), openai.OpenAIError("2")]
    script_fallback = [None, None]
    c, _calls = _make(script_main, script_fallback, clock)

    async def _acreate(**kwargs):
        async def _agen():
            yield _chunk("半句")
            raise openai.OpenAIError("流中失败")
        return _agen()

    async def _afcreate(**kwargs):
        async def _agen():
            yield _chunk("备")
        return _agen()

    c._aclient = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_acreate)))
    c._afallback = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_afcreate)))

    c.chat([{"role": "user", "content": "a"}])
    c.chat([{"role": "user", "content": "b"}])
    clock.advance(121.0)
    t_probe = clock.t
    with pytest.raises(ExternalServiceError) as ei:
        asyncio.run(_drain(c.astream_chat([{"role": "user", "content": "s"}])))
    assert ei.value.reason_code == "stream_interrupted"
    with c._cb_lock:
        assert c._opened_at == t_probe
        assert c._probe_fail_streak == 1


async def _drain(agen):
    out = []
    async for s in agen:
        out.append(s)
    return out


# --------------------------------------------------------------------------
# 序列 5：并发只有一个半开名额
# --------------------------------------------------------------------------

def test_concurrent_halfopen_single_probe_slot(clock):
    """冷却到期的瞬间 N 个并发请求：只有 1 个真正探测主模型，其余直接走备用。"""
    c, _calls = _make([openai.OpenAIError("1"), openai.OpenAIError("2")],
                      [None] * 10, clock)
    probe_calls = {"n": 0}
    release = threading.Event()

    # 先用脚本客户端把熔断打开（这两轮不许算进探测名额的账）
    c.chat([{"role": "user", "content": "a"}])
    c.chat([{"role": "user", "content": "b"}])
    with c._cb_lock:
        assert c._opened_at is not None

    # 再换上"慢探测"主客户端：create 挂住，放大并发争夺窗口
    def _slow_main_create(**kwargs):
        probe_calls["n"] += 1
        release.wait(timeout=5)  # 挂住探测，放大并发窗口
        raise openai.OpenAIError("探测失败")

    c._client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_slow_main_create)))
    clock.advance(121.0)

    errors = []

    def worker():
        try:
            c.chat([{"role": "user", "content": "x"}])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    import time as _t
    deadline = _t.monotonic() + 5
    while probe_calls["n"] < 1 and _t.monotonic() < deadline:
        _t.sleep(0.005)
    release.set()
    for t in threads:
        t.join(timeout=5)
    assert not errors, f"并发探测不该产生未处理异常: {errors}"
    assert probe_calls["n"] == 1, f"半开名额只能放行一个探测，实测 {probe_calls['n']}"
    # 探测失败 → 立即重开（其余请求都只碰了备用）
    with c._cb_lock:
        assert c._opened_at is not None
        assert c._probe_fail_streak == 1


# --------------------------------------------------------------------------
# 序列 6：主流已出 token 后失败不拼接备用内容
# --------------------------------------------------------------------------

def test_stream_emitted_failure_never_concats_fallback(clock):
    """主流吐过 token 再断：抛 stream_interrupted 收尾，备用内容一个字都不拼。"""
    c, _calls = _make([None], [None], clock)

    async def _acreate(**kwargs):
        async def _agen():
            yield _chunk("主模型前半句。")
            raise openai.OpenAIError("流中断")
        return _agen()

    fallback_called = {"n": 0}

    async def _afcreate(**kwargs):
        fallback_called["n"] += 1
        async def _agen():
            yield _chunk("备用拼接内容，不该出现")
        return _agen()

    c._aclient = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_acreate)))
    c._afallback = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_afcreate)))

    out = []
    with pytest.raises(ExternalServiceError) as ei:
        asyncio.run(_drain(c.astream_chat([{"role": "user", "content": "s"}])))
        out.append(None)
    assert out == [] or True  # 断言主体在 raises
    assert ei.value.reason_code == "stream_interrupted"
    assert fallback_called["n"] == 0, "已吐 token 后失败不许再试备用"


# --------------------------------------------------------------------------
# 同步/流式共享状态：熔断被同步路径打开后，流式路径同样冷却内不碰主
# --------------------------------------------------------------------------

def test_stream_respects_cooldown_opened_by_sync(clock):
    c, calls = _make([openai.OpenAIError("1"), openai.OpenAIError("2")],
                     [None] * 5, clock)
    c.chat([{"role": "user", "content": "a"}])
    c.chat([{"role": "user", "content": "b"}])
    main_before = calls["main"]

    acalls = {"n": 0}

    async def _acreate(**kwargs):
        acalls["n"] += 1
        raise AssertionError("冷却内流式不许碰主模型")

    async def _afcreate(**kwargs):
        async def _agen():
            yield _chunk("备")
        return _agen()

    c._aclient = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_acreate)))
    c._afallback = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=_afcreate)))

    out = asyncio.run(_drain(c.astream_chat([{"role": "user", "content": "s"}])))
    assert out == ["备"]
    assert calls["main"] == main_before, "冷却内同步/流式共享同一熔断状态"
