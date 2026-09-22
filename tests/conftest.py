"""pytest 共享装置：假环境变量 + 服务替身。

三条硬规矩（本文件是它们的唯一执行点）：

1. **测试进程绝不读取 .env 里的真实密钥** —— 在任何项目模块被导入之前，就把
   所有可能携带密钥的环境变量替换成占位值。`config.settings` 在 import 时会调
   `load_dotenv`，而 `load_dotenv` 默认**不覆盖**已存在的环境变量，所以只要这里
   先铺一层，真实 `.env` 就进不来。
2. **测试绝不写进仓库 storage/** —— `DATA_DIR` 一律指向临时目录（每个用例再
   细化到各自的 `tmp_path`）。
3. **全局服务注册表 `services` 用例之间互不污染** —— 每个用例前后对
   `ServiceRegistry` 内部字典做快照/还原。
4. **进程级单例的可变状态同样隔离（R05a）** —— KEEPER（短期记忆管家）、
   TURN_REGISTRY（活跃轮登记）、_BG_TASKS（后台写回任务）都是模块级单例，
   它们不在 `services` 注册表里，之前从没人清：一个用例真跑完一轮 handle，
   下一用例就能在 KEEPER 里读到上一用例的对话（隔离探针实测复现）。
   现在每例前后做快照/还原；后台任务必须等其实际结束（含 to_thread 线程里的
   同步函数跑完）才恢复 DB/registry——只清空任务集合等于什么都没做。
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

# ---- 项目根入 sys.path：让 `import orchestration.pipeline` 之类的顶层导入可用 ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- 1. 在任何项目模块被导入前，铺一层假环境变量（绝不带真 key）----
# 一个会话级别的大临时目录：即便有模块在 import 期就取 DATA_DIR，也落在临时区，
# 不会写进仓库 storage/。
_SESSION_DATA_DIR = tempfile.mkdtemp(prefix="aria-test-data-")

_FAKE_ENV = {
    # LLM（主 + 备用）
    "LLM_API_KEY": "test-llm-key",
    "LLM_BASE_URL": "http://127.0.0.1:9/v1",
    "LLM_MODEL": "test-model",
    "LLM_SUPPORTS_TOOL_CALL": "true",
    # 家族分发（J13-4）也必须钉住：不钉就会从真 .env 漏进来，
    # 作者哪天写了 LLM_FAMILY=qwen，家族相关的测试会跟着悄悄变行为。
    "LLM_FAMILY": "auto",
    "LLM_FALLBACK_FAMILY": "auto",
    "LLM_FALLBACK_API_KEY": "",
    "LLM_FALLBACK_BASE_URL": "http://127.0.0.1:9/v1",
    "LLM_FALLBACK_MODEL": "test-fallback-model",
    # 语音
    "ASR_API_KEY": "",
    "ASR_BASE_URL": "",
    "ASR_MODEL": "test-asr-model",
    "TTS_API_KEY": "",
    "TTS_BASE_URL": "",
    "TTS_MODEL": "test-tts-model",
    "TTS_VOICE": "",
    # 外部服务
    "WEATHER_API_KEY": "",
    "SEARCH_API_KEY": "",
    "IMAGE_GEN_API_KEY": "",
    "IMAGE_GEN_MODEL": "test-image-model",
    # Realtime 实时专线
    "REALTIME_API_KEY": "",
    "REALTIME_MODEL": "test-realtime-model",
    "REALTIME_VOICE": "Maia",
    "REALTIME_WORKSPACE_ID": "",
    # 向量
    "EMBEDDING_API_KEY": "",
    "EMBEDDING_MODEL": "test-embedding-model",
    "EMBEDDING_DIMENSION": "2560",
    # 存储 / 鉴权 / 监听
    "DATA_DIR": _SESSION_DATA_DIR,
    "ACCESS_TOKEN": "test-token",
    "BIND_HOST": "127.0.0.1",
}
os.environ.update(_FAKE_ENV)


def _clear_settings_cache() -> None:
    """清掉 Settings / app config 的 lru_cache，让新设的 DATA_DIR 立刻生效。"""
    try:
        from config.settings import get_settings, load_app_config
    except Exception:
        return
    get_settings.cache_clear()
    load_app_config.cache_clear()


# --------------------------------------------------------------------------
# R05a：进程级单例的跨用例隔离（KEEPER / TURN_REGISTRY / _BG_TASKS）
# --------------------------------------------------------------------------
def _keeper_snapshot(keeper):
    """KEEPER 四张内部字典的快照（锁内；列表浅拷一层防原地 append）。"""
    with keeper._lock:
        return (
            {sid: list(msgs) for sid, msgs in keeper._contexts.items()},
            dict(keeper._summaries),
            dict(keeper._touched),
            {sid: list(msgs) for sid, msgs in keeper._pending.items()},
        )


def _keeper_restore(keeper, snap) -> None:
    """原地把四张字典恢复成快照。不清空重建而是 clear+update：别处可能握着
    KEEPER 实例（_ENGINE 建在唯一 KEEPER 上），字典对象身份不动最稳。"""
    contexts, summaries, touched, pending = snap
    with keeper._lock:
        keeper._contexts.clear()
        keeper._contexts.update(contexts)
        keeper._summaries.clear()
        keeper._summaries.update(summaries)
        keeper._touched.clear()
        keeper._touched.update(touched)
        keeper._pending.clear()
        keeper._pending.update(pending)


def _drain_bg_tasks(tasks: set, timeout: float = 10.0) -> None:
    """等后台写回任务实际结束，超时大声失败（R05a）。

    任务体是 `asyncio.to_thread(fn)`：fn 跑在默认线程池的线程上，**外层
    task.cancel 不会让线程里的同步函数停下来**。所以这里绝不能只把集合清空
    ——必须先取消未完成的任务、再等它们真正 done（asyncio.run 退出时本就会
    cancel 所有任务并 shutdown 默认线程池，这里是对 TestClient 等其他入口的
    兜底），超时没结束就报错：宁可这个用例红，也不能让它的后台写入漏进
    下一个用例。
    """
    for t in list(tasks):
        if not t.done():
            t.cancel()
    deadline = time.monotonic() + timeout
    while any(not t.done() for t in list(tasks)):
        if time.monotonic() > deadline:
            raise AssertionError(
                f"后台写回任务在 {timeout}s 内未结束，会串染下一用例："
                f"{[t for t in tasks if not t.done()]}"
            )
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """每个用例：独立 DATA_DIR（tmp_path）+ 干净的全局服务注册表 + 重置 DB 单例
    + KEEPER / TURN_REGISTRY / _BG_TASKS 快照还原（R05a）。

    get_db() 是模块级懒加载单例：只换 DATA_DIR 不重置它，第二个用例起拿到的
    还是第一个用例临时目录里的库——跨用例数据串味（身份冻结测试实测踩中）。
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ACCESS_TOKEN", "test-token")
    _clear_settings_cache()

    import data.sqlite_store as _sqlite_mod

    saved_db = _sqlite_mod._db
    _sqlite_mod._db = None

    from shared.singletons import services

    saved_services = dict(services._services)
    saved_errors = dict(services._errors)
    services._services = {}
    services._errors = {}

    # 模块级单例不在 services 注册表里，单独快照（R05a）。turn_id 计数器
    # （TurnRegistry._ids）**有意**不还原：跨用例唯一性比可复现更重要。
    import orchestration.pipeline as _pipeline_mod
    from orchestration.cancellation import TURN_REGISTRY

    keeper = _pipeline_mod.KEEPER
    saved_keeper = _keeper_snapshot(keeper)
    saved_active = dict(TURN_REGISTRY._active)
    saved_bg = set(_pipeline_mod._BG_TASKS)

    # R06a：装配状态（_ASSEMBLY_STATE）也是进程级单例——每个用例必须从
    # "未装配"开始。否则上一个用例留下的 status=ready 会让下一个用例的
    # lifespan 跳过 bootstrap，直接面对被本 fixture 清空的 services 注册表
    # （集成冒烟实测：kv_store 不可用）。装配屏障的 Event 同步清零。
    import main as _main_mod

    _main_mod._ASSEMBLY_STATE.update(
        {"servers": 0, "watcher": None, "status": "idle", "error": None}
    )
    _main_mod._ASSEMBLY_DONE.clear()

    try:
        yield
    finally:
        # 顺序即语义：先等后台线程真正收工（它们可能还在写库/写 KEEPER），
        # 再恢复内存状态，最后才动 DB 单例——反过来会让后台线程写进恢复后的库。
        _drain_bg_tasks(_pipeline_mod._BG_TASKS)
        _keeper_restore(keeper, saved_keeper)
        with TURN_REGISTRY._lock:
            TURN_REGISTRY._active.clear()
            TURN_REGISTRY._active.update(saved_active)
        _pipeline_mod._BG_TASKS.clear()
        _pipeline_mod._BG_TASKS.update(saved_bg)

        try:
            if _sqlite_mod._db is not None and _sqlite_mod._db is not saved_db:
                _sqlite_mod._db.close()
        except Exception:
            pass
        _sqlite_mod._db = saved_db
        services._services = saved_services
        services._errors = saved_errors
        _clear_settings_cache()


# --------------------------------------------------------------------------
# 服务替身（参考工作区 verify_c1b.py 的写法整理而来）
# --------------------------------------------------------------------------
class FakeLLM:
    """可脚本化的 LLM 替身，覆盖 LLMClient 对外三个能力。

    - `chat` / `chat_with_tools` 是同步的；用 chat_responses / tool_responses 排队，
      队列里塞 Exception 实例就表示"这次调用抛错"。
    - `astream_chat` 是异步生成器；stream_scripts 每个元素是一段分块列表，
      分块可以是 str（正常吐出）或 Exception（吐到这里抛错）。
    """

    def __init__(self, chat_responses=None, tool_responses=None, stream_scripts=None):
        self.chat_responses = list(chat_responses or [])
        self.tool_responses = list(tool_responses or [])
        self.stream_scripts = [list(s) for s in (stream_scripts or [])]
        self.chat_calls = []
        self.tool_calls = []
        self.stream_calls = []

    def chat(self, messages, temperature=None, max_tokens=None):
        self.chat_calls.append(messages)
        if not self.chat_responses:
            raise RuntimeError("FakeLLM: 没有脚本化的 chat 响应了")
        resp = self.chat_responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    def chat_with_tools(self, messages, tools_catalog):
        self.tool_calls.append((messages, tools_catalog))
        if not self.tool_responses:
            raise RuntimeError("FakeLLM: 没有脚本化的 chat_with_tools 响应了")
        resp = self.tool_responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def astream_chat(
        self, messages, temperature=None, max_tokens=None, enable_thinking=None, **kwargs
    ):
        self.stream_calls.append(messages)
        if not self.stream_scripts:
            return
        chunks = self.stream_scripts.pop(0)
        for chunk in chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class FakePerception:
    """感知结果替身：run() 固定返回预设的 (IntentResult, EmotionResult, subtext, extras)。"""

    def __init__(self, intent=None, emotion=None, subtext="", extras=None):
        from shared.types import EmotionResult, IntentResult, PerceptionExtras

        self.intent = intent or IntentResult(intent="chat")
        self.emotion = emotion or EmotionResult()
        self.subtext = subtext
        self.extras = extras or PerceptionExtras()
        self.calls = []

    def run(self, text, recent_context=None):
        self.calls.append((text, recent_context))
        return self.intent, self.emotion, self.subtext, self.extras


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_perception():
    return FakePerception()


@pytest.fixture
def register_services():
    """把替身注册进全局 services 的便捷函数：`register_services(llm=..., kv_store=...)`。"""
    from shared.singletons import services

    def _register(**named):
        for name, obj in named.items():
            services.register(name, obj)

    return _register
