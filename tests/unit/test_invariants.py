"""不变式测试：把 CLAUDE.md 的"关键单例与不变式"用断言钉死。

清单（来自本仓《计划与设计.md》批次 B4 的 8 条不变式，docs/archive/ 已删但项目方有完整记录）：
1. KEEPER 唯一：SessionMemoryKeeper 全项目只有一份
2. TurnRegistry 拒 cancel(turn_id=None)（需先传 session_id）：F4 防误杀
3. 取消不写：写回里有 _check_cancel 检查点
4. writeback 顺序：note_user_reply 必须在 RelationshipTracker.update 之后
5. proactive 默认关：proactive.enabled 默认 False
6. 家族分发：_is_qwen 必须正确判定 qwen / claude / gpt
7. portrait_tag_limit ≤ 5：与 prompt "不超过 5 个" 一致
8. _KV_UPDATE_TABLES 白名单：非 profile/portrait/relationship/self 一律 ValueError
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from capability import memory as mem_mod
from capability import proactive as proactive_mod
from orchestration import cancellation
from orchestration import pipeline as pipeline_mod


# =================== 1. KEEPER 唯一 ===================

def test_keeper_is_module_singleton() -> None:
    """orchestration/pipeline.py 顶层 KEEPER 是 SessionMemoryKeeper 实例。"""
    assert isinstance(pipeline_mod.KEEPER, pipeline_mod.SessionMemoryKeeper)


# =================== 2. TurnRegistry 拒 cancel(turn_id=None) ===================

def test_turn_registry_rejects_none_turn_id() -> None:
    """cancel(session_id, turn_id=None) 必须拒绝（F4）：返回 False，不动活跃轮。

    实现选了静默路径（return False）而不是抛异常——生产环境不污染日志，
    客户端不会因拿到 None 崩。这条断言保证：调用没 raise、活跃轮没被动。
    """
    reg = cancellation.TurnRegistry()
    sid = "test-sess"
    handle = reg.start(sid)
    assert handle.turn_id is not None, "start 应返回带 turn_id 的 handle"
    result = reg.cancel(sid, turn_id=None)
    assert result is False, "turn_id=None 必须返回 False 而不是 True"
    assert reg.get(sid) is handle, "turn_id=None 不能误杀现有活跃轮"


# =================== 3. _writeback 有取消检查点 ===================

def test_writeback_has_cancel_checkpoints() -> None:
    """_writeback 函数体内必须有 ≥2 次 self._check_cancel(handle) 调用。

    每个子步骤（KEEPER / chat_log / distiller / tracker / note_user_reply）之前
    各插一个——取消的轮什么都不写（CLAUDE.md 有意的设计取舍 #3）。
    """
    src = Path("orchestration/pipeline.py").read_text(encoding="utf-8")
    fn_start = src.find("def _writeback(")
    fn_end = src.find("\n    def ", fn_start + 1)
    body = src[fn_start:fn_end if fn_end > 0 else None]
    count = body.count("_check_cancel(handle)")
    assert count >= 4, f"_writeback 应有 ≥4 个取消检查点（KEEPER/chat_log/distiller/tracker），实测 {count}"


# =================== 4. writeback 顺序：tracker 在 note_user_reply 之前 ===================

def test_writeback_tracker_before_note_user_reply() -> None:
    """note_user_reply(session_id) 必须在 RelationshipTracker().update 之后调用。

    否则首轮会被预创建空 relationship、tracker 的"首次初始化"分支失效，
    default_intimacy 被吞成 0（开发日志 BUG-20260916-首聊 已踩中）。
    """
    src = Path("orchestration/pipeline.py").read_text(encoding="utf-8")
    fn_start = src.find("def _writeback(")
    fn_end = src.find("\n    def ", fn_start + 1)
    body = src[fn_start:fn_end if fn_end > 0 else None]
    tracker_pos = body.find("RelationshipTracker")
    note_pos = body.find("note_user_reply(session_id)")
    assert tracker_pos > 0, "_writeback 必须实例化 RelationshipTracker"
    assert note_pos > 0, "_writeback 必须调用 note_user_reply"
    assert tracker_pos < note_pos, (
        f"顺序反了：tracker 在 L{fn_start + tracker_pos}，"
        f"note_user_reply 在 L{fn_start + note_pos}。"
    )


# =================== 5. proactive 默认关 ===================

def test_proactive_enabled_defaults_false() -> None:
    """proactive.enabled 默认 False（CLAUDE.md "主动性宁缺毋滥"）。"""
    cfg = proactive_mod.load_app_config()
    assert cfg["proactive"]["enabled"] is False


# =================== 6. 家族分发：_is_qwen 必须正确判定 ===================

def _make_llm_with_model(model: str):
    """绕过 LLMClient.__init__（它要真 key + 网络）只看 _is_qwen 的判定。"""
    from tools.llm_client import LLMClient
    client = LLMClient.__new__(LLMClient)
    client._model = model
    return client


def test_is_qwen_recognizes_qwen_family() -> None:
    client = _make_llm_with_model("qwen3.8-flash")
    assert client._is_qwen() is True


def test_is_qwen_recognizes_qwen2_5() -> None:
    """qwen2.5（任何带 qwen 子串的家族成员）应被认作 qwen 家族。"""
    client = _make_llm_with_model("qwen2.5-72b-instruct")
    assert client._is_qwen() is True


def test_is_qwen_rejects_claude() -> None:
    client = _make_llm_with_model("claude-sonnet-5")
    assert client._is_qwen() is False


def test_is_qwen_rejects_gpt() -> None:
    client = _make_llm_with_model("gpt-4o")
    assert client._is_qwen() is False


# =================== 7. portrait_tag_limit 与 prompt 一致 ===================

def test_portrait_tag_limit_matches_prompt() -> None:
    """K6 已统一口径：portrait_tag_limit ≤ 5（与 prompt "不超过 5 个" 一致）。"""
    cfg = mem_mod.load_app_config()
    limit = int(cfg["memory"]["portrait_tag_limit"])
    assert limit <= 5


# =================== 8. _KV_UPDATE_TABLES 白名单 ===================

def test_kv_update_rejects_non_whitelisted_tables(tmp_path, monkeypatch, register_services) -> None:
    """KVStoreTool.update 只能更新白名单（profile/portrait/relationship/self）内的表。

    以前用 writable 标记判断，错误类型不统一；现在非白名单一律 ValueError。
    """
    from tools.storage import KVStoreTool

    # 替身：让 KVStoreTool 可以构造（_profile/_portrait 等会被实例化时连真 sqlite）
    # 这里用 monkeypatch 走 SQLiteStorage 的 tmp_path
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from config.settings import get_settings, load_app_config
    get_settings.cache_clear()
    load_app_config.cache_clear()

    kv = KVStoreTool()
    for table in ("session", "image", "route_config", "persona_config"):
        with pytest.raises(ValueError):
            kv.update(table, "x", lambda d: d)  # noqa: ARG005


# =================== 9. I10：双端口只能装配一次 ===================

def test_double_lifespan_assembles_once(monkeypatch) -> None:
    """两个 server 跑同一个 app 对象时，装配只许发生一次。

    `python main.py` 会在 daemon 线程里再起一个 uvicorn.Server 跑 8443，
    和主线程的 8000 共用同一个 `app`——于是 lifespan 被执行两遍。
    没有 _ASSEMBLY_STATE 守卫时的三个后果：
      ① bootstrap() 两遍：服务重复构造、services registry 被二次覆盖；
      ② IdleDiaryWatcher 两个实例：防重入的 _done 是**实例级**集合，跨实例不去重
         → 同一天同一会话可能写两篇日记、主动消息双发；
      ③ _ensure_access_token 两个线程并发 → 可能各生成一个口令互相覆盖 .env。
    """
    import asyncio

    from fastapi import FastAPI

    import main as main_mod

    calls = {"token": 0, "bootstrap": 0, "new": 0, "start": 0, "stop": 0}

    def _count(key):
        def _inc(*_a, **_k):
            calls[key] += 1
        return _inc

    monkeypatch.setattr(main_mod, "_ensure_access_token", _count("token"))
    monkeypatch.setattr(main_mod, "bootstrap", _count("bootstrap"))
    # 用裸 FastAPI 顶掉 build_app：隔离 gradio 的 lifespan，本测试只验装配守卫
    monkeypatch.setattr(main_mod, "build_app", lambda: FastAPI())

    class FakeWatcher:
        def __init__(self) -> None:
            calls["new"] += 1

        def start(self) -> None:
            calls["start"] += 1

        def stop(self) -> None:
            calls["stop"] += 1

    # lifespan 内部是 `from capability.proactive import IdleDiaryWatcher`（调用时才查），
    # 所以 patch 模块属性就生效
    monkeypatch.setattr(proactive_mod, "IdleDiaryWatcher", FakeWatcher)
    # 守卫是模块级状态，必须归零再测，否则上一个用例的残留会让 first 判定失真
    monkeypatch.setitem(main_mod._ASSEMBLY_STATE, "servers", 0)
    monkeypatch.setitem(main_mod._ASSEMBLY_STATE, "watcher", None)

    app = main_mod.create_app()
    factory = app.router.lifespan_context

    async def _two_servers() -> None:
        # 外层 = 先起来的 server，内层 = 第二个 server；两者重叠期正是双端口的真实形态
        async with factory(app):
            assert calls["bootstrap"] == 1
            assert calls["new"] == 1 and calls["start"] == 1
            async with factory(app):
                assert calls["bootstrap"] == 1, "第二个 server 不该再装配一遍"
                assert calls["token"] == 1, "口令兜底也只该跑一次"
                assert calls["new"] == 1, "只能有一个 IdleDiaryWatcher 实例"
            # 内层退出后外层还在对外服务，巡检不能被停掉
            assert calls["stop"] == 0, "先退出的 server 不该停掉还在服务的那个的巡检"
        assert calls["stop"] == 1, "最后一个 server 退出时才收尾"
        assert main_mod._ASSEMBLY_STATE["servers"] == 0
        assert main_mod._ASSEMBLY_STATE["watcher"] is None

    asyncio.run(_two_servers())