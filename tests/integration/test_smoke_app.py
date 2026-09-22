"""集成冒烟测试：真起 FastAPI 应用，打真实路由。

覆盖四件事：
1. 健康检查 200 且返回全部装配服务键（带 X-Access-Token）；
2. 不带口令访问 /api/* 被门卫挡下（401）；
3. OpenAPI 里注册的 /api/* 路由数量 >= 15；
4. 免鉴权根路径不 401。

**失败语义（R05c）**：本地 import、lifespan 装配、路由层的错误一律**让用例红**，
不再 `except Exception: pytest.skip` 一概跳过——那是把"应用起不来"伪装成
"环境不满足"，启动异常连白屏都算不上（隔离探针实测：注入 RuntimeError 后
5 个用例全体 SKIPPED，测试报告一片绿）。外部服务用明确替身：唯一在 lifespan
里可能触网的是向量集合装配（`ensure_ready`），钉成本地 no-op；LLM/ASR/TTS
的客户端构造不联网，真调用不在冒烟范围。密钥类缺失由 conftest 的假环境变量
兜住，本文件不再需要任何 skip。
"""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

TEST_TOKEN = "test-token"


@pytest.fixture
def client(monkeypatch):
    """起一个真应用（走 lifespan 装配）。任何一步失败都让用例红（R05c）。"""
    # 外部服务替身：ensure_ready 建集合可能触网/等超时，冒烟里钉成 no-op。
    # raising=True：方法改名/删掉时这里立刻红，而不是静默失去隔离。
    import tools.storage as storage

    monkeypatch.setattr(storage.VectorStoreTool, "ensure_ready", lambda self: None,
                        raising=True)

    from main import app  # 本地 import 失败 = 缺陷，不 skip
    with TestClient(app) as c:  # lifespan 装配失败 = 缺陷，不 skip
        yield c


def test_health_ok_with_token(client):
    resp = client.get("/api/system/health", headers={"X-Access-Token": TEST_TOKEN})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["services"], dict)
    # bootstrap() 的装配清单。断言**名字集合**而不是数量：数量对不上只告诉你
    # "10 != 9"，集合对不上会直接指出多了/少了哪一个。
    assert set(body["services"]) == {
        "logger", "kv_store", "vector_store", "tool_registry", "tool_executor",
        "diary_writer", "llm", "asr", "tts", "persona_engine",
    }
    # 每个都装配成功（失败的服务在这里是 False，不是缺键）
    assert body["services"] == {k: True for k in body["services"]}


def test_health_unauthorized_without_token(client):
    resp = client.get("/api/system/health")
    assert resp.status_code == 401


def test_api_routes_registered(client):
    # 注意：枚举路由必须走 openapi()，不能遍历 app.routes——
    # 本工程的 include_router 产出 _IncludedRouter 包装对象，遍历看不到任何 API 路由。
    paths = client.app.openapi().get("paths", {})
    api_paths = [p for p in paths if p.startswith("/api/")]
    assert len(api_paths) >= 15


def test_root_is_public(client):
    # 免鉴权路径：根路径应返回前端壳（200）或兜底重定向，而不该被门卫 401
    resp = client.get("/")
    assert resp.status_code in (200, 307, 308)


def test_relationship_endpoint_exposes_mood(client):
    """GET /api/memory/relationship 返回体必须带 mood（前端「当前情绪」读它）。

    回归守卫：本批曾删掉 mood_baseline、改由 MoodEngine 写 mood，前端一度仍读
    旧字段 → 永远显示默认「平静」。这里锁死 endpoint 真能给出 mood。
    """
    from shared.singletons import services

    kv = services.get("kv_store")
    kv.update(
        "relationship",
        "default",
        lambda d: {**(d or {}), "mood": "心软", "intimacy": 21, "interaction_count": 3},
    )
    resp = client.get(
        "/api/memory/relationship?session_id=default",
        headers={"X-Access-Token": TEST_TOKEN},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "mood" in body
    assert body["mood"] == "心软"
    assert "mood_baseline" not in body


def test_internal_error_returns_safe_500(client, monkeypatch):
    """接口层：程序错误（NameError）→ 500 + internal_error 安全响应，
    不被二次兜底成 200 成功回复（R27b 验收：astream/handle/接口层三层验证）。"""
    from orchestration.pipeline import DialoguePipeline

    def boom(self, state):  # noqa: ARG001
        raise NameError("接口层注入的本地缺陷XYZ")

    monkeypatch.setattr(DialoguePipeline, "_compose", boom)
    resp = client.post(
        "/api/chat/send",
        json={"text": "随便聊聊", "session_id": "r27b-api"},
        headers={"X-Access-Token": TEST_TOKEN},
    )
    assert resp.status_code == 500
    body = resp.json()
    assert body.get("error") == "internal_error"
    # 本地缺陷细节不外泄
    assert "XYZ" not in resp.text


def test_voice_control_failure_sends_fixed_text(client, monkeypatch):
    """WS 换音色控制失败：对外只发固定安全文案，str(exc) 细节不外泄（R27a）。

    走真实 WS 端点检查完整载荷（不只看最终 UI）：先在 realtime 路由上跑通
    一轮（替身实时客户端），再发换音色控制帧触发 set_voice 失败——
    以前 `f"换音色没成功：{exc}"` 会把内部异常细节推给客户端。
    """
    from shared.singletons import services

    services.get("kv_store").write(
        "route_config", "ws-r27a",
        {"route": "realtime", "fail_count": 0, "auto_degrade": True},
    )

    from tools import realtime as rt_mod

    class _FakeRT:
        """替身实时专线客户端：round_trip 正常、set_voice 必炸。"""

        def __init__(self, session_id, instructions=None):
            pass

        async def connect(self):
            pass

        def set_time_hint(self, hint):
            pass

        async def refresh_time_config(self):
            pass

        async def round_trip(self, buffer):
            return {"user_text": "你好", "reply_text": "嗯。",
                    "reply_audio": b"\x00\x00", "emotion": ""}

        async def set_voice(self, voice):
            raise RuntimeError("内部细节SECRET-WS")

        async def close(self):
            pass

    monkeypatch.setattr(rt_mod, "RealtimeDialogClient", _FakeRT)

    texts = []
    # WS 握手走查询串传 token（interaction/api.py 的 _TokenGuard：浏览器 WS
    # 无法带自定义头，这是既有约定）
    with client.websocket_connect(
        "/api/voice/stream?session_id=ws-r27a&token=test-token"
    ) as ws:
        ws.send_bytes(b"\x00\x01" * 160)      # 一段"录音"
        ws.send_text("END")
        ws.send_text('{"voice":"新音色"}')     # 控制帧（文本帧，8.5 平台坑）
        for _ in range(10):
            msg = ws.receive()
            if msg.get("type") in ("websocket.disconnect", None):
                break
            t = msg.get("text")
            if t is None:
                continue                       # 音频二进制帧，跳过
            texts.append(t)
            if "换音色没成功" in t:
                break
    joined = "".join(texts)
    assert "SECRET-WS" not in joined, f"WS 泄漏内部异常细节: {texts!r}"
    assert any("换音色没成功" in t for t in texts), f"应收到固定安全文案，实测 {texts!r}"
