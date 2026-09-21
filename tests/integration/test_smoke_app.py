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
