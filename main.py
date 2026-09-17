"""程序入口：装配服务、建 Web 应用、起服务器。

启动方式：
    python main.py        推荐。证书 + 双端口：HTTP 8000 + HTTPS 8443（手机走 HTTPS）
    uvicorn main:app      只有单端口 HTTP（8000）
注意：`uvicorn main:app` **出不了 8443**——应用对象管不了第二个端口。手机等严格浏览器
      要 HTTPS 才能用麦克风，所以那种场景请用 `python main.py`。

装配时机：import 本模块**不产生副作用**——不建任何服务、不起后台线程。真正的装配
发生在应用 startup 的 lifespan 里。`app` 对象仍在模块顶层建好，所以 `uvicorn main:app`
依然可用（代价是既不生成证书、也不开 8443）。

P3-4 说明（codex 2026-09-17 审查指出）：`/api/system/health` 是**故意无需 token**——
它是网关监控 / 运维探针的入口；要 ACCESS_TOKEN 才能访问会让"全裸"状态没法被
监控看到（自打监控脸的意图）。这是有意的设计选择，代价是 enumeration 信标风险
（拿到 health 的人能知道 `bind_host` + `_UNPROTECTED_STATE`）。运维侧缓解：
把 `/api/system/health` 放在反向代理白名单内 / 用网络层 ACL 限制访问源。
"""

import os
import secrets
import threading
import time as _time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from bootstrap import bootstrap
from config.settings import get_settings
from interaction.api import build_app

# 视为“仅本机”的监听地址：绑这些地址不会被局域网/公网直接访问，不配口令也安全
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

# K10 监控位：当前是否处于"无口令对外监听"的危险状态。
# 设了之后 /api/system/health 会持续暴露给监控/告警系统。
# 配套每 10 分钟在日志里喊一次，直到有人配 ACCESS_TOKEN（或 BIND_HOST 改回 loopback）。
_UNPROTECTED_STATE = {"active": False, "since": "", "reason": "", "bind_host": ""}
_UNPROTECTED_LOCK = threading.Lock()
_alarm_started = False


def _start_unprotected_alarm() -> None:
    """无口令对外监听状态被设置后，启动 daemon 线程每 10 分钟在控制台吼一次。

    设计取舍：监控/告警系统更可靠的来源是 /api/system/health 的 _UNPROTECTED_STATE，
    控制台告警是兜底——开发机/单人自用没接监控时不会让告警沉到底。
    线程级不重复启动：模块级 _alarm_started 标志防多线程并发重复。
    """
    global _alarm_started
    if _alarm_started:
        return
    _alarm_started = True

    def _loop():
        while True:
            _time.sleep(600)  # 10 分钟
            with _UNPROTECTED_LOCK:
                st = dict(_UNPROTECTED_STATE)
            if not st.get("active"):
                return
            # P1-2 自停（codex 2026-09-17 审查指出）：原版"持续告警"实际是"永久告警"——
            # 没人改 active=False，daemon 一直吼到进程退出。
            # 自停策略：每轮重新读 .env，发现 ACCESS_TOKEN 已配上 → 自动降级。
            try:
                get_settings.cache_clear()
                if get_settings().access_token:
                    with _UNPROTECTED_LOCK:
                        _UNPROTECTED_STATE["active"] = False
                    print("[K10] ACCESS_TOKEN 已配上，告警自动停止。")
                    return
            except Exception:
                pass
            print("=" * 64)
            print(f"[持续警告·K10] 站点仍以无 ACCESS_TOKEN 对 {st.get('bind_host', '?')} 监听中。")
            print(f"               自 {st.get('since', '?')} 起，原因：{st.get('reason', '?')}")
            print("               现在任何 LAN 内设备都能访问所有接口与语音通道。")
            print("=" * 64)

    threading.Thread(target=_loop, daemon=True, name="unprotected-alarm").start()



def create_app() -> FastAPI:
    """建应用对象（挂中间件 / 路由 / 静态目录）并接上 lifespan。

    便宜、无副作用：不初始化任何服务，服务的装配交给 lifespan（startup 时执行）。
    """
    app = build_app()
    # build_app 里若装了 gradio，它会把自己的 lifespan 包在 app.router.lifespan_context 上；
    # 这里同样“包一层”，绝不能直接覆盖，否则会把 gradio 的 startup 事件丢掉（/ui 就废了）
    old_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # startup：口令兜底（F5）→ 装配所有服务 → 起闲置巡检线程。
        # 口令兜底原来只在 run() 里做——`uvicorn main:app` 绕过它，BIND_HOST=0.0.0.0
        # 且没配口令时全站（记忆/档案/聊天）不鉴权对外。挪进 lifespan 后任何
        # 启动方式都先过这道防线。
        from config.settings import get_settings

        _ensure_access_token(get_settings())
        bootstrap()
        from capability.proactive import IdleDiaryWatcher

        watcher = IdleDiaryWatcher()
        watcher.start()
        try:
            async with old_lifespan(app) as state:
                yield state
        finally:
            watcher.stop()  # shutdown：叫停巡检线程

    app.router.lifespan_context = lifespan
    return app


# app 在模块顶层建好：`uvicorn main:app` 仍然能用（只是没有证书和 8443）
app = create_app()


def _serve(config: uvicorn.Config) -> None:
    uvicorn.Server(config).run()


def _upsert_env_token(env_path: Path, token: str) -> None:
    """把 ACCESS_TOKEN 写进 .env：已有该行就替换那一行，没有就追加。

    只动 ACCESS_TOKEN 这一行，其余行（真实 API key 都在里面）原样保留、绝不改写。
    """
    lines = []
    replaced = False
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("ACCESS_TOKEN="):
                lines.append(f"ACCESS_TOKEN={token}")
                replaced = True
            else:
                lines.append(line)
    if not replaced:
        lines.append(f"ACCESS_TOKEN={token}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_access_token(settings) -> None:
    """无口令且对外监听时，自动生成随机口令并落 .env，避免整个站点“裸奔”。

    为什么自动生成而不是拒绝启动：前端 index.html 已内置
    “401 → 弹框输入口令 → 存 localStorage → 刷新重试”的完整链路，自动生成后
    手机首次访问弹一次框、把控制台打印的口令粘进去即可，体验是通的；
    直接拒绝启动会让人以为程序坏了。
    写 .env 失败则只强烈提示、不中断启动（也不假装已经鉴权）。
    """
    if settings.access_token:
        return
    if settings.bind_host in _LOOPBACK_HOSTS:
        return  # 只听本机，不配口令也安全

    new_token = secrets.token_urlsafe(24)
    try:
        from config.settings import PROJECT_ROOT

        _upsert_env_token(PROJECT_ROOT / ".env", new_token)
    except Exception as exc:
        with _UNPROTECTED_LOCK:
            _UNPROTECTED_STATE["active"] = True
            _UNPROTECTED_STATE["since"] = datetime.now(timezone.utc).isoformat()
            _UNPROTECTED_STATE["reason"] = f"写入 .env 失败：{exc}"
            _UNPROTECTED_STATE["bind_host"] = settings.bind_host
        _start_unprotected_alarm()
        print("=" * 64)
        print(f"[警告] 当前无访问口令且对外监听（BIND_HOST={settings.bind_host}），")
        print(f"       自动写入 .env 失败：{exc}")
        print("       请手动在 .env 配置 ACCESS_TOKEN，否则局域网内任何人都能访问。")
        print("       已注册 /api/system/health 持续暴露此状态；监控可 GET 检测。")
        print("=" * 64)
        return

    # 让“本次运行”立刻生效：写回环境变量并清掉 Settings 缓存，
    # 否则已被 lru_cache 缓存的旧实例仍读到空口令，等于没设防
    os.environ["ACCESS_TOKEN"] = new_token
    from config.settings import get_settings

    get_settings.cache_clear()
    print("=" * 64)
    print(f"[提示] 当前无口令且对外监听（BIND_HOST={settings.bind_host}），")
    print("       已自动生成随机访问口令并写入 .env 的 ACCESS_TOKEN：")
    print(f"           {new_token}")
    print("       手机 / 浏览器首次访问会弹框，把上面的口令粘进去即可。")
    print("=" * 64)


def run() -> None:
    """标准启动路径（`python main.py` 与 run_local.py 都走这里）：口令兜底 → 证书 → 双端口。"""
    from config.settings import get_settings
    from tools.certgen import ensure_cert

    _ensure_access_token(get_settings())
    settings = get_settings()  # 口令可能刚生成，重新取一份拿到最新状态

    cert, key = ensure_cert()
    # HTTP 8000：默认绑 0.0.0.0——手机也能直接走 http（要只听本机就把 BIND_HOST 改成 127.0.0.1）
    # （Via 这类 WebView 浏览器对 http 页面的麦克风放行，且明文 ws:// 不会被
    # 掐半开连接；桌面 Chrome 需要麦克风时请用下面的 https 地址）
    # HTTPS 8443：自签名证书，首次访问点"高级→继续访问"
    if cert:
        https_cfg = uvicorn.Config(
            app, host=settings.bind_host, port=8443,
            ssl_certfile=cert, ssl_keyfile=key, log_level="warning",
        )
        threading.Thread(
            target=_serve, args=(https_cfg,), daemon=True, name="https-8443"
        ).start()
        print("HTTPS 已就绪: https://<本机IP>:8443/  （Chrome 等严格浏览器的手机端用这个）")
        print("HTTP  已就绪: http://<本机IP>:8000/   （Via 等 WebView 浏览器的手机端 + 桌面）")
        print("（双端口/手机 HTTPS 只有 `python main.py` 有；`uvicorn main:app` 只有单端口 HTTP）")
    else:
        print("HTTP  已就绪: http://<本机IP>:8000/（未生成证书，仅单端口 HTTP）")
    uvicorn.run(app, host=settings.bind_host, port=8000)


if __name__ == "__main__":
    run()
