"""程序入口：装配服务、建 Web 应用、起服务器。

启动方式：
    python main.py        推荐。证书 + 双端口：HTTP 8000 + HTTPS 8443（手机走 HTTPS）
    uvicorn main:app      只有单端口 HTTP（8000）
注意：`uvicorn main:app` **出不了 8443**——应用对象管不了第二个端口。手机等严格浏览器
      要 HTTPS 才能用麦克风，所以那种场景请用 `python main.py`。

装配时机：import 本模块**不产生副作用**——不建任何服务、不起后台线程。真正的装配
发生在应用 startup 的 lifespan 里。`app` 对象仍在模块顶层建好，所以 `uvicorn main:app`
依然可用（代价是既不生成证书、也不开 8443）。
"""

import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from bootstrap import bootstrap
from interaction.api import build_app

# 视为“仅本机”的监听地址：绑这些地址不会被局域网/公网直接访问，不配口令也安全
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


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
        print("=" * 64)
        print(f"[警告] 当前无访问口令且对外监听（BIND_HOST={settings.bind_host}），")
        print(f"       自动写入 .env 失败：{exc}")
        print("       请手动在 .env 配置 ACCESS_TOKEN，否则局域网内任何人都能访问。")
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
