"""程序入口：装配服务、建 Web 应用、起服务器。

启动方式：
    python main.py        推荐。证书 + 双端口：HTTP 8000 + HTTPS 8443（手机走 HTTPS）
    uvicorn main:app      只有单端口 HTTP（8000）
注意：`uvicorn main:app` **出不了 8443**——应用对象管不了第二个端口。手机等严格浏览器
      要 HTTPS 才能用麦克风，所以那种场景请用 `python main.py`。

装配时机：import 本模块**不产生副作用**——不建任何服务、不起后台线程。真正的装配
发生在应用 startup 的 lifespan 里。`app` 对象仍在模块顶层建好，所以 `uvicorn main:app`
依然可用（代价是既不生成证书、也不开 8443）。

关于 `/api/system/health` 的鉴权（口径修正 2026-09-17）：它**走全站鉴权**，不带口令是 401
（`tests/integration/test_smoke_app.py::test_health_unauthorized_without_token` 把这条锁死了）。
早先一版注释写它"故意无需 token、监控可 GET 检测无口令状态"，与代码和测试都不符，已改。
故意不把它加进 `_PUBLIC_PATHS`：配了口令之后它还免鉴权，等于给任何 LAN 对端一个枚举信标
（能读到九个服务的存活状态）。"无口令对外监听"的告警只有控制台一条通道，见 _UNPROTECTED_STATE。
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
# **唯一出口是下面的控制台告警线程**（每 10 分钟喊一次，直到配上 ACCESS_TOKEN 或
# BIND_HOST 改回 loopback）。
# ⚠ 别再把"/api/system/health 会持续暴露此状态"这句话写回注释——它已经假过两次了：
#   health 只返回 services.check_status()，从来没报过这个状态。真要接上去得先把这个
#   dict 挪进 shared/singletons.py（interaction 直接读 main 的模块状态是向上 import，
#   违反分层规则），而单人自用没有监控消费者，不值得为它破分层。
_UNPROTECTED_STATE = {"active": False, "since": "", "reason": "", "bind_host": ""}
_UNPROTECTED_LOCK = threading.Lock()
_alarm_started = False


def _start_unprotected_alarm() -> None:
    """无口令对外监听状态被设置后，启动 daemon 线程每 10 分钟在控制台吼一次。

    这是该状态**唯一**的出口（health 端点不报它，理由见 _UNPROTECTED_STATE 注释）。
    开发机/单人自用没接监控，所以告警必须落到人一定看得见的地方——控制台。
    线程级不重复启动：模块级 _alarm_started 标志防多线程并发重复
    （双端口下 lifespan 会跑两遍，见 _ASSEMBLY_STATE 注释）。
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



# I10：双端口 = 两个 uvicorn.Server 跑**同一个 app 对象** → lifespan 会被各执行一次。
# 没有进程级守卫时的三个后果（都是实测/代码级确认的，不是理论风险）：
#   ① bootstrap() 装配两遍——服务全部重复构造、services registry 被二次覆盖；
#   ② IdleDiaryWatcher 起两个实例——防重入的 _done 集合是**实例级**的，跨实例无法去重
#      → 同一天同一会话可能写两篇日记、主动消息双发；
#   ③ _ensure_access_token 被两个线程并发调用 → 可能各生成一个口令互相覆盖 .env。
# 解法是"一次装配 + 引用计数"：第一个进来的 server 装配，最后一个退出的收尾。
# 用引用计数而不是"只在主线程装配"，是因为 `uvicorn main:app` 单端口路径也走这里，
# 而且两个 server 的启动先后顺序不保证（HTTPS 是 daemon 线程）。
_ASSEMBLY_LOCK = threading.Lock()
_ASSEMBLY_STATE = {"servers": 0, "watcher": None}


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
        # 但双端口下 lifespan 会跑两遍，所以装配整体只挂在"第一个 server"上（I10）。
        with _ASSEMBLY_LOCK:
            first = _ASSEMBLY_STATE["servers"] == 0
            _ASSEMBLY_STATE["servers"] += 1
        if first:
            _ensure_access_token(get_settings())
            bootstrap()
            from capability.proactive import IdleDiaryWatcher

            watcher = IdleDiaryWatcher()
            watcher.start()
            with _ASSEMBLY_LOCK:
                _ASSEMBLY_STATE["watcher"] = watcher
        try:
            # gradio 的 lifespan 仍按每个 server 各进一次——ASGI lifespan 协议是
            # 每连接一次的，这里不能也跟着"只跑一遍"，否则第二个 server 起不来。
            async with old_lifespan(app) as state:
                yield state
        finally:
            # 最后一个 server 退出才停巡检线程。HTTPS 那条是 daemon 线程，Ctrl-C 时
            # 主线程的 server 先收尾——所以判"归零"而不是无条件 stop，
            # 否则先退出的那个会把还在对外服务的另一个的巡检停掉。
            with _ASSEMBLY_LOCK:
                _ASSEMBLY_STATE["servers"] = max(0, _ASSEMBLY_STATE["servers"] - 1)
                last = _ASSEMBLY_STATE["servers"] == 0
                watcher = _ASSEMBLY_STATE["watcher"] if last else None
                if last:
                    _ASSEMBLY_STATE["watcher"] = None
            if watcher is not None:
                watcher.stop()

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


def _effective_bind_host(settings) -> str:
    """实际会绑到哪个地址——**优先看命令行，其次才看 .env**（F4）。

    为什么不能只看 `settings.bind_host`：`uvicorn main:app --host 0.0.0.0` 时，
    真正的 socket 是 0.0.0.0，而 `.env` 里写着 `BIND_HOST=127.0.0.1` 也照样被读到。
    于是"只听本机所以不用口令"的判定成立、口令不生成，**而站点其实对整个 LAN 裸奔**
    （记忆/档案/聊天全可读）。这就是文档里说的 token-fallback gap 的残留变体。

    取不到命令行参数时回落到 env，行为与改动前一致。
    判不准的情况（比如通过别的 WSGI/ASGI 容器起）一律按"非回环"处理——
    宁可多生成一个口令（前端有 401 弹框链路，体验是通的），也不要静默裸奔。
    """
    import sys

    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg in ("--host", "-h") and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--host="):
            return arg.split("=", 1)[1]
    return settings.bind_host


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
    if _effective_bind_host(settings) in _LOOPBACK_HOSTS:
        return  # 确认只听本机，不配口令也安全

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
        print("       本进程会每 10 分钟在控制台再喊一次，直到你配上口令。")
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
    if settings.bind_host not in _LOOPBACK_HOSTS:
        # F5a：**不加 HSTS**。自签证书 + HSTS 是个陷阱——证书重生/换机器之后，
        # 浏览器会在 max-age 内拒绝访问且**没有"继续前往"的绕过入口**，
        # 本地部署会被自己锁在门外。真正需要说清的是下面这条：
        print("!" * 64)
        print(f"[提示] 正在对 {settings.bind_host} 监听（非仅本机）。")
        print("       走 HTTP 8000 时 X-Access-Token 头和 ?token= 全程明文，")
        print("       同一局域网内被动嗅探一次即可拿到口令，进而读走全部记忆。")
        print("       要手机用请走 HTTPS 8443；要出门在外用建议 Tailscale 而不是暴露端口。")
        print("!" * 64)
    if cert:
        https_cfg = uvicorn.Config(
            app, host=settings.bind_host, port=8443,
            ssl_certfile=cert, ssl_keyfile=key, log_level="warning",
            # F3：uvicorn 的 access_log 记的是 get_path_with_query_string——
            # 每次 <img src="/uploads/x.png?token=..."> 加载和 WS 握手都会把口令
            # 打进控制台。api.py 里正是为"免得口令进访问日志"才把查参 token 限制在
            # WS 与 /uploads/，这个豁免把自己拆了。作者日常把日志贴给 AI 排障，风险不低。
            access_log=False,
        )
        threading.Thread(
            target=_serve, args=(https_cfg,), daemon=True, name="https-8443"
        ).start()
        print("HTTPS 已就绪: https://<本机IP>:8443/  （Chrome 等严格浏览器的手机端用这个）")
        print("HTTP  已就绪: http://<本机IP>:8000/   （Via 等 WebView 浏览器的手机端 + 桌面）")
        print("（双端口/手机 HTTPS 只有 `python main.py` 有；`uvicorn main:app` 只有单端口 HTTP）")
    else:
        print("HTTP  已就绪: http://<本机IP>:8000/（未生成证书，仅单端口 HTTP）")
    uvicorn.run(app, host=settings.bind_host, port=8000, access_log=False)  # F3，理由同上


if __name__ == "__main__":
    run()
