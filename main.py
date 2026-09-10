"""程序入口：装配服务、建 Web 应用、起服务器。

启动：python main.py
起来之后：
    http://127.0.0.1:8000/                     主界面（本机桌面 / Via 类 WebView 浏览器的手机端）
    https://<本机局域网IP>:8443/                主界面（Chrome 等严格浏览器的手机端）
    http://127.0.0.1:8000/ui                  旧聊天页（Gradio 兜底）
    http://127.0.0.1:8000/api/system/health   健康检查
"""

import threading

import uvicorn

from interaction.api import build_app
from shared.singletons import services

# 服务先建好（LLM、语音、存储这些），单个失败不拦启动，用到再报错
services.init_all()

# 闲置太久自动写日记的后台巡检，跟着服务一起起
from capability.proactive import IdleDiaryWatcher

_watcher = IdleDiaryWatcher()
_watcher.start()

app = build_app()


def _serve(config: uvicorn.Config) -> None:
    uvicorn.Server(config).run()


if __name__ == "__main__":
    from tools.certgen import ensure_cert

    cert, key = ensure_cert()
    # HTTP 8000：绑 0.0.0.0——手机也能直接走 http
    # （Via 这类 WebView 浏览器对 http 页面的麦克风放行，且明文 ws:// 不会被
    # 掐半开连接；桌面 Chrome 需要麦克风时请用下面的 https 地址）
    # HTTPS 8443：自签名证书，首次访问点"高级→继续访问"
    if cert:
        https_cfg = uvicorn.Config(
            app, host="0.0.0.0", port=8443,
            ssl_certfile=cert, ssl_keyfile=key, log_level="warning",
        )
        t = threading.Thread(target=_serve, args=(https_cfg,), daemon=True, name="https-8443")
        t.start()
        print("HTTPS 已就绪: https://<本机IP>:8443/  （Chrome 等严格浏览器的手机端用这个）")
        print("HTTP  已就绪: http://<本机IP>:8000/   （Via 等 WebView 浏览器的手机端 + 桌面）")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
