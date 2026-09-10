"""本地运行入口（局域网可访问）

启动方式：
  python run_local.py

然后本机打开 http://localhost:8000
同局域网设备用 http://你的IP:8000 访问
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.singletons import services
services.init_all()

from capability.proactive import IdleDiaryWatcher
_watcher = IdleDiaryWatcher()
_watcher.start()

from interaction.api import build_app
app = build_app()

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("Aria · 本地运行")
    print("新前端：http://localhost:8000")
    print("旧Gradio：http://localhost:8000/ui")
    print("健康检查：http://localhost:8000/api/system/health")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)