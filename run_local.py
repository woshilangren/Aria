"""本地运行入口（局域网可访问）——和 `python main.py` 完全等价。

保留这个文件只为兼容旧的启动习惯：直接把 main.run() 借过来，证书生成、
双端口、口令自动生成一样不少，两个入口行为彻底一致、不再分叉。
"""

from main import run

if __name__ == "__main__":
    run()
