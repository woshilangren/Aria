"""持久唯一 ID 生成（R15a/R15d/R17b）。

轮 ID 必须**跨重启不碰撞**：TurnRegistry 的进程内递增数重启后从 1 重数，
与历史 chat_log/回执撞键。uuid4 无时序含义但绝对不撞；持久表与取消门
一律用这里出的 ID。纯标准库，无上层 import。
"""

import uuid


def new_turn_id() -> str:
    """生成跨重启不碰撞的轮 ID（32 位十六进制串）。"""
    return uuid.uuid4().hex
