"""调度层 - 轮次取消原语。

一轮对话可能在生成中途被"更新的同一会话轮次"顶掉（前端打断、或客户端先 cancel 再
发新消息）。这套原语把"哪一轮还活着"记清楚，让上层能协作式地放弃旧轮。

- TurnHandle：单轮的取消句柄。用 threading.Event 而不是 asyncio.Event——管道里会有
  `await asyncio.to_thread(...)` 的同步段，跨线程查询取消状态要安全。
- TurnRegistry：进程内全局登记处，维护"每个会话当前活跃的那一轮"这一不变式。

本轮（C1a）只在管道里 start/finish，并留少量 is_cancelled 检查点；
完整的检查点插桩与 cancelled 写回分支在后续批次补。
"""

import itertools
import threading


class TurnCancelled(Exception):
    """本轮的取消信号：被更新的同一会话轮次取代时抛出。"""


class TurnHandle:
    """单轮的取消句柄。"""

    def __init__(self, session_id: str, turn_id: int):
        self.session_id = session_id
        self.turn_id = turn_id
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        """置位取消信号（幂等）。"""
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        """本轮的取消信号是否已置位。"""
        return self._cancelled.is_set()


class TurnRegistry:
    """进程内全局的活跃轮登记处。

    核心是一条不变式：**每个会话同一时刻最多一个活跃轮**。新轮 start 时会自动取消
    该会话上一个活跃轮——这是消除"客户端先 cancel 再 stream 发新消息"这类时序竞态的关键。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active: dict[str, TurnHandle] = {}
        self._ids = itertools.count(1)

    def start(self, session_id: str) -> TurnHandle:
        """开一轮：先把该会话上一个活跃轮取消掉，再登记并返回新句柄。"""
        with self._lock:
            prev = self._active.get(session_id)
            if prev is not None:
                prev.cancel()
            handle = TurnHandle(session_id=session_id, turn_id=next(self._ids))
            self._active[session_id] = handle
            return handle

    def get(self, session_id: str) -> "TurnHandle | None":
        """取某会话当前活跃轮；没有则 None。"""
        with self._lock:
            return self._active.get(session_id)

    def cancel(self, session_id: str, turn_id: "int | None" = None) -> bool:
        """取消某会话的活跃轮（可选按 turn_id 精确匹配）。

        给"打了又删、不发新消息"这种兜底场景用；指的是它当前登记的那个轮。
        """
        with self._lock:
            handle = self._active.get(session_id)
            if handle is None:
                return False
            if turn_id is not None and handle.turn_id != turn_id:
                return False
            handle.cancel()
            return True

    def finish(self, session_id: str, turn_id: int) -> None:
        """轮次正常结束就登出，避免登记表无限增长。"""
        with self._lock:
            handle = self._active.get(session_id)
            if handle is not None and handle.turn_id == turn_id:
                del self._active[session_id]


# 进程内全局单例：所有管道实例共用同一份登记，取消语义才跨实例一致
TURN_REGISTRY = TurnRegistry()
