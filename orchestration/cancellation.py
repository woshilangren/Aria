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

from shared.types import CommitReceipt


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
        """取消某会话的活跃轮（按 turn_id 精确匹配）。

        turn_id=None 一律拒绝（F4）：竞态下"start 事件还没到、turnId 还是 null"
        的取消请求会无条件顶掉当前登记的轮——而那时登记的可能已经是**新一轮**
        （用户连发时新轮 start 先到、旧 cancel 后到），误杀后新轮永远没有回复。
        前端拿不到 turnId 就不该发 cancel；新轮 start 自带"顶掉旧轮"语义，
        兜底天然存在，这里的拒绝不会留下没人管的旧轮。
        """
        with self._lock:
            if turn_id is None:
                return False
            handle = self._active.get(session_id)
            if handle is None or handle.turn_id != turn_id:
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


class CommitGate:
    """取消与提交的**同一原子门**（R15d，8.11.3）。

    取消（cancel_turn）与提交（run_commit）在同一把锁下裁决：
    - **取消先赢** → 门内置 cancelled 标记，随后的 run_commit 直接返回
      cancelled（数据事务根本不开始）；
    - **提交先成功** → 迟到的取消返回 already_committed（带原回执），
      已提交历史与派生任务资格不受影响；
    - **处理中** → cancel_turn 与 run_commit 在同一把锁上排队，天然
      "等待确切结果"——不存在提前声称 cancelled/committed 的假答案；
    - **提交失败** → 失败回执不进回执表：此后的取消返回 cancelled
      （不保留 COMMITTED 状态）。

    纪律：run_commit 传入的 fn 是**数据层事务**（无网络/无 LLM）——门锁
    会横跨整个 fn，fn 里混进慢调用就会把取消请求也堵死。调用方把
    run_commit 放 to_thread（不堵事件循环）。锁顺序固定：只拿门锁，
    fn 内部的 SQLite 锁在门锁之内获取、之内释放，无反向获取路径。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cancelled: set = set()
        self._receipts: dict = {}

    def cancel_turn(self, turn_id: str, reason: str = "") -> dict:
        """按精确 turn_id 取消。返回 {status: cancelled | already_committed}。"""
        with self._lock:
            if turn_id in self._receipts:
                rc = self._receipts[turn_id]
                # 迟到取消：提交已成功——只停输出，历史与派生资格不动
                return {"status": "already_committed",
                        "receipt": rc,
                        "reason": reason or "late_cancel"}
            self._cancelled.add(turn_id)
            return {"status": "cancelled", "reason": reason or ""}

    def run_commit(self, turn_id: str, fn) -> dict:
        """在门内执行提交：先裁决取消，未取消才跑数据事务并登记回执。

        fn：无参可调用，内部完成数据层事务（kv.commit_turn），返回回执
        dict（status=committed/failed）。失败回执**不登记**——提交失败
        不保留 COMMITTED 状态，此后的取消仍返回 cancelled。
        """
        with self._lock:
            if turn_id in self._cancelled:
                return {"status": "cancelled", "reason": "cancel_won"}
            receipt = fn()
            if receipt.get("status") == "committed":
                self._receipts[turn_id] = receipt
            return receipt

    def get_receipt(self, turn_id: str):
        """查已登记的回执；没有返回 None。"""
        with self._lock:
            return self._receipts.get(turn_id)

    def is_committed(self, turn_id: str) -> bool:
        """该轮是否已提交。派生任务以此判断资格——新轮启动对旧轮 handle
        的取消**不许**波及已提交轮的派生任务（它们看门，不看 handle）。"""
        with self._lock:
            return turn_id in self._receipts


# 进程内全局唯一：取消与提交的裁决必须走同一扇门，两扇门就没有原子性可言
COMMIT_GATE = CommitGate()


def make_receipt(**kwargs) -> CommitReceipt:
    """把数据层返回的回执 dict 规范成 CommitReceipt（缺字段按类型默认补）。"""
    return CommitReceipt(**{k: v for k, v in kwargs.items()})
