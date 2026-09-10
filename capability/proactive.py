"""能力层 - 闲置主动关怀

伴侣不能只会被动接话：用户聊完就走、闲了半个多小时，该自己把今天
的对话整理成日记收好。这个模块就是后台巡检员，每隔一会儿看一眼
"谁闲太久了还没写日记"，该写就写，写完不声张。
"""

import threading
from datetime import datetime, timedelta

from config.settings import load_app_config
from shared.singletons import services


class IdleDiaryWatcher:
    """闲置太久自动写日记的后台看门人。

    起一条守护线程，每分钟醒一次点名：哪个会话的最后一条聊天
    超过 idle_minutes 还没给它写过当天的日记，就补一篇。
    """

    # 巡检节奏：一分钟看一眼足够了，日记不赶时间
    _CHECK_INTERVAL = 60

    def __init__(self):
        self._done = set()      # 已经写过的 (session_id, 当天日期)，防止反复重写
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """把巡检线程跑起来，重复调用没副作用。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="idle-diary-watcher"
        )
        self._thread.start()

    def stop(self) -> None:
        """叫停巡检（进程退出时守护线程本来也会跟着走，这接口主要给测试用）。"""
        self._stop.set()

    def _loop(self) -> None:
        # Event.wait 兼当 sleep 和停机开关：stop() 一喊立马醒过来退出
        while not self._stop.wait(self._CHECK_INTERVAL):
            try:
                self._sweep()
            except Exception:
                pass  # 巡检挂一轮无所谓，下一轮接着来

    def _sweep(self) -> None:
        """点名一遍所有会话，闲够久的补写当天日记。"""
        idle_minutes = load_app_config()["proactive"].get("idle_minutes", 30)
        threshold = datetime.now() - timedelta(minutes=idle_minutes)

        for entry in services.get("kv_store").last_chat_per_session():
            last_time = (entry.get("last_time") or "")[:19]
            if not last_time:
                continue
            try:
                last_dt = datetime.fromisoformat(last_time)
            except ValueError:
                continue  # 时间戳格式不对就跳过，别为一条脏数据惊动整个巡检
            if last_dt >= threshold:
                continue  # 还没闲够，下轮再看

            session_id = entry["session_id"]
            # 日记跟着最后聊天的日期走：闲超时往往跨了零点，日期不能想当然用今天
            date = last_time[:10]
            if (session_id, date) in self._done:
                continue  # 这天的日记已经补过了，别反复让模型白写
            # skip_if_exists：进程重启 _done 清零也不怕，在册的日记直接跳过
            diary = services.get("diary_writer").write_diary(session_id, date, skip_if_exists=True)
            if diary:
                self._done.add((session_id, date))
