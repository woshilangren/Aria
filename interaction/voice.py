"""交互层 - 实时语音流

VoiceStreamIO 把 WebSocket 包成音频流的收发通道，只管流的进出。
VoiceReceiver 是这条连接上**唯一的接收协程**（R16a）：帧的分类、字节
预算、断连观察都在这里收口，消费方（voice_stream 主循环）只管取事件。
"""

import asyncio

from starlette.websockets import WebSocketState


class VoiceStreamIO:
    """WebSocket 上的音频流收发通道，一个会话一条。"""

    def __init__(self, websocket):
        self._ws = websocket
        self._session_id = ""

    @property
    def alive(self) -> bool:
        """连接是否还活着（F10）：cascade 生成期间轮询它，挂断即取消该轮。"""
        try:
            return self._ws.client_state == WebSocketState.CONNECTED
        except Exception:
            return False

    async def open_stream(self, session_id: str) -> None:
        """接受 WebSocket 连接，标记这条流属于哪个会话。"""
        await self._ws.accept()
        self._session_id = session_id

    async def raw_receive(self) -> dict:
        """原始 receive。R16a：底层收包只许 VoiceReceiver 一个协程调用——
        以前主循环边收边生成，生成期间 receive 停摆，断连和上传都看不见。"""
        return await self._ws.receive()

    async def play_chunk(self, audio_chunk: bytes) -> None:
        """给用户放一帧语音（二进制帧直接推下去）。"""
        await self._ws.send_bytes(audio_chunk)

    async def send_text(self, text: str) -> None:
        """推一条文本消息（转写结果、回复文字都走这）。"""
        await self._ws.send_text(text)

    async def close_stream(self) -> None:
        """关闭音频流。"""
        await self._ws.close()


class VoiceReceiver:
    """每条语音 WS 只有一个 receive 协程（R16a），帧在这里分发成三类事件：

    - ("audio", bytes)：音频块。已收未消费的累计字节有硬上限（原 10 MB
      防线）——生成卡住时客户端连续上传也不突破：超限**明确丢弃**（丢弃量
      累计进 dropped_bytes，消费方择机告知用户），绝不无限排队；
    - ("text", str)：文本帧（END / ping / 控制指令），语义与旧循环一致；
    - ("disconnect", None)：连接断了。生成中的轮次由此**立刻**被看见，
      不再等生成完才收摊。

    队列满与预算超限都走"丢弃"而不是阻塞——接收协程一停，断连就观察
    不到，那是本类存在的意义。END/ping 保持文本帧（前端协议不变）。
    """

    def __init__(self, stream: VoiceStreamIO, max_pending_bytes: int):
        self._stream = stream
        self._max = max_pending_bytes
        self._pending = 0  # 已收未消费的音频字节
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._disconnected = asyncio.Event()
        self.dropped_bytes = 0  # 预算/队列超限被丢弃的音频字节（消费方观察点）
        self._task: asyncio.Task = None

    @property
    def disconnected(self) -> asyncio.Event:
        return self._disconnected

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                message = await self._stream.raw_receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    self._offer(("text", message.get("text") or ""))
                else:
                    chunk = message.get("bytes") or b""
                    if not chunk:
                        continue
                    if self._pending + len(chunk) > self._max:
                        self.dropped_bytes += len(chunk)
                        continue
                    self._pending += len(chunk)
                    if not self._offer(("audio", chunk)):
                        self._pending -= len(chunk)
                        self.dropped_bytes += len(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            # 断连在 starlette 里可能是 disconnect 消息，也可能让 receive 直接
            # 抛（WebSocketDisconnect / RuntimeError，手机切后台最常见）——
            # 殊途同归：标记断连并唤醒消费方，这行 except 就是为此存在
            pass
        self._disconnected.set()
        self._offer(("disconnect", None))  # best-effort：Event 兜底，不依赖入队成功

    def _offer(self, item) -> bool:
        """非阻塞入队：满了就丢（音频丢预算账，文本/断连有 Event 兜底）。"""
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            if item[0] == "audio":
                self.dropped_bytes += len(item[1])
            return False

    async def get(self):
        """取下一事件（FIFO）。断连后不再收新事件；断连**之前**已入队的
        照常消费完，排空才返回 ("disconnect", None)——生成路径用 disconnected
        Event 在跑生成前拦截，这里只保证顺序可预期。"""
        get_task = asyncio.ensure_future(self._queue.get())
        disc_task = asyncio.ensure_future(self._disconnected.wait())
        try:
            done, _pending = await asyncio.wait(
                {get_task, disc_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if disc_task in done and get_task not in done:
                return ("disconnect", None)
            item = get_task.result()
            if item[0] == "audio":
                self._pending -= len(item[1])
            return item
        finally:
            for t in (get_task, disc_task):
                if not t.done():
                    t.cancel()

    def stop(self) -> None:
        """收摊：停接收协程、清空队列与字节账（取消生成及后台接收的完整清理）。"""
        if self._task is not None:
            self._task.cancel()
            self._task = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._pending = 0
