"""交互层 - 实时语音流

VoiceStreamIO 把 WebSocket 包成音频流的收发通道，只管流的进出。
"""

from starlette.websockets import WebSocketState


class VoiceStreamIO:
    """WebSocket 上的音频流收发通道，一个会话一条。"""

    def __init__(self, websocket):
        self._ws = websocket
        self._session_id = ""
        # 最近一条文本帧的内容：控制指令（比如换音色）从这里取
        self.last_text = ""

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

    async def read_chunk(self) -> bytes:
        """收一帧用户语音。客户端发文本就返回空字节，上层当控制信号处理。"""
        message = await self._ws.receive()
        if message.get("text") is not None:
            self.last_text = message.get("text") or ""
            return b""
        return message.get("bytes") or b""

    async def play_chunk(self, audio_chunk: bytes) -> None:
        """给用户放一帧语音（二进制帧直接推下去）。"""
        await self._ws.send_bytes(audio_chunk)

    async def send_text(self, text: str) -> None:
        """推一条文本消息（转写结果、回复文字都走这）。"""
        await self._ws.send_text(text)

    async def close_stream(self) -> None:
        """关闭音频流。"""
        await self._ws.close()
