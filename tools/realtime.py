"""工具层 - Realtime 实时语音客户端

一条 WebSocket 直连百炼 Qwen-Omni-Realtime 多模态模型：
用户语音进（16k PCM），回复的语音（24k PCM）和文字出来，
识别、对话、合成由模型一线完成，不用 ASR、TTS 两段接力。

协议要点（Manual 模式，按段说话）：
- 连上后先发 session.update：定音色、定开场白（记忆从这里注入）、关自动断句
- 每轮：input_audio_buffer.append（Base64 音频）→ commit → response.create
- 收：conversation.item.input_audio_transcription.completed（用户说了啥）
     response.audio_transcript.delta（回复文字）、response.audio.delta（回复音频）
     response.done（这轮说完）
- 自定义音色预留：set_voice 运行中随时换，下轮回复就用新声音
"""

import asyncio
import base64
import json

import websockets

from tools.speech import pcm_to_wav


class RealtimeDialogClient:
    """百炼 Realtime 模型的对话客户端，一个语音会话一条连接。

    连接里模型自己记得本轮通话说过什么（模型侧上下文），
    项目记忆系统（开场白注入 + 每轮写回）由交互层负责，这里只管协议。
    """

    # 一轮回复的等待上限：模型要边生成边合成音频，长回复给足时间
    _RESPONSE_TIMEOUT = 120.0
    # 服务端确认开场配置的等待上限
    _CONFIG_TIMEOUT = 10.0

    def __init__(self, session_id: str, voice: str = "", instructions: str = ""):
        from config.settings import get_settings

        cfg = get_settings()
        self._session_id = session_id
        self._api_key = cfg.realtime_api_key or cfg.tts_api_key or cfg.llm_api_key
        self._model = cfg.realtime_model
        self._voice = voice or cfg.realtime_voice
        self._instructions = instructions
        # 时段分寸提示：通话会跨时段（晚上 11 点打到凌晨），每轮由交互层刷新，
        # 变了才重发 session.update，不多花一次往返
        self._time_hint = ""
        self._sent_time_hint = None

        # 业务空间 ID 配了就走专属域名（更快），没配走通用域名（照样能用）
        workspace = cfg.realtime_workspace_id.strip()
        host = f"{workspace}.cn-beijing.maas.aliyuncs.com" if workspace else "dashscope.aliyuncs.com"
        self._url = f"wss://{host}/api-ws/v1/realtime?model={self._model}"

        self._ws = None            # websockets 连接
        self._recv_task = None     # 收事件的后台任务
        self._configured = asyncio.Event()  # 服务端确认 session.update
        self._round_done = asyncio.Event()  # 当前这轮 response.done
        self._round_error = ""
        # 最近一个真正跑通过对话的音色：服务端有"先确认后断连"的坏毛病，
        # 换了坏音色得有个退路，回滚到它保证通话不断
        self._last_good_voice = self._voice
        # 每轮收集的东西
        self._user_text = ""
        self._reply_text = ""
        self._reply_audio = bytearray()
        self._emotion = ""

    # ---------- 对外接口 ----------

    async def connect(self) -> None:
        """连上服务端并发开场配置。音色名写错、key不对这类问题在这里当场暴露。"""
        if self._ws is not None:
            return
        if not self._api_key or self._api_key.startswith("your-"):
            raise RuntimeError("Realtime 没配 API Key（.env 里填百炼 key，或留空复用 LLM_API_KEY）")

        self._ws = await websockets.connect(
            self._url,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            max_size=16 * 1024 * 1024,
        )
        self._configured = asyncio.Event()
        # 上一轮对话留下的报错得清掉，不然会把好连接误杀成配置失败
        self._round_error = ""
        self._recv_task = asyncio.create_task(self._recv_loop())
        await self._send({"type": "session.update", "session": self._session_config()})

        done, _ = await asyncio.wait(
            [asyncio.create_task(self._configured.wait())], timeout=self._CONFIG_TIMEOUT
        )
        if not done or self._round_error:
            message = self._round_error or "服务端没确认开场配置"
            await self.close()
            raise RuntimeError(f"Realtime 会话配置失败：{message}")

    def update_instructions(self, instructions: str) -> None:
        """换开场白（记忆注入内容）。已连接时只影响下次重连，不拆当前上下文。"""
        self._instructions = instructions

    def set_time_hint(self, hint: str) -> None:
        """记录本轮的时段提示，refresh_time_config 时才真正下发。"""
        self._time_hint = hint or ""

    async def refresh_time_config(self) -> None:
        """时段提示变了就重发 session.update（配置其余部分原样），没变不动。

        通话经常跨时段：11 点开始打到 12 点，"可以问吃晚饭"就得变成
        "催他睡觉"。重发不影响模型侧已攒的对话上下文。
        """
        if self._ws is None or self._time_hint == self._sent_time_hint:
            return
        self._sent_time_hint = self._time_hint
        await self._send({"type": "session.update", "session": self._session_config()})

    async def set_voice(self, voice: str) -> None:
        """自定义音色预留接口：运行中换音色，下轮回复生效。

        服务端点头（session.updated）才算换成功；被拒绝（error）就退回
        原音色并报错，不让连接带着坏配置走进下一轮。
        """
        voice = (voice or "").strip()
        if not voice or voice == self._voice:
            return
        old_voice = self._voice
        self._voice = voice
        if self._ws is None:
            return  # 还没连上：先记着，下次 connect 一起配
        self._round_error = ""
        self._configured = asyncio.Event()
        try:
            await self._send({"type": "session.update", "session": self._session_config()})
            done, _ = await asyncio.wait(
                [asyncio.create_task(self._configured.wait())], timeout=self._CONFIG_TIMEOUT
            )
        except Exception:
            done = set()
        if not done or self._round_error:
            self._voice = old_voice
            raise RuntimeError(
                f"服务端不认这个音色（{self._round_error or '没等到确认'}）"
            )
        # 服务端对坏音色是"先回确认、紧接着悄悄断连"：等半拍看连接还活着没
        await asyncio.sleep(0.5)
        if self._ws is None:
            self._voice = old_voice
            raise RuntimeError("服务端不认这个音色（确认后立刻断开了连接）")

    async def round_trip(self, audio: bytes) -> dict:
        """送进一整段用户录音，等模型说完，拿回这轮结果。

        返回 {"user_text", "reply_text", "reply_audio", "emotion"}。
        连接断了会自动重连重试一次，还不行才把异常抛给上层降级。
        音色被服务端拒过（重连配置失败）就退回最近跑通的音色再试。
        """
        for attempt in (1, 2):
            try:
                if self._ws is None:
                    try:
                        await self.connect()
                    except Exception:
                        if self._voice == self._last_good_voice:
                            raise
                        self._voice = self._last_good_voice
                        await self.connect()
                return await self._do_round(audio)
            except Exception:
                await self.close()
                if attempt == 2:
                    raise
                # 第一次没跑通：音色不是跑通过的那个就退回去，别带着坏配置再炸一次
                self._voice = self._last_good_voice
        return {}

    async def close(self) -> None:
        """收摊：停收事件任务、断连接。断开前服务端自己结账上下文。"""
        if self._recv_task is not None:
            self._recv_task.cancel()
            self._recv_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ---------- 内部实现 ----------

    def _session_config(self) -> dict:
        """session.update 的内容：音色 + 开场白（含时段提示）+ 关自动断句（Manual 模式）。"""
        instructions = self._instructions
        if self._time_hint:
            instructions = f"{instructions}\n\n{self._time_hint}" if instructions else self._time_hint
        return {
            "modalities": ["text", "audio"],
            "voice": self._voice,
            "instructions": instructions,
            # turn_detection 置空 = Manual 模式：客户端攒完整段再提交，节奏自己控制
            "turn_detection": None,
        }

    async def _do_round(self, audio: bytes) -> dict:
        """一轮完整对话：append → commit → response.create → 等说完。"""
        self._user_text = ""
        self._reply_text = ""
        self._reply_audio = bytearray()
        self._emotion = ""
        self._round_error = ""
        self._round_done = asyncio.Event()

        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio).decode("ascii"),
            }
        )
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})

        await asyncio.wait_for(self._round_done.wait(), timeout=self._RESPONSE_TIMEOUT)
        if self._round_error:
            raise RuntimeError(f"Realtime 服务端报错：{self._round_error}")
        # 这轮真跑通了，当前音色记成"可用音色"，之后重连失败就拿它兜底
        self._last_good_voice = self._voice
        # 服务端回的是 24kHz 裸 PCM，浏览器播不了：包上 WAV 头再下发。
        # 没收到音频就给空字节，别让一个 44 字节的空壳 WAV 走完播放流程
        pcm = bytes(self._reply_audio)
        reply_audio = pcm_to_wav(pcm, sample_rate=24000) if pcm else b""
        return {
            "user_text": self._user_text,
            "reply_text": self._reply_text,
            "reply_audio": reply_audio,
            "emotion": self._emotion or "neutral",
        }

    async def _recv_loop(self) -> None:
        """后台收事件，按类型分拣进当前这轮的收集槽。"""
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                self._dispatch(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass  # 连接断开交给收尾处理，别让异常把任务炸了
        finally:
            self._ws = None
            # 正在等回复的话，别让人干等超时，直接报连接断开
            if not self._round_done.is_set():
                self._round_error = self._round_error or "与模型的连接断开了"
                self._round_done.set()
            self._configured.set()

    def _dispatch(self, event: dict) -> None:
        etype = event.get("type", "")
        if etype == "session.updated":
            self._configured.set()
        elif etype == "conversation.item.input_audio_transcription.completed":
            # 服务端内置识别（qwen3-asr-flash-realtime）给出的用户原话，记忆写回靠它
            self._user_text = (event.get("transcript") or "").strip()
        elif etype == "conversation.item.input_audio_transcription.delta":
            # 顺路白拿的情绪标签（happy/sad/angry...），喂给好感度账本
            if event.get("emotion"):
                self._emotion = event["emotion"]
        elif etype == "response.audio_transcript.delta":
            self._reply_text += event.get("delta") or event.get("text") or ""
        elif etype == "response.audio.delta":
            b64 = event.get("delta") or ""
            if b64:
                self._reply_audio += base64.b64decode(b64)
        elif etype == "response.done":
            self._round_done.set()
        elif etype == "error":
            err = event.get("error") or {}
            parts = [str(err.get("code", "")), str(err.get("message", ""))]
            self._round_error = "：".join(p for p in parts if p) or "未知错误"
            self._round_done.set()
            self._configured.set()

    async def _send(self, payload: dict) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime 连接还没建立")
        await self._ws.send(json.dumps(payload, ensure_ascii=False))
