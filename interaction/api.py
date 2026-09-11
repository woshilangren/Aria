"""交互层 - FastAPI 服务与路由

文字聊天走 /api/chat/send，实时语音走 /api/voice/stream，
语音路由切换走 /api/system/voice-route，Gradio 界面挂到 /ui。
"""

import asyncio
import hmac
import json
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shared.types import InputMessage

# 上传文件大小上限（20MB）；一次从上传流里读多少字节做累加校验。
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024
# 语音缓冲上限：10MB 裸 PCM16 ≈ 16kHz 单声道约 5 分钟音频，够用又能防无界增长。
_MAX_VOICE_BUFFER_BYTES = 10 * 1024 * 1024


def _build_realtime_instructions(session_id: str) -> str:
    """组装实时专线的开场白：人设 + 语音通话规矩 + 关于用户的记忆。

    每次建立连接注入一份，模型在整条连接里都带着这份背景说话。
    记忆和文字聊天共用同一套仓库：档案、画像、关系、长期记忆、日记、最近几轮。
    """
    from capability.memory import MemoryRecaller
    from orchestration.pipeline import KEEPER
    from shared.singletons import services

    parts = []
    try:
        persona = services.get("kv_store").read("persona_config", "")
        story = getattr(persona, "background_story", "") if persona else ""
        if story:
            parts.append(f"【你是谁】\n{story}")
    except Exception:
        pass  # 人设读不到就空着，通话照常

    # 时间感知：通话里她看不见钟，时段分寸（深夜别问吃饭这类）得明说
    from tools.misc import ClockTool

    _clock = ClockTool()
    parts.append(
        f"【当前时间】\n{_clock.now()}（{_clock.period()}）\n{_clock.time_guidance()}".rstrip()
    )

    parts.append(
        "【现在是在语音通话里】\n"
        "你的话会被直接转成语音播出去，所以：\n"
        "1. 全程口语化，句子短，一次别超过三句。\n"
        "2. 不要用任何排版符号、列表、表情，那读出来会很怪。\n"
        "3. 像打电话一样自然接话，别一次说一大段。\n"
        "4. 说话一定要带情绪：开心、激动、生气、温柔都放进语气里，跟着聊天内容起伏。\n"
        "5. 情绪只能放在声音里，绝对不许写进转出来的文字："
        "任何情绪标签、括号补充、舞台指示（比如 [开心]、（压低声音））一个字都不能出现，"
        "用户会直接看到你的转写文字。"
    )

    try:
        bundle = MemoryRecaller().recall("", session_id)
    except Exception:
        bundle = None
    if bundle:
        lines = []
        if bundle.profile:
            pairs = "；".join(f"{k}：{v}" for k, v in bundle.profile.items() if v)
            if pairs:
                lines.append(f"档案：{pairs}")
        tags = (bundle.portrait or {}).get("portrait_tags") or []
        if tags:
            lines.append(f"印象标签：{'、'.join(tags)}")
        rel = bundle.relationship or {}
        if rel.get("stage"):
            lines.append(f"关系阶段：{rel['stage']}（亲密度 {rel.get('intimacy', 0)}）")
        for item in bundle.distilled:
            text = (item.get("content") or "").strip()
            if text:
                lines.append(f"记着的事：{text}")
        for item in bundle.diaries:
            text = (item.get("content") or "").strip()
            if text:
                lines.append(f"日记片段：{text}")
        if lines:
            parts.append("【关于用户的记忆】\n" + "\n".join(lines))

    recent = KEEPER.get_context(session_id)[-6:]
    if recent:
        chat = "\n".join(
            f"{'用户' if m['role'] == 'user' else '你'}：{m['content']}" for m in recent
        )
        parts.append(f"【最近聊过的】\n{chat}")
    return "\n\n".join(parts)


def _realtime_writeback(session_id: str, user_text: str, reply_text: str, emotion: str) -> None:
    """实时专线聊完一轮，按主对话管道的写回规矩落账。

    短期记忆、聊天记录、硬事实沉淀、亲密度一样别落，
    和文字聊天共用同一份短期记忆，两条线才连得起来。
    """
    from capability.memory import ConversationDistiller, RelationshipTracker
    from orchestration.pipeline import KEEPER
    from shared.singletons import services
    from tools.speech import strip_emotion_marks

    if not user_text and not reply_text:
        return
    # 模型就算违规把情绪标记写进了转写文字，落库前也剥干净
    reply_text = strip_emotion_marks(reply_text)
    KEEPER.restore(session_id)
    if user_text:
        KEEPER.append(session_id, "user", user_text)
    if reply_text:
        KEEPER.append(session_id, "assistant", reply_text)

    try:
        kv = services.get("kv_store")
        if user_text:
            kv.write(
                "session", session_id,
                {"role": "user", "text": user_text, "intent": "", "emotion": emotion or "", "mode": "voice"},
            )
        if reply_text:
            kv.write(
                "session", session_id,
                {"role": "assistant", "text": reply_text, "intent": "", "emotion": emotion or "", "mode": "voice"},
            )
    except Exception:
        pass  # 落库失败不打断通话

    try:
        ConversationDistiller().distill_turn(session_id, user_text)
        RelationshipTracker().update(session_id, emotion or "neutral")
    except Exception:
        pass


def _parse_voice_control(text: str) -> dict:
    """解析语音连接上的控制帧，目前只认换音色指令 {"voice": "音色名"}。

    END 和普通文本不是控制指令，返回空字典让调用方按正常语音处理。
    """
    raw = (text or "").strip()
    if not raw or raw == "END" or not raw.startswith("{"):
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    if isinstance(data, dict) and isinstance(data.get("voice"), str) and data["voice"].strip():
        return {"voice": data["voice"].strip()}
    return {}


def _get_orchestrator():
    """对话编排器全局就一份，第一次用到才建。"""
    global _orchestrator
    if _orchestrator is None:
        from orchestration.pipeline import DialoguePipeline

        _orchestrator = DialoguePipeline()
    return _orchestrator


_orchestrator = None


def _get_gateway():
    """网关三件套（收消息、渲染回复、记连接）也是全局一份。"""
    global _gateway
    if _gateway is None:
        from interaction.gateway import ConnectionKeeper, MessageReceiver, OutputRenderer

        _gateway = {
            "receiver": MessageReceiver(),
            "renderer": OutputRenderer(),
            "connections": ConnectionKeeper(),
        }
    return _gateway


_gateway = None


# ---------- 文字聊天路由 ----------
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])


def _upload_to_data_url(upload_path: str) -> str:
    """把 /uploads/ 下的图片读成 base64 data URL，模型不依赖网络就能看。

    模型跑在云端，访问不到本地局域网地址；直接把图片字节内嵌进消息，
    本地测试和云端部署都能正常看图。读不到就返回空，不硬塞失败。
    """
    import base64
    from config.settings import get_settings

    try:
        fname = Path(upload_path).name
        fpath = get_settings().data_dir / "uploads" / fname
        if not fpath.exists():
            return ""
        data = fpath.read_bytes()
        ext = Path(fname).suffix.lower().lstrip(".") or "png"
        ext = "jpeg" if ext == "jpg" else ext
        return f"data:image/{ext};base64," + base64.b64encode(data).decode()
    except Exception:
        return ""


# 用户明确要求 AI"开口说话"时的触发词；命中就在文字回复后额外合成一条语音播出去。
# 明确表示不要语音的不算，避免把拒绝听成"请发语音"。
_VOICE_ASK_WORDS = ("发语音", "用语音", "语音回复", "语音回答", "语音说", "说句话", "说话给我听",
                    "读给我听", "读给我", "读出来", "念给我", "念出来", "音频回复", "用声音", "voice")
_VOICE_REFUSE_WORDS = ("别发语音", "不要语音", "不发语音", "别用语音", "别读给我", "别说话", "别念")


def _wants_voice(text: str) -> bool:
    """用户这句话是不是在点名要 AI 用声音回。"""
    t = (text or "").lower()
    if any(k in t for k in _VOICE_REFUSE_WORDS):
        return False
    return any(k in t for k in _VOICE_ASK_WORDS)


@chat_router.post("/send")
async def send_message(payload: dict) -> dict:
    """接收用户文字消息，返回角色的回复。"""
    text = (payload.get("text") or "").strip()
    if not text:
        # 空消息不进图：白跑一整条管线还让模型对着空气说话
        return {"text": "一句话都不说吗？想聊什么直接说吧。", "output_mode": "text", "image_path": ""}
    gateway = _get_gateway()
    image_url = payload.get("attachment") or payload.get("image_url") or ""
    # 只认 /uploads/ 开头的相对图片地址；模型在云端拉不到本地，直接读文件转 base64 内嵌
    if image_url and (image_url.startswith("/uploads/") and image_url.lower().split("?")[0].endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"))):
        image_url = _upload_to_data_url(image_url)
    else:
        image_url = ""
    message = gateway["receiver"].receive(text, payload.get("session_id", "default"), image_url=image_url)
    reply = await _get_orchestrator().handle(message)
    # render 内部有秒级同步网络调用，扔线程池，别堵住事件循环（否则并发语音通话一起僵死）
    data = await asyncio.to_thread(gateway["renderer"].render, reply, message.session_id)
    # 兜底：AI 这轮没给自己标语音、但用户明确点名要语音时，把简短正文也合一条语音；
    # 已主动带语音就不重复。TTS 失败只是少声音，文字照常回。
    if _wants_voice(text) and not data.get("voice_audio"):
        try:
            audio = await asyncio.to_thread(
                gateway["renderer"].render_voice, data.get("text") or "", message.session_id
            )
            data["voice_audio"] = base64.b64encode(audio).decode()
        except Exception:
            pass
    return data


def _sse_frame(obj: dict) -> str:
    """把一个事件序列化成一帧 SSE（`data: {...}\n\n`）。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@chat_router.post("/stream")
async def stream_message(payload: dict, request: Request):
    """流式对话：句级 SSE，逐帧推 start/emotion/sentence/voice/image/refuse/done/error。

    与 /send 共用同一份管道实现（astream），差别只在把事件逐帧推给前端、
    并在管道内部完成语音合成（synthesize_voice=True）。
    客户端断开会取消该轮，别让后端继续烧 token。
    """
    text = (payload.get("text") or "").strip()
    session_id = payload.get("session_id", "default")

    async def _gen():
        from orchestration.cancellation import TURN_REGISTRY

        # 空消息不跑管道，直接回一句提示（与 /send 一致的兜底话术）
        if not text:
            tip = "一句话都不说吗？想聊什么直接说吧。"
            yield _sse_frame({"type": "start", "turn_id": 0, "session_id": session_id, "text": ""})
            yield _sse_frame({"type": "sentence", "seq": 1, "text": tip})
            yield _sse_frame({"type": "done", "full_text": tip, "output_mode": "text", "image_path": ""})
            return

        gateway = _get_gateway()
        image_url = payload.get("attachment") or payload.get("image_url") or ""
        # 与 /send 同一套图片处理：只认 /uploads/ 相对路径，转 base64 内嵌
        if image_url and image_url.startswith("/uploads/") and image_url.lower().split("?")[0].endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")
        ):
            image_url = _upload_to_data_url(image_url)
        else:
            image_url = ""
        message = gateway["receiver"].receive(text, session_id, image_url=image_url)

        turn_id = None
        try:
            async for ev in _get_orchestrator().astream(
                message, synthesize_voice=True, want_voice=_wants_voice(text)
            ):
                if ev.get("type") == "start":
                    turn_id = ev.get("turn_id")
                if await request.is_disconnected():
                    # 前端走了：取消该轮，后端尽快收工
                    TURN_REGISTRY.cancel(session_id, turn_id)
                    break
                yield _sse_frame(ev)
        except asyncio.CancelledError:
            # 服务端把这个响应生成器取消了，同样把该轮标掉
            TURN_REGISTRY.cancel(session_id, turn_id)
            raise

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(_gen(), media_type="text/event-stream", headers=headers)


@chat_router.post("/cancel")
async def cancel_message(payload: dict) -> dict:
    """取消某会话当前活跃的那一轮对话（后端据此停止生成、且不写记忆）。"""
    from orchestration.cancellation import TURN_REGISTRY

    session_id = payload.get("session_id", "default")
    turn_id = payload.get("turn_id")
    TURN_REGISTRY.cancel(session_id, turn_id)
    return {"ok": True}


@chat_router.get("/history")
async def chat_history(session_id: str = "default", n: int = 100) -> dict:
    """拉取某个会话的对话流水，前端刷新后靠它恢复聊天记录。

    库里存的可能是带 <voice> 标签的原文（实时链路为了合成语音而保留），
    在这里统一剥掉——展示用干净正文，语音内容另存 voice_text 供按需回放，
    免得标签原文露在对话里。
    """
    from interaction.gateway import extract_voice
    from shared.singletons import services

    recs = services.get("kv_store").recent_chat(session_id, n)
    items = []
    for r in recs:
        role = r.get("role")
        text = r.get("text", "") or ""
        entry = {
            "role": role,
            "text": text,
            "mode": r.get("mode", "text"),
            "intent": r.get("intent", ""),
            "emotion": r.get("emotion", ""),
            "time": r.get("time", ""),
        }
        if role == "assistant":
            voice, clean = extract_voice(text)
            entry["text"] = (clean or voice or text).strip()
            entry["voice_text"] = voice
        items.append(entry)
    return {"items": items}


@chat_router.post("/voice-replay")
async def chat_voice_replay(payload: dict) -> dict:
    """历史语音条回放：同一段话只合成一次，之后直接从磁盘缓存读出。"""
    import base64

    from interaction.gateway import get_cached_voice, save_cached_voice

    text = (payload.get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "没有要合成的内容"}
    try:
        audio = get_cached_voice(text)
        if audio is None:
            # 合成是同步网络调用，扔线程池，别堵事件循环
            audio = await asyncio.to_thread(
                _get_gateway()["renderer"].render_voice,
                text,
                payload.get("session_id", "default"),
            )
            save_cached_voice(text, audio)
        return {"ok": True, "voice_audio": base64.b64encode(audio).decode()}
    except Exception as e:  # TTS 失败只影响这一条语音，文字还在
        return {"ok": False, "error": str(e)}


@chat_router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """接收本地文件/图片，落盘到 storage/uploads，返回可访问 URL 供前端展示/回传。"""
    import time
    from config.settings import get_settings

    raw = file.filename or "file"
    safe_ext = Path(raw).suffix.lower()
    # 只放行常见图片和文档后缀，避免传上来奇怪的脚本
    # 特别注意不放行 .svg：SVG 能内嵌 <script>，存成静态文件后被浏览器同源执行会读走
    # localStorage 里的口令（存储型 XSS），所以直接从白名单里掐掉
    allowed = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
               ".pdf", ".txt", ".md", ".doc", ".docx", ".xls", ".xlsx", ".csv"}
    if safe_ext not in allowed:
        return {"ok": False, "error": f"不支持的文件类型：{safe_ext or '（无后缀）'}"}
    # 防路径穿越 / 隐藏文件
    if not raw or raw.startswith(".") or ".." in raw:
        return {"ok": False, "error": "非法文件名"}

    up_dir = get_settings().data_dir / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    # 用时间戳+安全名，避免重名覆盖 & 非法字符
    stamp = time.strftime("%Y%m%d%H%M%S")
    fname = f"{stamp}_{Path(raw).name}"
    dest = up_dir / fname
    # 分块读边写边累加：超限立刻停、删掉半成品文件，别把整个大文件先读进内存再判断（会 OOM）
    total = 0
    oversize = False
    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    oversize = True
                    break
                f.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    if oversize:
        dest.unlink(missing_ok=True)
        return {"ok": False, "error": "文件超过 20MB 上限"}
    return {"ok": True, "url": f"/uploads/{fname}", "name": Path(raw).name}


# ---------- 实时语音路由 ----------
voice_router = APIRouter(prefix="/api/voice", tags=["voice"])


@voice_router.websocket("/stream")
async def voice_stream(websocket: WebSocket) -> None:
    """实时语音双向通道，三条路由共用这个端点。

    客户端把整段录音当二进制帧发上来，发一条文本 "END" 表示说完。
    服务端按当前路由处理：e2e 直接进语音对话引擎，cascade 转文字走
    主对话管道再合成音频，realtime 直连实时专线（连接时注入记忆、每轮写回）。
    文本帧除了 END 还可以发 {"voice": "音色名"}，给实时专线当场换音色。
    回复文字和音频分帧推回去。
    """
    from interaction.voice import VoiceStreamIO
    from orchestration.managers import ROUTE_E2E, ROUTE_REALTIME, VoiceRouteManager
    from tools.speech import strip_emotion_marks

    stream = VoiceStreamIO(websocket)
    manager = VoiceRouteManager()
    # 会话号从连接参数带进来（/api/voice/stream?session_id=xxx），不带就落回 default
    session_id = websocket.query_params.get("session_id") or "default"
    # 前端统一发 16kHz 单声道裸 PCM16（WebAudio 采的），三条路由都按这个格式吃
    audio_fmt, audio_rate = "pcm", 16000
    await stream.open_stream(session_id)
    import time as _time

    _conn_start = _time.time()

    buffer = b""
    realtime_client = None  # 实时专线客户端跟着这条连接走，切走路由就关掉省资源
    try:
        while True:
            chunk = await stream.read_chunk()
            # 空帧是客户端的控制信号，END 表示这段话说完了
            if not chunk:
                # 文本帧里可能藏着控制指令（比如换音色），先拆一下再看
                if stream.last_text:
                    control = _parse_voice_control(stream.last_text)
                    stream.last_text = ""
                    if control:
                        if realtime_client is not None:
                            try:
                                await realtime_client.set_voice(control["voice"])
                                await stream.send_text(f"音色已换成 {control['voice']}")
                            except Exception as exc:
                                await stream.send_text(f"换音色没成功：{exc}")
                        else:
                            await stream.send_text("当前不在实时专线路由上，换音色指令没生效")
                        continue

                if not buffer:
                    continue
                route = manager.current_route(session_id)
                # 切走实时专线就关连接；切回来时会重建，顺便把最新记忆重新注入一遍
                if route != ROUTE_REALTIME and realtime_client is not None:
                    await realtime_client.close()
                    realtime_client = None
                try:
                    if route == ROUTE_E2E:
                        from capability.perception import PerceptionPipeline
                        from capability.skills import SpeechDialogEngine
                        from orchestration.pipeline import KEEPER

                        # ASR+LLM+TTS 是几十秒的同步网络调用，必须扔线程池，
                        # 不然语音通话期间整个事件循环（所有请求）一起卡死
                        out = await asyncio.to_thread(
                            SpeechDialogEngine().handle, buffer, session_id, audio_fmt, audio_rate
                        )
                        user_text, reply_text, audio = (
                            out["user_text"],
                            out["reply_text"],
                            out["reply_audio"],
                        )
                        print(f"[voice] e2e 识别: {user_text[:40]} | 回复: {reply_text[:40]} | 音频 {len(audio)} bytes")
                        # e2e 也得留痕：感知一次拿情绪标签，写回交给共用的落账函数
                        emotion = ""
                        try:
                            pipeline = PerceptionPipeline()
                            _intent, emo = await asyncio.to_thread(
                                pipeline.run, user_text, KEEPER.get_context(session_id)[-4:]
                            )
                            emotion = emo.emotion
                        except Exception:
                            pass
                        await asyncio.to_thread(
                            _realtime_writeback, session_id, user_text, reply_text, emotion
                        )
                    elif route == ROUTE_REALTIME:
                        from tools.realtime import RealtimeDialogClient
                        from tools.misc import ClockTool

                        if realtime_client is None:
                            realtime_client = RealtimeDialogClient(
                                session_id,
                                instructions=_build_realtime_instructions(session_id),
                            )
                            await realtime_client.connect()
                        # 每轮刷新时段提示：通话跨时段（深夜别问吃饭）时她才反应得过来
                        _c = ClockTool()
                        realtime_client.set_time_hint(
                            f"【当前时间】现在是{_c.now()}（{_c.period()}）。{_c.time_guidance()}".rstrip()
                        )
                        await realtime_client.refresh_time_config()
                        out = await realtime_client.round_trip(buffer)
                        user_text, reply_text = out["user_text"], out["reply_text"]
                        audio = out.get("reply_audio") or b""
                        print(f"[voice] realtime 识别: {user_text[:40]} | 回复: {reply_text[:40]} | 音频 {len(audio)} bytes")
                        # 写回里有向量库/SQLite 网络与磁盘动作，同样别堵事件循环
                        await asyncio.to_thread(
                            _realtime_writeback, session_id, user_text, reply_text, out.get("emotion", "")
                        )
                        if user_text:
                            await stream.send_text(f"我听到的是：{user_text}")
                    else:
                        gateway = _get_gateway()
                        message = await asyncio.to_thread(
                            gateway["receiver"].receive_voice, buffer, session_id, audio_fmt, audio_rate
                        )
                        print(f"[voice] cascade 识别OK: {message.text[:40]}")
                        await stream.send_text(f"我听到的是：{message.text}")
                        reply = await _get_orchestrator().handle(message)
                        print(f"[voice] cascade 回复OK: {reply.text[:40]}")
                        audio = b""
                        try:
                            # 合成用原文：TTS 会把开头的情绪标记拆出来转成指令
                            audio = await asyncio.to_thread(
                                gateway["renderer"].render_voice, reply.text, session_id
                            )
                            print(f"[voice] cascade TTSOK: {len(audio)} bytes")
                        except Exception:
                            audio = b""
                        # 标签只给合成器用：推给前端的文字剥干净，不展示标记
                        reply_text = strip_emotion_marks(reply.text)
                except Exception:
                    import traceback

                    print("[voice] 这一轮处理失败：", traceback.format_exc(limit=3))
                    from orchestration.managers import FallbackController

                    manager.report_failure(session_id)
                    reply_text = FallbackController().fallback_reply("voice_route")
                    audio = b""
                    # 实时专线这轮砸了，连接可能已经脏了，关掉下轮重建
                    if realtime_client is not None:
                        await realtime_client.close()
                        realtime_client = None

                # 回复前客户端可能已经断开（切后台/浏览器杀连接）：
                # 硬发会炸出 RuntimeError 还刷日志，软处理丢弃结果即可
                try:
                    await stream.send_text(reply_text)
                    if audio:
                        await stream.play_chunk(audio)
                except Exception:
                    print(f"[voice] 回复没送出去：客户端断开了（连接持续 {_time.time() - _conn_start:.0f}s）")
                buffer = b""
            else:
                buffer += chunk
                # 客户端一直发二进制帧却从不发 END，buffer 会无界增长到 OOM：
                # 到上限就清空并提醒用户，异常咽掉别炸连接
                if len(buffer) > _MAX_VOICE_BUFFER_BYTES:
                    buffer = b""
                    try:
                        await stream.send_text("语音太长了，我先截断了，你分几次说吧")
                    except Exception:
                        pass
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        # 客户端断开后循环再 receive 会炸这个（手机切后台/省电杀连接最常见），
        # 打一行人话就收摊，别让 ASGI 异常刷屏
        print(f"[voice] 连接断开：{exc}（连接持续 {_time.time() - _conn_start:.0f}s）")
    finally:
        print(f"[voice] 连接关闭（持续 {_time.time() - _conn_start:.0f}s）")
        # close() 可能抛异常，各自独立包裹：绝不能让它把后面的连接簿清理一起跳过，
        # 否则 ConnectionKeeper 里会残留死会话
        if realtime_client is not None:
            try:
                await realtime_client.close()
            except Exception:
                pass
        try:
            _get_gateway()["connections"].on_disconnect(session_id)
        except Exception:
            pass


# ---------- 系统路由 ----------
system_router = APIRouter(prefix="/api/system", tags=["system"])


@system_router.get("/health")
async def health() -> dict:
    """健康检查，顺带报一下各服务活没活着。"""
    from shared.singletons import services

    status = services.check_status()
    return {"status": "ok", "services": status}


# ---------- 头像管理：读写 storage/avatar.json ----------
_AVATAR_FILE = Path(__file__).resolve().parent.parent / "storage" / "avatar.json"


class _AvatarUpdate(BaseModel):
    avatar_url: str


def _read_avatar(default: str = "") -> dict:
    try:
        if _AVATAR_FILE.exists():
            return json.loads(_AVATAR_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"avatar_url": default}


@system_router.get("/avatar")
async def get_avatar() -> dict:
    """读取当前角色的头像 URL。没设置过就返回空。"""
    return _read_avatar()


@system_router.put("/avatar")
async def set_avatar(payload: _AvatarUpdate) -> dict:
    """设置角色的头像 URL。URL 会存进 storage/avatar.json，下次启动也记得。"""
    data = {"avatar_url": (payload.avatar_url or "").strip()}
    _AVATAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    _AVATAR_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "avatar_url": data["avatar_url"]}


@system_router.post("/voice-route")
async def switch_voice_route(payload: dict) -> dict:
    """切换实时语音路由（e2e 端到端 / cascade 级联 / realtime 实时专线）。"""
    from orchestration.managers import VoiceRouteManager

    route = VoiceRouteManager().switch(
        payload.get("session_id", "default"), payload.get("route", "cascade")
    )
    return {"route": route}


@system_router.get("/voice-route")
async def get_voice_route(session_id: str = "default") -> dict:
    """查当前语音路由，前端想知道现在走的哪条线就问它。"""
    from orchestration.managers import VoiceRouteManager

    return {"route": VoiceRouteManager().current_route(session_id)}


@system_router.get("/config")
async def get_runtime_config() -> dict:
    """读 LLM 运行时参数：思考开关、强度（temperature / max_tokens）。"""
    from config.settings import load_app_config

    llm = load_app_config()["llm"]
    return {
        "thinking": llm.get("enable_thinking", False),
        "temperature": llm.get("temperature", 0.7),
        "max_tokens": llm.get("max_tokens", 1024),
    }


class _RuntimeConfig(BaseModel):
    thinking: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@system_router.post("/config")
async def set_runtime_config(payload: _RuntimeConfig) -> dict:
    """写 LLM 运行时参数，追加存进 config.json，下次对话立即生效。"""
    from config.settings import load_app_config

    cfg = load_app_config()
    llm = cfg.setdefault("llm", {})
    if payload.thinking is not None:
        llm["enable_thinking"] = payload.thinking
    if payload.temperature is not None:
        llm["temperature"] = max(0.0, min(2.0, payload.temperature))
    if payload.max_tokens is not None:
        llm["max_tokens"] = max(64, int(payload.max_tokens))
    # 落盘到项目根 config.json
    import json as _json
    from config.settings import PROJECT_ROOT

    path = PROJECT_ROOT / "config.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(cfg, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return {"ok": True, **llm}


# ---------- 记忆管理 API ----------
memory_router = APIRouter(prefix="/api/memory", tags=["memory"])


class _ProfileUpdate(BaseModel):
    field: str
    value: str | list | None


@memory_router.get("/profile")
async def _get_profile(session_id: str = "default"):
    from shared.singletons import services
    return services.get("kv_store").read("profile", session_id) or {}


@memory_router.put("/profile")
async def _update_profile(update: _ProfileUpdate, session_id: str = "default"):
    from capability.memory import ProfileUpdater
    ok = ProfileUpdater().set_field(session_id, update.field, update.value)
    return {"ok": ok}


@memory_router.get("/portrait")
async def _get_portrait(session_id: str = "default"):
    from shared.singletons import services
    return services.get("kv_store").read("portrait", session_id) or {}


@memory_router.get("/relationship")
async def _get_relationship(session_id: str = "default"):
    from shared.singletons import services
    return services.get("kv_store").read("relationship", session_id) or {}


@memory_router.get("/diaries")
async def _get_diaries():
    from shared.singletons import services

    return services.get("vector_store").list_diaries_enriched()


@memory_router.delete("/diaries/{diary_id}")
async def _delete_diary(diary_id: str):
    from shared.singletons import services

    return {"ok": services.get("vector_store").delete_diary(diary_id)}


@memory_router.get("/memories")
async def _get_memories(q: str = "", session_id: str = "default"):
    from shared.singletons import services

    vs = services.get("vector_store")
    if q:
        return vs.search_memory(q, top_k=10)
    return vs.list_memories(limit=20)


@memory_router.delete("/memories/{memory_id}")
async def _delete_memory(memory_id: str):
    from shared.singletons import services

    return {"ok": services.get("vector_store").delete_memory(memory_id)}


# ---------- 应用工厂 ----------
class _TokenGuard:
    """访问口令门卫：.env 配了 ACCESS_TOKEN 就全站鉴权（HTTP 查头/查参，WS 查握手参数）。

    公网部署时记忆、档案、API key 全暴露在请求路径上，没这道门等于裸奔。
    口令留空（默认）则完全放行——本机/内网使用不想输口令就不配。
    """

    def __init__(self, app):
        self.app = app

    # 静态资源路径免鉴权（头像/PWA图标/CA证书等，浏览器加载时不可能带自定义 header）。
    # 根路径 "/" 也放行：要返回前端壳 HTML，PWA standalone 启动才能加载出 JS，
    # 之后 JS 再弹口令框登录；数据/语音/记忆等接口仍强制鉴权。
    # 注意 /uploads/ 不在免鉴权之列：上传文件可能含私人内容，必须凭口令访问
    # （前端给 <img>/<a> 拼 ?token=，见 index.html 的 _withToken）。
    _PUBLIC_PATHS = ("/static/", "/ca.crt", "/manifest.json", "/sw.js")

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if path.startswith(self._PUBLIC_PATHS) or path in ("/", "/index.html"):
            await self.app(scope, receive, send)
            return

        from config.settings import get_settings

        token = get_settings().access_token
        if not token or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        got = ""
        for k, v in scope.get("headers") or []:
            if k == b"x-access-token":
                got = v.decode("utf-8", "ignore")
                break
        # 查询串 token 只放行两类：WebSocket 握手（沿用既有约定，前端 WS 走查询串传 token），
        # 以及 /uploads/（<img>/<a> 加载时无法带自定义请求头，只能靠查参）。
        # 其它路径（含所有 /api/*）只认 X-Access-Token 头，免得口令进访问日志和 Referer。
        if not got and (scope["type"] == "websocket" or path.startswith("/uploads/")):
            qs = scope.get("query_string", b"").decode("utf-8", "ignore")
            for part in qs.split("&"):
                if part.startswith("token="):
                    got = part[6:]
                    break

        # 常量时间比较，别用 == 逐字符比（会泄露口令长度/前缀）；两边先 encode，
        # 非 ASCII 口令也不会因为 str/bytes 混比抛异常
        if hmac.compare_digest(got.encode("utf-8"), token.encode("utf-8")):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            # 握手阶段直接拒掉，别让无口令的人建立语音通道
            await send({"type": "websocket.close", "code": 1008})
        else:
            from fastapi.responses import JSONResponse

            resp = JSONResponse({"detail": "需要访问口令"}, status_code=401)
            await resp(scope, receive, send)


class _NosniffStaticFiles(StaticFiles):
    """给静态文件的每个响应补 X-Content-Type-Options: nosniff。

    即便上传目录里混入了可被当作脚本解析的文件，浏览器也不会凭内容嗅探去执行它。
    StaticFiles 不方便按目录统一加头，覆盖 file_response 是最小的改法。
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        return resp


def build_app() -> FastAPI:
    """构建完整的 Web 应用：路由 + Gradio 界面。"""
    from config.settings import get_settings
    app = FastAPI(title="Aria - AI Companion", version="1.0.0")
    # 公网部署必配的访问口令门卫（.env 的 ACCESS_TOKEN，留空则不鉴权）
    app.add_middleware(_TokenGuard)

    app.include_router(system_router)
    app.include_router(chat_router)
    app.include_router(voice_router)
    app.include_router(memory_router)

    # 新前端文件：frontend/index.html（和 main.py 同级）
    _frontend_file = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    _frontend_dir = _frontend_file.parent

    if _frontend_file.exists():
        # 挂载静态资源（图片/css 等）
        app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="frontend")
        # 上传文件目录，前端/回传通过 /uploads/xxx 访问
        _up_dir = get_settings().data_dir / "uploads"
        _up_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/uploads", _NosniffStaticFiles(directory=str(_up_dir)), name="uploads")
        # 根路径 → 新前端。no-store：前端改版后手机浏览器不许再拿缓存的旧页面，
        # 旧版采集的音频格式后端已经不认了，页面落后一个版本语音就是"没反应"
        @app.get("/", include_in_schema=False)
        async def _root_to_new_ui():
            return FileResponse(str(_frontend_file), headers={"Cache-Control": "no-store"})

        # PWA：manifest + service worker 必须挂在根路径（scope 才覆盖整站）
        @app.get("/manifest.json", include_in_schema=False)
        async def _manifest():
            return FileResponse(
                _frontend_dir / "manifest.json",
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/sw.js", include_in_schema=False)
        async def _sw():
            return FileResponse(
                _frontend_dir / "sw.js",
                media_type="application/javascript",
                headers={"Cache-Control": "no-store"},
            )

        # 手机安装本地 CA 用的（装一次，Chrome 对这套 https 就完全信任）
        @app.get("/ca.crt", include_in_schema=False)
        async def _ca():
            from tools.certgen import ca_cert_path

            return FileResponse(
                ca_cert_path(),
                media_type="application/x-x509-ca-cert",
                filename="rin-rootCA.crt",
            )
    else:
        # 没有新前端就用旧的路由兜底
        @app.get("/", include_in_schema=False)
        def _root_to_ui():
            return RedirectResponse(url="/ui")

    # Gradio 装不上就跳过，纯 API 也能跑
    try:
        import gradio as gr

        from interaction.ui import build_gradio_app

        app = gr.mount_gradio_app(app, build_gradio_app(), path="/ui")
    except ImportError:
        pass

    return app
