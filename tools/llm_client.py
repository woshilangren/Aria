"""LLM 调用统一走这个文件，全项目不许自己另外建 OpenAI 客户端。

好处就一个：换模型、换地址、加重试，改这里一处就够了。
主模型（当前 claude-sonnet-5，走 nonelinear 的 OpenAI 兼容端点）连不上或
连续报错时，自动切备用（glm-5.3-flash）。参数按模型家族分发（见 _is_qwen）。
"""

import json
import threading
import time
from typing import AsyncIterator

from openai import AsyncOpenAI, OpenAI

from config.settings import get_settings, load_app_config


class LLMClient:
    """对话补全、带工具的补全，都从这走。主模型挂了备用顶上。"""

    def __init__(self):
        cfg = get_settings()
        if not cfg.llm_api_key:
            raise RuntimeError("LLM_API_KEY 没配，聊天功能没法用")
        # 主客户端：模型/地址/key 全来自 .env（当前 claude-sonnet-5 + nonelinear）
        self._client = OpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            timeout=60,
        )
        self._model = cfg.llm_model

        # 备用客户端：GLM 只在主模型降级时顶上；没配 key 就置 None，不挡主模型干活
        if cfg.llm_fallback_api_key:
            self._fallback = OpenAI(
                api_key=cfg.llm_fallback_api_key,
                base_url=cfg.llm_fallback_base_url,
                timeout=90,
            )
            self._fallback_model = cfg.llm_fallback_model
        else:
            self._fallback = None
            self._fallback_model = ""

        # 异步客户端：只给流式用（C1a 先加接口，尚未接进管道）。同步客户端一律保留——
        # 记忆摘要 / 画像 / 感知 / 日记那几处还在用同步的 chat()。
        self._aclient = AsyncOpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            timeout=60,
        )
        if cfg.llm_fallback_api_key:
            self._afallback = AsyncOpenAI(
                api_key=cfg.llm_fallback_api_key,
                base_url=cfg.llm_fallback_base_url,
                timeout=90,
            )
        else:
            self._afallback = None

        # 降级状态：主模型连续失败达到 max_errors 次就打开熔断（记下打开时刻），
        # 冷却期内彻底不碰主模型，省得每次白等 60 秒超时；冷却到期半开放行一次
        fb_cfg = load_app_config()["llm"].get("fallback", {})
        self._max_errors = fb_cfg.get("max_errors", 2)
        self._cooldown = float(fb_cfg.get("cooldown_seconds", 120) or 120)
        # 半开探测专用短超时：探测不再复用客户端默认的 60 秒，避免聊天节奏下每条都白等
        self._probe_timeout = float(fb_cfg.get("probe_timeout_seconds", 5) or 5)
        self._error_count = 0
        self._opened_at: float | None = None  # 熔断打开时刻（单调时钟），None=未打开
        # 半开探测连续失败次数：每失败一次探测超时翻倍（封顶 60 秒）。
        # 否则主模型首包常态超过探测短超时（开思考/高峰期）时，探测永远失败、
        # 熔断永不闭合——"主模型其实活着，体验却永远停在小模型"且无人察觉。
        self._probe_fail_streak = 0
        # 熔断状态被线程池（chat）、事件循环（astream）、巡检线程（日记 LLM）三方
        # 无锁共享时，"判断 circuit_open → 更新计数"是复合操作，交错读写会互相误判。
        # 一把小锁只保护状态字段，不包网络调用。
        self._cb_lock = threading.Lock()

    def _probe_timeout_now(self) -> float:
        """本次半开探测用的超时：连续失败就逐次翻倍，封顶 60 秒（客户端默认值）。"""
        return min(60.0, self._probe_timeout * (2 ** self._probe_fail_streak))

    def _is_qwen(self) -> bool:
        """主模型是不是 qwen 系——qwen 专有参数只对它发。"""
        return "qwen" in (self._model or "").lower()

    # ---------- 内部：统一的主备切换，切换逻辑只写这一遍 ----------
    def _complete(self, **kwargs) -> object:
        """所有请求的必经之路。kwargs 是发给 chat.completions.create 的参数。

        返回原始响应对象，取 content 还是 tool_calls 由调用方自己决定。
        """
        last_error = None

        # 熔断是否仍处于打开状态：打开且未到冷却 → 直接走备用，连主模型都不碰
        # （这是修掉“冷却期里还白等主模型 60 秒超时”的关键）
        with self._cb_lock:
            circuit_open = (
                self._opened_at is not None
                and (time.monotonic() - self._opened_at) < self._cooldown
            )
            is_probe = (not circuit_open) and self._opened_at is not None

        if not circuit_open:
            # 这次是不是"半开探测"：熔断原本开着（_opened_at 非空）且已过冷却，才会走到这里。
            # 探测复用主客户端，但默认 60 秒超时对人聊天的节奏太长（几乎每条都白等），
            # 所以只给探测传短超时；正常调用绝不传，保持客户端默认 60 秒。
            # Qwen3 系非流式调用必须关思考模式，不然官方直接拒绝请求；
            # Claude 系（走 OpenAI 兼容代理）则相反：enable_thinking 是 qwen 专有
            # 参数，temperature 在 Sonnet 5 上已移除（发原生物理会 400）——
            # 两个都不带，交给模型默认（Claude 自带 adaptive thinking）
            if self._is_qwen():
                thinking = load_app_config()["llm"].get("enable_thinking", False)
                kwargs.setdefault("extra_body", {"enable_thinking": thinking})
            else:
                kwargs.pop("extra_body", None)
                kwargs.pop("temperature", None)
            if is_probe:
                kwargs["timeout"] = self._probe_timeout_now()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                # 成功即闭合：清空错误计数、清掉熔断时刻、探测超时回到最短档
                with self._cb_lock:
                    self._error_count = 0
                    self._opened_at = None
                    self._probe_fail_streak = 0
                return resp
            except Exception as exc:
                with self._cb_lock:
                    self._error_count += 1
                    last_error = exc
                    if is_probe:
                        self._probe_fail_streak += 1
                    # 静默降级会让"主模型 key 失效"这种事藏好几个星期没人发现，至少喊一声
                    print(f"[llm] 主模型调用失败({self._error_count}/{self._max_errors})，走备用：{exc}")
                    if self._error_count >= self._max_errors:
                        # 连续失败到阈值：打开熔断；半开再次失败时也会走到这，等于刷新打开时刻
                        self._opened_at = time.monotonic()

        # 备用顶上：GLM-5 系是"始终思考"模型，不吃 enable_thinking，摘掉换
        # thinking_effort=low——实测能把每条回复的推理开销砍半（500→250 上下），
        # 不然聊天每条都要先默默想十几秒才开口，用户只会觉得"怎么这么慢"
        if self._fallback is not None:
            kwargs.pop("extra_body", None)
            kwargs.pop("timeout", None)  # 短超时只给半开探测用，备用走它自己的默认超时
            kwargs["extra_body"] = {"thinking_effort": "low"}
            kwargs["model"] = self._fallback_model
            try:
                resp = self._fallback.chat.completions.create(**kwargs)
                # 备用每成功一次就把主模型错误计数衰减一格，而不是只等主模型自己成功清零——
                # 否则相隔几小时的两次网络抖动各自被兜住、计数却累到阈值，白开一轮 120 秒熔断，
                # 期间感知/画像/日记全换小模型，用户感知为"她突然变了个人"
                with self._cb_lock:
                    self._error_count = max(0, self._error_count - 1)
                return resp
            except Exception as exc:
                raise RuntimeError(f"主模型和备用模型都挂了：主={last_error}，备={exc}") from exc

        raise RuntimeError(f"LLM 调用失败，也没配备用模型：{last_error}")

    # ---------- 对外：两个能力，签名永远不许变 ----------
    def chat(self, messages: list, temperature: float = None, max_tokens: int = None) -> str:
        """普通对话补全，返回助手回复的文本。主模型挂了自动换备用顶上。"""
        cfg = load_app_config()["llm"]
        if temperature is None:
            temperature = cfg["temperature"]
        if max_tokens is None:
            max_tokens = cfg["max_tokens"]

        resp = self._complete(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_with_tools(self, messages: list, tools_catalog: list) -> dict:
        """带工具列表的一轮补全。

        返回两种结果之一：
        - {"type": "tool_call", "name": ..., "arguments": {...}, "call_id": ...}
        - {"type": "final", "content": "..."}
        只跑一轮，循环轮数由上层控制。
        """
        cfg = load_app_config()["llm"]
        resp = self._complete(
            model=self._model,
            messages=messages,
            tools=tools_catalog,
            tool_choice="auto",
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        msg = resp.choices[0].message
        if getattr(msg, "tool_calls", None):
            call = msg.tool_calls[0]
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            return {
                "type": "tool_call",
                "name": call.function.name,
                "arguments": args,
                "call_id": call.id,
            }
        return {"type": "final", "content": msg.content or ""}

    # ---------- 异步流式（C1a 只提供能力，尚未接入管道）----------
    async def astream_chat(
        self, messages: list, temperature: float = None,
        max_tokens: int = None, enable_thinking: bool = None,
    ) -> AsyncIterator[str]:
        """异步流式补全：逐块 yield 文本增量。

        与同步 chat() 共用同一套熔断/降级状态（_opened_at / _error_count / _cooldown /
        _probe_timeout），不另起炉灶。

        enable_thinking：本轮是否让模型认真想。None = 读 config 的全局开关；
        True/False = 按本轮输入复杂度逐轮覆盖（管道判定）。

        流式降级的硬约束：**只能在还没吐出任何 token 之前换备用**——一旦已经 yield 过
        内容，再换备用也是接不上的半截话，所以那种情况直接抛，由上层收尾。
        """
        cfg = load_app_config()["llm"]
        if temperature is None:
            temperature = cfg["temperature"]
        if max_tokens is None:
            max_tokens = cfg["max_tokens"]

        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        last_error = None

        # 熔断打开且未到冷却 → 直接走备用（与同步 _complete 同一判断、同一把锁）
        with self._cb_lock:
            circuit_open = (
                self._opened_at is not None
                and (time.monotonic() - self._opened_at) < self._cooldown
            )
            is_probe = (not circuit_open) and self._opened_at is not None
        if not circuit_open:
            # 半开探测：只在这时传短超时（连续失败逐次翻倍，见 _probe_timeout_now）
            # qwen 专有参数只对 qwen 发（见 _complete 里的同款处理）
            if self._is_qwen():
                if enable_thinking is None:
                    thinking = load_app_config()["llm"].get("enable_thinking", False)
                else:
                    thinking = enable_thinking
                kwargs.setdefault("extra_body", {"enable_thinking": thinking})
            else:
                kwargs.pop("extra_body", None)
                kwargs.pop("temperature", None)
            if is_probe:
                kwargs["timeout"] = self._probe_timeout_now()
            stream = None
            try:
                stream = await self._aclient.chat.completions.create(**kwargs)
            except Exception as exc:
                with self._cb_lock:
                    self._error_count += 1
                    last_error = exc
                    if is_probe:
                        self._probe_fail_streak += 1
                    print(f"[llm] 主模型流式建流失败({self._error_count}/{self._max_errors})，走备用：{exc}")
                    if self._error_count >= self._max_errors:
                        self._opened_at = time.monotonic()
            if stream is not None:
                # 建流成功：正常逐块产出。中途出错：吐过内容就直接抛；没吐过则回落去试备用
                emitted = False
                try:
                    async for chunk in stream:
                        delta = self._delta_of(chunk)
                        if delta:
                            emitted = True
                            yield delta
                    with self._cb_lock:
                        self._error_count = 0
                        self._opened_at = None
                        self._probe_fail_streak = 0
                    return
                except Exception as exc:
                    with self._cb_lock:
                        self._error_count += 1
                        last_error = exc
                        if self._error_count >= self._max_errors:
                            self._opened_at = time.monotonic()
                    if emitted:
                        raise  # 已经吐过 token，换备用也接不上，交给上层收尾
                    # 一个 token 都没吐：落到下面走备用

        # 备用顶上：GLM 系始终思考，换成 thinking_effort=low（同同步路径）
        if self._afallback is not None:
            kwargs.pop("extra_body", None)
            kwargs.pop("timeout", None)  # 短超时只给半开探测用，备用走它自己的默认超时
            kwargs["extra_body"] = {"thinking_effort": "low"}
            kwargs["model"] = self._fallback_model
            try:
                stream = await self._afallback.chat.completions.create(**kwargs)
            except Exception as exc:
                raise RuntimeError(f"主模型和备用模型都挂了（流式）：主={last_error}，备={exc}") from exc
            async for chunk in stream:
                delta = self._delta_of(chunk)
                if delta:
                    yield delta
            with self._cb_lock:
                self._error_count = max(0, self._error_count - 1)  # 备用成功，错误计数衰减（同同步路径）
            return

        raise RuntimeError(f"LLM 流式调用失败，也没配备用模型：{last_error}")

    @staticmethod
    def _delta_of(chunk) -> str:
        """从流式分块里取文本增量；取不到（无 choices / 非文本增量）返回空串。"""
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        return getattr(delta, "content", None) or ""
