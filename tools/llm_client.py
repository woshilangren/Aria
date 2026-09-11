"""LLM 调用统一走这个文件，全项目不许自己另外建 OpenAI 客户端。

好处就一个：换模型、换地址、加重试，改这里一处就够了。
主模型（qwen3.8-flash）连不上或连续报错时，自动切备用（glm-5.3-flash）。
"""

import json
import time

from openai import OpenAI

from config.settings import get_settings, load_app_config


class LLMClient:
    """对话补全、带工具的补全，都从这走。主模型挂了备用顶上。"""

    def __init__(self):
        cfg = get_settings()
        if not cfg.llm_api_key:
            raise RuntimeError("LLM_API_KEY 没配，聊天功能没法用")
        # 主客户端：qwen3.8-flash，走百炼的 OpenAI 兼容接口
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

        # 降级状态：主模型连续失败达到 max_errors 次就打开熔断（记下打开时刻），
        # 冷却期内彻底不碰主模型，省得每次白等 60 秒超时；冷却到期半开放行一次
        fb_cfg = load_app_config()["llm"].get("fallback", {})
        self._max_errors = fb_cfg.get("max_errors", 2)
        self._cooldown = float(fb_cfg.get("cooldown_seconds", 30) or 30)
        self._error_count = 0
        self._opened_at: float | None = None  # 熔断打开时刻（单调时钟），None=未打开

    # ---------- 内部：统一的主备切换，切换逻辑只写这一遍 ----------
    def _complete(self, **kwargs) -> object:
        """所有请求的必经之路。kwargs 是发给 chat.completions.create 的参数。

        返回原始响应对象，取 content 还是 tool_calls 由调用方自己决定。
        """
        last_error = None

        # 熔断是否仍处于打开状态：打开且未到冷却 → 直接走备用，连主模型都不碰
        # （这是修掉“冷却期里还白等主模型 60 秒超时”的关键）
        circuit_open = (
            self._opened_at is not None
            and (time.monotonic() - self._opened_at) < self._cooldown
        )

        if not circuit_open:
            # 正常态每次都试主模型；半开态（冷却到期）也只放行这一次探测。
            # Qwen3 系非流式调用必须关思考模式，不然官方直接拒绝请求
            thinking = load_app_config()["llm"].get("enable_thinking", False)
            kwargs.setdefault("extra_body", {"enable_thinking": thinking})
            try:
                resp = self._client.chat.completions.create(**kwargs)
                # 成功即闭合：清空错误计数、清掉熔断时刻
                self._error_count = 0
                self._opened_at = None
                return resp
            except Exception as exc:
                self._error_count += 1
                last_error = exc
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
            kwargs["extra_body"] = {"thinking_effort": "low"}
            kwargs["model"] = self._fallback_model
            try:
                return self._fallback.chat.completions.create(**kwargs)
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
