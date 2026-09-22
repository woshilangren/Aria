"""LLM 调用统一走这个文件，全项目不许自己另外建 OpenAI 客户端。

好处就一个：换模型、换地址、加重试，改这里一处就够了。
主模型（当前 claude-sonnet-5，走 nonelinear 的 OpenAI 兼容端点）连不上或
连续报错时，自动切备用（glm-5.3-flash）。参数按模型家族分发（主备同一套判定，
见 _apply_family_params）——发错参数不是"多带一个"，是直接 400。
"""

import json
import threading
import time
from typing import AsyncIterator

import openai
from openai import AsyncOpenAI, OpenAI

from shared.types import ExternalServiceError

from config.settings import get_settings, load_app_config

# 家族配置的合法取值（J13-4）。认不出的一律当 auto 处理：把配置写错一个字母
# 就让整个 LLM 链路起不来，比"家族猜错"损失大得多。
_KNOWN_FAMILIES = ("qwen", "claude", "glm")


def _family_of(configured: str, model: str) -> str:
    """把"显式家族配置 + 模型名"归一成 qwen / claude / glm / other。

    J13-4：以前判家族只有 `"qwen" in model` 这一条，于是 qwq / qvq 这种
    qwen 系但名字里不含 "qwen" 的模型被当成 Claude 剥掉 enable_thinking ——
    开源 qwen3 非流式默认 thinking=true，剥了参数官方直接 400。
    现在家族可以在 .env 里显式声明（LLM_FAMILY / LLM_FALLBACK_FAMILY）。

    `auto`（默认值）保留**原来那条一模一样的**按名字猜：默认行为必须与改动前
    逐字一致，否则现有部署（.env 一行没改）升级完当场炸。代价是 qwq / qvq
    仍然猜不出来——这类模型必须显式写 LLM_FAMILY=qwen，注释里说清楚比
    偷偷放宽猜测规则安全（放宽会改变所有 auto 部署的行为）。
    """
    fam = (configured or "auto").strip().lower()
    if fam in _KNOWN_FAMILIES:
        return fam
    name = (model or "").lower()
    if "qwen" in name:
        return "qwen"
    if "glm" in name:
        return "glm"
    if "claude" in name:
        return "claude"
    return "other"


async def _aclose_quietly(stream) -> None:
    """关掉流式响应，且**绝不抛**（J5）。

    openai 的 AsyncStream.close() 是协程，内部 aclose 掉 httpx response。
    为什么必须吞掉 close 自己的异常：它只在 finally 里被调用，这时候抛出去
    会**替换掉**正在往外传的真正事故异常——排查时看到的是"关流失败"，
    真正的原因反而没了。关流失败最坏是漏一条连接，丢根因是白查一轮。
    """
    if stream is None:
        return
    try:
        await stream.close()
    except Exception as exc:
        print(f"[llm] 关闭流式响应时出错（已忽略）：{exc}")


class LLMClient:
    """对话补全、带工具的补全，都从这走。主模型挂了备用顶上。"""

    def __init__(self):
        cfg = get_settings()
        if not cfg.llm_api_key:
            raise RuntimeError("LLM_API_KEY 没配，聊天功能没法用")
        # J4：max_retries 必须显式写 0。SDK 默认 max_retries=2，也就是每次
        # create() 内部最多打 3 发；主模型 60s 超时 + 备用 90s 超时叠起来，
        # 最坏墙钟 3×60 + 3×90 ≈ 450 秒才向用户报错——用户早就关页面了。
        # 更坏的是它污染熔断：自研熔断器把"一次 create() 抛错"记一格，
        # 而那一格里 SDK 已经偷偷重试过 3 次，等于 max_errors=2 实际是
        # "6 次网络失败"，抖动一下就误开熔断、把感知/画像/日记全切小模型。
        # 重试语义收归熔断器一家管：SDK 一次都不许自己重试。
        #
        # 主客户端：模型/地址/key 全来自 .env（当前 claude-sonnet-5 + nonelinear）
        self._client = OpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            timeout=60,
            max_retries=0,
        )
        self._model = cfg.llm_model
        # 家族（J13-4）：主备各判各的，构造期定死一次——运行中不会换模型
        self._family = _family_of(cfg.llm_family, cfg.llm_model)

        # 备用客户端：GLM 只在主模型降级时顶上；没配 key 就置 None，不挡主模型干活
        if cfg.llm_fallback_api_key:
            self._fallback = OpenAI(
                api_key=cfg.llm_fallback_api_key,
                base_url=cfg.llm_fallback_base_url,
                timeout=90,
                max_retries=0,
            )
            self._fallback_model = cfg.llm_fallback_model
        else:
            self._fallback = None
            self._fallback_model = ""
        self._fallback_family = _family_of(cfg.llm_fallback_family, cfg.llm_fallback_model)

        # 异步客户端：只给流式用（C1a 先加接口，尚未接进管道）。同步客户端一律保留——
        # 记忆摘要 / 画像 / 感知 / 日记那几处还在用同步的 chat()。
        self._aclient = AsyncOpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url,
            timeout=60,
            max_retries=0,
        )
        if cfg.llm_fallback_api_key:
            self._afallback = AsyncOpenAI(
                api_key=cfg.llm_fallback_api_key,
                base_url=cfg.llm_fallback_base_url,
                timeout=90,
                max_retries=0,
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
        # 半开探测名额（J13-1）：True = 已经有一个请求在探测主模型。
        # 原来"能不能探测"是锁外的 check-then-act：冷却到期的那一瞬间，
        # N 个并发请求全看见 _opened_at 非空且已过冷却，于是全都跑去探测、
        # 各白等一个探测超时，_probe_fail_streak 也被加 N 次（指数退避一步跳到
        # 封顶）——注释写的"放行一次半开探测"实际放行 N 次。
        # 名额必须在 _cb_lock 内 check-and-set：抢到的探测，没抢到的直接走备用。
        self._probe_in_flight = False
        # 熔断状态被线程池（chat）、事件循环（astream）、巡检线程（日记 LLM）三方
        # 无锁共享时，"判断 circuit_open → 更新计数"是复合操作，交错读写会互相误判。
        # 一把小锁只保护状态字段，不包网络调用。
        self._cb_lock = threading.Lock()

    # 指数退避的指数封顶。不是防"退避太久"——60 秒那道 min 已经管住了；
    # 是防 `2 ** streak` 这个 int 本身大到转不成 float（streak 过 1024 就
    # OverflowError，而 min 拦不住它，因为溢出发生在乘法那一步）。
    # streak 每失败一次探测 +1、探测最快每 cooldown(120s) 一次 → 主模型连挂
    # 约 34 小时就到（正是"key 失效没人发现"那个场景）。取 6 是无损的：
    # 默认探测超时下 2**4 就已经顶到 60 秒天花板了。
    _PROBE_BACKOFF_MAX_POW = 6

    def _probe_timeout_now(self) -> float:
        """本次半开探测用的超时：连续失败就逐次翻倍，封顶 60 秒（客户端默认值）。"""
        return min(60.0, self._probe_timeout
                   * (2 ** min(self._probe_fail_streak, self._PROBE_BACKOFF_MAX_POW)))

    def _is_qwen(self) -> bool:
        """主模型是不是 qwen 系——qwen 专有参数只对它发。

        CLAUDE.md「有意的设计取舍」第 9 条点名的就是这个判定口子，保留方法名不动；
        实际的参数分派走 _apply_family_params（主备共用一套）。

        getattr 兜底是必要的，不是防御性冗余：tests/unit/test_invariants.py 用
        `LLMClient.__new__` 绕过 __init__（它要真 key）再只塞 `_model` 来验证判定，
        直接读 self._family 会 AttributeError。

        注意：这里**不要**再抽 `_main_family()` / `_fallback_family()` 之类的
        访问器方法——`_fallback_family` 已经和 __init__ 里的同名字符串属性撞了，
        属性会盖掉方法，调用时是 `TypeError: 'str' object is not callable`，
        整条备用降级路径当场死掉（本次改动实测踩中）。家族值就在属性里，直接读。
        """
        family = getattr(self, "_family", None)
        if family is None:
            family = _family_of("auto", getattr(self, "_model", ""))
        return family == "qwen"

    @staticmethod
    def _apply_family_params(kwargs: dict, family: str, thinking: bool) -> None:
        """按家族就地改 kwargs——发错参数不是"多带一个"，是直接 400。

        J13-3：以前只有主模型走家族分发，备用模型硬编 `thinking_effort=low`
        （那是 GLM 专有参数），备用一换成非 GLM 家族就 400；而备用 400 会被
        熔断器记成"备用也挂了"，主备一起判死，用户看到的是彻底没回复。
        现在主备共用这一处判定：

        - qwen：Qwen3 系非流式必须显式声明思考模式，不然官方直接拒请求；
        - glm：GLM-5 系"始终思考"、不吃 enable_thinking，换 thinking_effort=low
          ——实测能把每条回复的推理开销砍半（500→250 上下），不然聊天每条都要
          先默默想十几秒才开口，用户只会觉得"怎么这么慢"；
        - claude / other：enable_thinking 是 qwen 专有、temperature 在 Sonnet 5 上
          已移除，两个都不带，交给模型默认（Claude 自带 adaptive thinking）。

        注意 auto 家族下 GLM 当**主**模型时行为有一处变化：原来它落进 else 分支
        被剥掉 temperature，现在会带上 thinking_effort=low。这是有意的——
        主备对称才是这条的目的，且 GLM 本来就吃这个参数（备用一直在发）。
        """
        if family == "qwen":
            kwargs.setdefault("extra_body", {"enable_thinking": thinking})
        elif family == "glm":
            # 主模型若是 qwen，这里会把它的 enable_thinking 整个换掉，不能 setdefault
            kwargs["extra_body"] = {"thinking_effort": "low"}
        else:
            kwargs.pop("extra_body", None)
            kwargs.pop("temperature", None)

    def _main_failure_text(self, last_error, cooling: bool, probe_blocked: bool) -> str:
        """把"主模型为什么没干活"翻成人话（J13-2）。

        熔断打开时主模型压根没被调用，last_error 一直是 None，原来的报错就成了
        "主=None，备=xxx"——出了事故翻日志根本看不出主模型是挂了、还是我们
        根本没试它。占位串必须自己说清是哪种。
        """
        if last_error is not None:
            return str(last_error)
        if cooling:
            return (
                f"熔断打开中（主模型连挂 {self._max_errors} 次，"
                f"冷却 {self._cooldown:.0f}s 内不再尝试）"
            )
        if probe_blocked:
            return "熔断半开、探测名额被别的请求占着（本次没碰主模型）"
        return "主模型未被调用（原因不明）"

    # ---------- 内部：统一的主备切换，切换逻辑只写这一遍 ----------
    def _complete(self, **kwargs) -> object:
        """所有请求的必经之路。kwargs 是发给 chat.completions.create 的参数。

        返回原始响应对象，取 content 还是 tool_calls 由调用方自己决定。
        """
        last_error = None
        thinking = load_app_config()["llm"].get("enable_thinking", False)

        # 熔断是否仍处于打开状态：打开且未到冷却 → 直接走备用，连主模型都不碰
        # （这是修掉“冷却期里还白等主模型 60 秒超时”的关键）
        #
        # 半开探测名额（J13-1）必须在锁内 check-and-set：原来 is_probe 是锁外算的，
        # 冷却到期的那一瞬间 N 个并发线程全看见"可以探测"，全跑去主模型各白等一个
        # 探测超时，_probe_fail_streak 也被加 N 次（指数退避一步跳到封顶 60 秒）。
        # 现在抢到名额的那个探测，没抢到的直接走备用——用户不必陪着一起等。
        with self._cb_lock:
            cooling = (
                self._opened_at is not None
                and (time.monotonic() - self._opened_at) < self._cooldown
            )
            is_probe = False
            probe_blocked = False
            if not cooling and self._opened_at is not None:
                if self._probe_in_flight:
                    probe_blocked = True
                else:
                    self._probe_in_flight = True
                    is_probe = True

        if not cooling and not probe_blocked:
            # 这次是不是"半开探测"：熔断原本开着（_opened_at 非空）且已过冷却，才会走到这里。
            # 探测复用主客户端，但默认 60 秒超时对人聊天的节奏太长（几乎每条都白等），
            # 所以只给探测传短超时；正常调用绝不传，保持客户端默认 60 秒。
            #
            # 参数分派和探测超时的计算也放在 try 里面：它们一旦抛，finally 才还能
            # 把探测名额还回去。名额漏了的后果是熔断从此再也放不出探测、
            # 永久钉死在备用模型上——那比一次调用失败严重得多。
            # （这里曾真炸过一次：_probe_fail_streak 无上限时 2**n 会 OverflowError。
            # 现在指数由 _PROBE_BACKOFF_MAX_POW 封顶，但 try 的位置不许挪出去——
            # 下一个人再加一个会抛的计算时，靠的就是它。）
            #
            # R27b：只把 **openai SDK 的异常**当供应商失败计数并转 ExternalServiceError；
            # TypeError/NameError 这类本地装配缺陷原样向上抛——它们不计入熔断、
            # 不许伪装成"模型挂了"（参数分发/结果装配是本地计算，不是外部故障）。
            try:
                self._apply_family_params(kwargs, self._family, thinking)
                if is_probe:
                    kwargs["timeout"] = self._probe_timeout_now()
                resp = self._client.chat.completions.create(**kwargs)
                # 成功即闭合：清空错误计数、清掉熔断时刻、探测超时回到最短档
                with self._cb_lock:
                    self._error_count = 0
                    self._opened_at = None
                    self._probe_fail_streak = 0
                return resp
            except openai.OpenAIError as exc:
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
            finally:
                # 名额无论成功失败都得还回去，否则熔断从此再也放不出探测、永久停在备用
                if is_probe:
                    with self._cb_lock:
                        self._probe_in_flight = False

        # 备用顶上：参数同样按家族发（J13-3），不再硬编 GLM 的 thinking_effort
        if self._fallback is not None:
            kwargs.pop("timeout", None)  # 短超时只给半开探测用，备用走它自己的默认超时
            self._apply_family_params(kwargs, self._fallback_family, thinking)
            kwargs["model"] = self._fallback_model
            try:
                resp = self._fallback.chat.completions.create(**kwargs)
                # 备用每成功一次就把主模型错误计数衰减一格，而不是只等主模型自己成功清零——
                # 否则相隔几小时的两次网络抖动各自被兜住、计数却累到阈值，白开一轮 120 秒熔断，
                # 期间感知/画像/日记全换小模型，用户感知为"她突然变了个人"
                with self._cb_lock:
                    self._error_count = max(0, self._error_count - 1)
                return resp
            except openai.OpenAIError as exc:
                main_why = self._main_failure_text(last_error, cooling, probe_blocked)
                raise ExternalServiceError(
                    "llm", "all_providers_failed", retryable=True,
                    detail=f"主={main_why}，备={exc}",
                ) from exc

        main_why = self._main_failure_text(last_error, cooling, probe_blocked)
        raise ExternalServiceError(
            "llm", "no_provider_available", retryable=True, detail=main_why,
        )

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
        _probe_timeout / _probe_in_flight），不另起炉灶。

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
        # 本轮的思考开关：主备两个 qwen 分支共用，所以提前算一次
        # （原版只在"主模型是 qwen"的分支里读，备用现在也按家族发参数就要用了）
        thinking = cfg.get("enable_thinking", False) if enable_thinking is None else enable_thinking

        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        last_error = None

        # 熔断打开且未到冷却 → 直接走备用（与同步 _complete 同一判断、同一把锁、
        # 同一份探测名额——两条路径并发共享状态，名额不共用就还是会放出 N 个探测）
        with self._cb_lock:
            cooling = (
                self._opened_at is not None
                and (time.monotonic() - self._opened_at) < self._cooldown
            )
            is_probe = False
            probe_blocked = False
            if not cooling and self._opened_at is not None:
                if self._probe_in_flight:
                    probe_blocked = True
                else:
                    self._probe_in_flight = True
                    is_probe = True

        if not cooling and not probe_blocked:
            # 半开探测：只在这时传短超时（连续失败逐次翻倍，见 _probe_timeout_now）
            # 家族专有参数按家族发（见 _apply_family_params，与同步路径共用一处）
            stream = None
            try:
                # 同 _complete：分派与超时计算也放进 try，它们万一抛了 finally 才还能
                # 归还探测名额（漏了 = 熔断永久钉死在备用上）
                self._apply_family_params(kwargs, self._family, thinking)
                if is_probe:
                    kwargs["timeout"] = self._probe_timeout_now()
                try:
                    stream = await self._aclient.chat.completions.create(**kwargs)
                except openai.OpenAIError as exc:
                    # R27b：只把 SDK 异常当供应商失败；本地装配缺陷原样上抛不计数
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
                    except openai.OpenAIError as exc:
                        with self._cb_lock:
                            self._error_count += 1
                            last_error = exc
                            if self._error_count >= self._max_errors:
                                self._opened_at = time.monotonic()
                        if emitted:
                            # R27b：吐过 token 换备用也接不上——统一转窄载体交给上层
                            # 收尾（管道按"外部服务失败"走允许的降级，不计内部错误）
                            raise ExternalServiceError(
                                "llm", "stream_interrupted", retryable=False,
                                detail=str(exc),
                            ) from exc
                        # 一个 token 都没吐：落到下面走备用
            finally:
                # J5：主模型的流必须在**每一条**退出路径上都关掉。三个漏点：
                #   1) 中途出错且没吐过内容 → 回落去试备用，主模型的流被丢下；
                #   2) 已经吐过内容再出错 → 直接抛给上层，流同样被丢下；
                #   3) 客户端断连 → 抛进 yield 点的是 GeneratorExit，它继承
                #      BaseException，上面那个 `except Exception` 根本抓不到。
                # 不关就是 httpx 连接等 GC 兜底：24/7 跑下来句柄和连接池一直漏。
                # close 自身容错（见 _aclose_quietly）——它绝不能把真正的事故异常盖掉。
                await _aclose_quietly(stream)
                # 探测名额无论成功失败都要还回去（J13-1），否则熔断再也放不出探测
                if is_probe:
                    with self._cb_lock:
                        self._probe_in_flight = False

        # 备用顶上：参数同样按家族发（J13-3），不再硬编 GLM 的 thinking_effort
        if self._afallback is not None:
            kwargs.pop("timeout", None)  # 短超时只给半开探测用，备用走它自己的默认超时
            self._apply_family_params(kwargs, self._fallback_family, thinking)
            kwargs["model"] = self._fallback_model
            try:
                stream = await self._afallback.chat.completions.create(**kwargs)
            except openai.OpenAIError as exc:
                main_why = self._main_failure_text(last_error, cooling, probe_blocked)
                raise ExternalServiceError(
                    "llm", "all_providers_failed", retryable=True,
                    detail=f"主={main_why}，备={exc}",
                ) from exc
            emitted = False
            try:
                async for chunk in stream:
                    delta = self._delta_of(chunk)
                    if delta:
                        emitted = True
                        yield delta
                with self._cb_lock:
                    self._error_count = max(0, self._error_count - 1)  # 备用成功，错误计数衰减（同同步路径）
            except openai.OpenAIError as exc:
                # R27b：备用流中途断流同样转窄载体——以前这里裸抛 SDK 异常，
                # 管道会把它误分成 internal_error（程序错误），污染错误语义
                raise ExternalServiceError(
                    "llm", "stream_interrupted", retryable=False,
                    detail=f"备用流中断（已出 {emitted}）：{exc}",
                ) from exc
            finally:
                # 备用的流也一样要关：断连/中途抛同样会把它丢在半路（与主模型同一类漏）
                await _aclose_quietly(stream)
            return

        main_why = self._main_failure_text(last_error, cooling, probe_blocked)
        raise ExternalServiceError(
            "llm", "no_provider_available", retryable=True, detail=main_why,
        )

    @staticmethod
    def _delta_of(chunk) -> str:
        """从流式分块里取文本增量；取不到（无 choices / 非文本增量）返回空串。"""
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        return getattr(delta, "content", None) or ""
