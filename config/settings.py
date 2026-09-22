"""配置模块：环境变量（.env）管密钥和地址，config.json 管业务参数。

这两类东西分开的原因很简单：密钥不能写进代码库，业务参数（温度、重试次数
这些）改起来不用动 .env。
"""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from os import getenv

# 项目根目录，别的地方要用路径就从这拿
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 读 .env，只在本模块第一次 import 时做一次
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    """所有配置项都集中在这，别的模块只管问这里要。"""

    # ---- LLM（主模型当前 claude-sonnet-5 / nonelinear，备用 GLM 只在主模型连不上时顶上；
    #      语音/向量/画图仍走百炼——LLM_API_KEY 与它们是独立变量）----
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    # 模型家族（J13-4）：qwen / claude / glm / auto。
    # 为什么不能靠模型名猜：原来只有 `"qwen" in model` 这一条判据，而 qwq / qvq 系
    # 名字里不含 "qwen"，会被当成 Claude 剥掉 enable_thinking —— 开源 qwen3 非流式
    # 默认 thinking=true，剥了参数就是 400。auto = 保留旧的按名字猜（默认值，
    # 现有部署一行 .env 都不用改，行为完全不变）；填了显式家族就以配置为准。
    llm_family: str
    llm_supports_tool_call: bool
    llm_fallback_api_key: str
    llm_fallback_base_url: str
    llm_fallback_model: str
    llm_fallback_family: str       # 同上，备用模型的家族

    # ---- 语音 / 图片 / 外部服务（没配就等于这个功能不可用）----
    asr_api_key: str
    asr_base_url: str
    asr_model: str
    tts_api_key: str
    tts_base_url: str
    tts_model: str
    tts_voice: str
    weather_api_key: str
    search_api_key: str
    image_gen_api_key: str
    image_gen_model: str

    # ---- Realtime 实时语音（一条线直连多模态模型，不用 ASR/TTS 两段接力）----
    realtime_api_key: str          # 留空自动复用百炼那把 TTS key
    realtime_model: str
    realtime_voice: str            # 默认音色，想换自定义音色改 .env 的 REALTIME_VOICE
    realtime_workspace_id: str     # 业务空间 ID，留空走通用域名

    # ---- 向量模型（dashscope 原生接口，不吃 OpenAI 兼容那套）----
    embedding_api_key: str
    embedding_model: str
    embedding_dimension: int

    # ---- 存储 ----
    data_dir: Path

    # ---- 访问口令（公网部署时必配：空=不鉴权，仅限本机/内网使用）----
    access_token: str

    # ---- 监听地址（默认 0.0.0.0 对外可访问；填 127.0.0.1 只听本机）----
    bind_host: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """把 .env 读成 Settings，用 lru_cache 保证全程只有这一份实例。"""
    return Settings(
        llm_api_key=getenv("LLM_API_KEY", ""),
        llm_base_url=getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        llm_model=getenv("LLM_MODEL", "qwen3.8-flash"),
        llm_family=getenv("LLM_FAMILY", "auto"),
        llm_supports_tool_call=getenv("LLM_SUPPORTS_TOOL_CALL", "true").lower() == "true",
        llm_fallback_api_key=getenv("LLM_FALLBACK_API_KEY", ""),
        llm_fallback_base_url=getenv("LLM_FALLBACK_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        llm_fallback_model=getenv("LLM_FALLBACK_MODEL", "glm-5.3-flash"),
        llm_fallback_family=getenv("LLM_FALLBACK_FAMILY", "auto"),
        asr_api_key=getenv("ASR_API_KEY", ""),
        asr_base_url=getenv("ASR_BASE_URL", ""),
        asr_model=getenv("ASR_MODEL", "qwen-audio-3.0-asr-flash-streaming"),
        tts_api_key=getenv("TTS_API_KEY", ""),
        tts_base_url=getenv("TTS_BASE_URL", ""),
        tts_model=getenv("TTS_MODEL", "qwen-audio-3.0-tts-plus"),
        tts_voice=getenv("TTS_VOICE", ""),
        weather_api_key=getenv("WEATHER_API_KEY", ""),
        search_api_key=getenv("SEARCH_API_KEY", ""),
        image_gen_api_key=getenv("IMAGE_GEN_API_KEY", ""),
        image_gen_model=getenv("IMAGE_GEN_MODEL", "qwen-image-3.0-pro"),
        realtime_api_key=getenv("REALTIME_API_KEY", ""),
        realtime_model=getenv("REALTIME_MODEL", "qwen3.5-omni-plus-realtime"),
        realtime_voice=getenv("REALTIME_VOICE", "Maia"),
        realtime_workspace_id=getenv("REALTIME_WORKSPACE_ID", ""),
        embedding_api_key=getenv("EMBEDDING_API_KEY", ""),
        embedding_model=getenv("EMBEDDING_MODEL", "qwen3-vl-embedding"),
        embedding_dimension=int(getenv("EMBEDDING_DIMENSION", "2560")),
        data_dir=Path(getenv("DATA_DIR", str(PROJECT_ROOT / "storage"))),
        access_token=getenv("ACCESS_TOKEN", ""),
        bind_host=getenv("BIND_HOST", "0.0.0.0"),
    )


# R07a：已知 section 的关键字段类型表——config.json 是用户可改的，坏值不校验
# 就会流到深处炸（"asr_seconds": "60s" 这类）。坏字段**单独**回默认并告警，
# 同 section 的好字段照常生效（不整包丢掉）。
_CFG_FIELD_KINDS = {
    "llm": {
        "temperature": "number", "max_tokens": "number",
        # R12c 的旧键与新键并存期：enable_thinking 是旧键（读取侧迁移到
        # thinking.enabled），fallback 是熔断配置段（llm_client 读取）
        "enable_thinking": "bool", "fallback": "dict",
    },
    "memory": {
        "max_turns_short_term": "number", "recall_top_k": "number",
        "portrait_tag_limit": "number", "max_sessions": "number",
        "profile_confidence_threshold": "number", "persona_decay_days": "number",
        "diary_max_entries": "number", "session_update_interval": "number",
    },
    "proactive": {
        "idle_minutes": "number", "enabled": "bool",
        "daily_max": "number", "min_intimacy": "number",
    },
    "tools": {"max_retry": "number", "max_tool_loop_rounds": "number"},
    "personality": {"quirk_rate": "number"},
    "thinking": {
        "enabled": "bool", "char_threshold": "number",
        "never_on_comfort": "bool", "never_on_emotions": "list",
        "force_if_last_poor": "bool",
    },
    "expression": {
        "minimal_input_chars": "number", "minimal_allowlist": "list",
        "reaction_tag": "str", "long_input_chars": "number",
        "long_input_require_split": "bool", "force_cut_chars": "number",
    },
    "timeouts": {
        "asr_seconds": "timeout", "tts_seconds": "timeout",
        "image_seconds": "timeout",
    },
    "log": {"level": "str"},
}


def _cfg_value_ok(value, kind: str) -> bool:
    """按字段种类校验值。number 拒绝 bool 与 NaN/Infinity（bool 是 int 的子类，
    不显式挡掉的话 "enabled": true 会顺手把 quirk_rate 顶成 1.0）。"""
    if kind == "number":
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value))
    if kind == "timeout":
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "str":
        return isinstance(value, str)
    if kind == "list":
        return isinstance(value, list)
    if kind == "dict":
        return isinstance(value, dict)
    return False


@lru_cache(maxsize=1)
def load_app_config() -> dict:
    """读 config.json 的业务参数。文件不存在或字段缺失就用默认值，不报错。

    R07a：JSON 根必须是对象；已知字段按类型校验，坏字段**单独**回默认并
    告警（同 section 的好字段照常生效）；未知 section/键保留但告警——
    不让一个拼写错误静默失效，也不让一个坏值炸掉整包配置。
    """
    defaults = {
        "llm": {"temperature": 0.7, "max_tokens": 1024},
        "memory": {
            "max_turns_short_term": 12,
            "recall_top_k": 5,
            "portrait_tag_limit": 5,  # 与 portrait 抽取 prompt 里的"标签不超过 5 个"对齐（memory.py:_decay_tags 取 [:limit]）
            "max_sessions": 8,                # KEEPER 同时持有的会话上下文上限，超出最久未用淘汰
            "profile_confidence_threshold": 0.6,  # 画像硬事实入库置信度门槛（参见 MemoryGatekeeper）
            "persona_decay_days": 14,         # 标签多久没再被提到就过期清理（见 _decay_tags）
        },
        "proactive": {
            "idle_minutes": 30,
            # 主动开口（N3）：默认关——宁缺毋滥，不招人烦是安全阀不是可选项。
            # 开了也要过"活跃窗口 + 亲密度门槛 + 每天上限 + 收手环"四道闸
            "enabled": False,
            "daily_max": 2,
            "min_intimacy": 20,
        },
        "tools": {
            "max_retry": 2,
            "max_tool_loop_rounds": 4,
        },
        # 随机小动作概率：0~1，0 关掉，1 每句都皮（安抚/危机轮永远不出手）
        "personality": {
            "quirk_rate": 0.12,
        },
        # 什么时候让模型"认真想"。不是全局开关——按输入复杂度逐轮判断
        "thinking": {
            "enabled": True,
            "char_threshold": 60,          # 输入达到这个字数就触发
            "never_on_comfort": True,      # 安抚轮永远不想：这时候要温度不是分析
            "never_on_emotions": ["sad", "angry"],
            "force_if_last_poor": True,    # 上一轮被判敷衍/出戏，这轮强制想
        },
        # 表达尺度：极简反应的放行规则 + 长输入放开字数
        "expression": {
            "minimal_input_chars": 40,     # 输入超过这个数，极简回复才算敷衍
            "minimal_allowlist": [],
            "reaction_tag": "@r",
            "long_input_chars": 60,        # 输入达到这个数就放开字数限制
            "long_input_require_split": True,
            "force_cut_chars": 40,
        },
        # 外部同步调用的墙钟上限（J3）。以前这三处 dashscope 调用完全没有超时，
        # 而它们都跑在 asyncio.to_thread 的**默认线程池**里：一个僵死请求占住一个
        # 池线程且永不释放，攒够 min(32, cpu+4) 个之后所有 offload 调用（包括
        # SQLite 写回）开始排队——表现是"整个服务卡死"，而日志里一条报错都没有。
        # 这是 24/7 运行最大的挂起风险，所以每个都必须有上限。
        "timeouts": {
            # 整段语音转写：一句话几秒到几十秒，60 秒还没回就是链路挂了，
            # 再等下去用户早就走开了
            "asr_seconds": 60,
            # 语音合成：SDK 原生吃 timeout_millis（超时抛 TimeoutError 并在 finally
            # 里关掉 websocket），比外面包线程池干净，所以直接把这个值传进去
            "tts_seconds": 60,
            # 画图任务轮询：qwen-image pro 出图常态十几秒到一两分钟，给 5 分钟。
            # 超时后 SDK 返回 status_code=408 / code=WaitTaskTimeout，不会挂住
            "image_seconds": 300,
        },
        "log": {"level": "INFO"},
    }
    path = PROJECT_ROOT / "config.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            # 配置文件写坏了也别让程序起不来，整包用默认值凑合，但必须喊一声
            print(f"[config] config.json 解析失败，整包回默认值（原件未改动）: {exc}")
            user_cfg = None
        if user_cfg is not None:
            if not isinstance(user_cfg, dict):
                print(f"[config] config.json 根必须是对象（实测 {type(user_cfg).__name__}），"
                      f"整包回默认值（原件未改动）")
            else:
                for section, values in user_cfg.items():
                    if section not in defaults:
                        print(f"[config] config.json 里出现未知 section「{section}」，已忽略")
                        continue
                    if not isinstance(values, dict):
                        print(f"[config] config.json 的「{section}」段必须是对象"
                              f"（实测 {type(values).__name__}），该段整段回默认")
                        continue
                    kinds = _CFG_FIELD_KINDS.get(section, {})
                    for key, val in values.items():
                        kind = kinds.get(key)
                        if kind is None:
                            print(f"[config] config.json 里出现未知键「{section}.{key}」，已忽略")
                            continue
                        if _cfg_value_ok(val, kind):
                            defaults[section][key] = val
                        else:
                            print(f"[config] 「{section}.{key}」类型/取值不合法"
                                  f"（实测 {val!r}），该字段回默认值")
    return defaults


def timeout_seconds(name: str, default: float) -> float:
    """安全读 `timeouts.<name>`（J3）：缺失 / 类型坏 / 非正数一律退回 default。

    为什么不让调用方直接 `load_app_config()["timeouts"][name]`：config.json 是用户
    可改的，写成 `"asr_seconds": "60s"` 或直接删掉整段都有可能——直接下标会抛
    KeyError 把语音链路整个炸掉，把字符串透传给 SDK 则会在更深的地方炸。
    兜底哲学要求"读不到返回默认值而不是抛异常"，这个口子就是那条纪律的落点。
    R07a：显式拒绝 bool（True 会 float 成 1.0 蒙混过关）与 NaN/Infinity——
    NaN 的 `> 0` 判定为 False 挡得住，Infinity 却能溜过去，必须用 isfinite。
    """
    try:
        raw = load_app_config().get("timeouts", {}).get(name, default)
        if isinstance(raw, bool):
            return float(default)
        val = float(raw)
    except (TypeError, ValueError, AttributeError):
        return float(default)
    if not math.isfinite(val) or val <= 0:
        return float(default)
    return val
