"""配置模块：环境变量（.env）管密钥和地址，config.json 管业务参数。

这两类东西分开的原因很简单：密钥不能写进代码库，业务参数（温度、重试次数
这些）改起来不用动 .env。
"""

import json
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
    llm_supports_tool_call: bool
    llm_fallback_api_key: str
    llm_fallback_base_url: str
    llm_fallback_model: str

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
        llm_supports_tool_call=getenv("LLM_SUPPORTS_TOOL_CALL", "true").lower() == "true",
        llm_fallback_api_key=getenv("LLM_FALLBACK_API_KEY", ""),
        llm_fallback_base_url=getenv("LLM_FALLBACK_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        llm_fallback_model=getenv("LLM_FALLBACK_MODEL", "glm-5.3-flash"),
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


@lru_cache(maxsize=1)
def load_app_config() -> dict:
    """读 config.json 的业务参数。文件不存在或字段缺失就用默认值，不报错。"""
    defaults = {
        "llm": {"temperature": 0.7, "max_tokens": 1024},
        "memory": {
            "max_turns_short_term": 12,
            "recall_top_k": 5,
            "portrait_tag_limit": 20,
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
        "log": {"level": "INFO"},
    }
    path = PROJECT_ROOT / "config.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            for section, values in user_cfg.items():
                if section in defaults and isinstance(values, dict):
                    defaults[section].update(values)
        except (json.JSONDecodeError, OSError):
            # 配置文件写坏了也别让程序起不来，用默认值凑合
            pass
    return defaults
