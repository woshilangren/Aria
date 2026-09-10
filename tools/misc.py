"""杂项工具：拿当前时间、记日志、解析模型的 JSON 输出。

没什么花样，但别的层都离不开。
"""

import json
import logging
import re

from config.settings import load_app_config
from tools.storage import KVStoreTool


def parse_llm_json(raw: str) -> dict:
    """模型的输出经常裹着 ```json 围栏或者多余的话，剥出里面的 JSON。

    解析不出来就抛异常，让调用方自己兜底。
    """
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?\s*|\s*```$", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("模型输出里没有 JSON")
    return json.loads(raw[start : end + 1])


class ClockTool:
    """给模型提供时间感知用的时间工具。"""

    def now(self) -> str:
        """返回本地时间字符串，直接塞进提示词里。"""
        from datetime import datetime

        return datetime.now().isoformat(timespec="seconds")

    def period(self) -> str:
        """把当前时间归成一段（早晨/上午/下午/晚上/深夜），提示词里用。"""
        from datetime import datetime

        hour = datetime.now().hour
        if 5 <= hour < 9:
            return "早晨"
        if 9 <= hour < 12:
            return "上午"
        if 12 <= hour < 14:
            return "中午"
        if 14 <= hour < 18:
            return "下午"
        if 18 <= hour < 23:
            return "晚上"
        return "深夜"

    def time_guidance(self) -> str:
        """按时段给一句"现在说什么得体、什么话错位"的分寸提示，拼进系统提示词。

        光给时间戳模型经常视而不见，凌晨还会问"吃晚饭了吗"——
        把分寸直接讲明白，比指望它自己从 ISO 8601 里悟出来靠谱。
        """
        return {
            "早晨": "现在是清晨：可以关心他睡得好不好、今天有什么安排；别拿晚安、吃晚饭这类错位话题开场。",
            "上午": "现在是上午：正常白天的聊天节奏即可。",
            "中午": "现在是中午：可以关心他吃午饭了没有、要不要午休。",
            "下午": "现在是下午：正常白天的聊天节奏即可。",
            "晚上": "现在是晚上：可以问晚饭吃了吗、今天过得怎么样。",
            "深夜": "现在是深夜：绝不要问吃饭、说早安这类白天话；语气放轻放低，先关心他怎么还醒着、催他早点休息。",
        }.get(self.period(), "")


class Logger:
    """日志：控制台一份、文件一份，工具调用还会单独记到 tool_calls.jsonl。"""

    def __init__(self):
        cfg = load_app_config()["log"]
        self._logger = logging.getLogger("aria")
        self._logger.setLevel(cfg["level"])
        self._kv = KVStoreTool()
        if not self._logger.handlers:  # 防止重复初始化时 handler 翻倍
            fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            self._logger.addHandler(console)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def log_tool_call(self, tool_name: str, arguments: dict, status: str, result: str = "") -> None:
        """工具调用单独记一笔，以后排查"它到底调了啥"全靠这个。"""
        self._kv.log({"tool": tool_name, "arguments": arguments, "status": status, "result": result[:200]})
        self._logger.info(f"工具调用 {tool_name} {status}")
