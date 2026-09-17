"""杂项工具：拿当前时间、记日志、解析模型的 JSON 输出。

没什么花样，但别的层都离不开。
"""

import json
import logging
import re

from config.settings import load_app_config
from tools.storage import KVStoreTool


def _escape_stray_quotes(text: str) -> str:
    """把 JSON 字符串值内部的裸英文引号转义掉（S12-2）。

    LLM 在 reason/文本里写「他说"你"字」这类带 0x22 引号的内容是高频事故，
    会直接把 JSON 撑坏。判定依据：字符串的**闭合引号**后面紧跟结构分隔符
    （,}:]）；否则这个引号属于值的内容，转义它。
    已知局限：值内引号后面恰好紧跟逗号时会误判——此时二次解析仍失败，
    调用方走原有兜底，行为与修复前一致，不放大风险。
    """
    out = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_string and ch == "\\":
            out.append(text[i:i + 2])  # 已是合法转义序列，原样保留
            i += 2
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
            else:
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                nxt = text[j] if j < n else ""
                if nxt in ",}:]":
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def parse_llm_json(raw: str) -> dict:
    """模型的输出经常裹着 ```json 围栏或者多余的话，剥出里面的 JSON。

    解析不出来先试一次引号修复（S12-2：值内裸引号是 LLM 高频事故），
    再失败才抛异常，让调用方自己兜底。
    """
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?\s*|\s*```$", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("模型输出里没有 JSON")
    body = raw[start : end + 1]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return json.loads(_escape_stray_quotes(body))


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

    # ---- 间隔感知（N1）：她知道"多久没见"，而不是对 ISO 时间戳视而不见 ----
    # 七档间隔 → 人话。真人对"五分钟没回"和"三天没聊"的接话方式完全不同，
    # 光把 last_interaction 的原始时间戳塞进 prompt 模型基本无视，必须翻成人话。
    _GAP_TIERS = (
        (5 * 60, "刚聊完没两句", "话可以接得上文，不用重新打招呼"),
        (60 * 60, "刚分开一会儿", ""),
        (4 * 3600, "半天没见", "可以顺口问一句他刚才在忙什么"),
        (12 * 3600, "大半天没聊了", "可以自然问问这一天过得怎么样"),
        (24 * 3600, "昨天聊完到现在", "可以带一点'隔了一天'的感觉开场"),
        (3 * 86400, "两三天没理我", "可以表达一点'这才回来'的意思，但别兴师问罪"),
        (float("inf"), "好一阵子没见了", "关系越熟越可以直说想念或小小的不满，开场不用装没事"),
    )

    def gap_perception(self, last_iso: str, now=None) -> str:
        """把"距上次聊天多久"翻成一条可注入 prompt 的分寸行。

        last_iso 是 relationship.last_interaction 的 ISO 时间串；缺失或格式坏
        返回空串（什么都不注入，绝不硬凑）。纯本地计算，零 LLM 成本。
        now 参数留给测试注入固定时刻，生产代码不用传。
        """
        if not last_iso:
            return ""
        from datetime import datetime

        try:
            last = datetime.fromisoformat(last_iso)
        except (ValueError, TypeError):
            return ""
        now = now or datetime.now()
        gap = (now - last).total_seconds()
        if gap < 0:  # 时钟被拨回去之类的脏数据，当"刚聊完"处理最安全
            gap = 0
        for threshold, phrase, hint in self._GAP_TIERS:
            if gap < threshold:
                line = f"距离上次聊天：{phrase}"
                if hint:
                    line += f"。{hint}"
                return line
        return ""


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


def has_key(key: str) -> bool:
    """key 存在且不是占位符（占位符以 your- 开头，比如 your-xxx）。

    合并自 tools/external.py:17 + tools/speech.py:106，两份逻辑完全一致，
    现统一到 tools.misc._has_key，两处 import 即可，避免后续两处实现漂移。
    """
    return bool(key) and not key.startswith("your-")
