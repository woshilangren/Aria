"""能力层 - 工具调用管线

三级降级：模型自己带工具调（L1）→ 模型规划+手动执行（L2）→ 规则表硬编码（L3）。
模型不配合、接口不支持、参数瞎填，都有的接，保证用户要的东西尽量能拿到。
"""

import re

from config.settings import get_settings, load_app_config
from shared.singletons import get_llm, services
from shared.types import ToolCallSpec, ToolCallResult
from tools.misc import parse_llm_json

# 意图到工具的固定对应关系，最底层的保命通道
RULE_TOOL_MAP = {
    "weather": "weather_query",
    "search": "web_search",
    "image": "image_gen",
    "diary": "diary_write",
}

# 工具循环最多转几轮，防止模型一直要调工具停不下来
_MAX_LOOP_ROUNDS = 4

# 抠城市用的正则
_CITY_PATTERNS = (
    re.compile(r"([\u4e00-\u9fa5]{2,8}?)(?:市)?的?天气"),
    re.compile(r"(?:在|去|到)([\u4e00-\u9fa5]{2,6}?)(?:玩|旅游|出差)"),
)

# 用户话里的指令词和时间词，抽参数前先剥掉，免得混进 city/query 里
_STRIP_WORDS = (
    "帮我", "麻烦", "劳驾", "请问", "查询", "查一下", "查查", "查下", "看看",
    "看下", "问一下", "告诉", "一下", "查", "现在", "今天", "明天", "后天",
    "当前", "目前", "画一张", "画个", "画幅", "生成一张", "生成个", "来一张",
    "帮我画", "画一下", "画",
)

# 按长度从长到短剥，"查一下"才不会被"查"先拆散
_STRIP_SORTED = tuple(sorted(_STRIP_WORDS, key=len, reverse=True))


def _strip_commands(text: str) -> str:
    """把"帮我查一下"这类指令词从句子里剥掉，剩下的大概率就是正主。"""
    out = text or ""
    for word in _STRIP_SORTED:
        out = out.replace(word, "")
    return out.strip()


def _extract_city(text: str) -> str:
    """从"上海天气""在东京玩"这类句子里抠城市名，抠不到就空手而归。"""
    for pattern in _CITY_PATTERNS:
        m = pattern.search(text or "")
        if m:
            return m.group(1)
    return ""


def _build_rule_args(intent: str, user_text: str, fallback_city: str) -> dict:
    """按意图给规则通道凑工具参数。凑不出像样的就给默认值。"""
    cleaned = _strip_commands(user_text)
    if intent == "weather":
        city = _extract_city(cleaned) or fallback_city or "上海"
        return {"city": city}
    if intent == "search":
        return {"query": cleaned or user_text, "top_k": 3}
    if intent == "image":
        return {"prompt": cleaned or user_text}
    if intent == "diary":
        return {}  # 写日记不需要参数，会话和日期运行时自己知道
    return {}


class ToolGuard:
    """工具调用前的安检：工具名存不存在、必填参数齐不齐。"""

    def check(self, spec: ToolCallSpec) -> tuple:
        meta = services.get("tool_registry").get_meta(spec.tool_name)
        if meta is None:
            return False, f"没有这个工具：{spec.tool_name}"
        required = (meta.get("parameters") or {}).get("required") or []
        for key in required:
            if key not in (spec.arguments or {}):
                return False, f"{spec.tool_name} 缺少必填参数：{key}"
        return True, ""


class ToolPlanner:
    """L2 通道：让模型自己挑工具、填参数，人工负责执行。"""

    def plan(self, user_text: str, intent: str) -> ToolCallSpec:
        registry = services.get("tool_registry")
        names = [t["function"]["name"] for t in registry.catalog()]
        prompt = (
            f"可用工具：{names}\n"
            f"用户说：{user_text}\n"
            f"识别出的意图：{intent}\n"
            "选一个最合适的工具并给出参数，按 JSON 返回："
            '{"tool_name": "工具名", "arguments": {参数}}。'
            "没有合适的工具就把 tool_name 留空。只返回 JSON。"
        )
        raw = get_llm().chat(
            [{"role": "user", "content": prompt}], temperature=0.1, max_tokens=128
        )
        data = parse_llm_json(raw)
        tool_name = (data.get("tool_name") or "").strip()
        if not tool_name:
            raise ValueError("模型认为不需要工具")
        return ToolCallSpec(tool_name=tool_name, arguments=data.get("arguments") or {})


class ToolCallOrchestrator:
    """工具调用的总指挥，L1 挂了找 L2，L2 挂了找 L3，一层层往下兜。"""

    def run(
        self,
        user_text: str,
        intent: str,
        fallback_city: str = "",
        session_id: str = "",
        should_cancel=None,
    ) -> dict:
        """跑完三级降级，返回 {"results": [ToolCallResult...], "final_text": str}。

        final_text 只有 L1 通道才有：模型自己把答案说完了。
        session_id 是运行时上下文，注入式工具（比如写日记）靠它知道在给谁干活。
        should_cancel 是调用方注入的"该不该停"回调（返回 True 就尽早收工）。
        它只是个普通可调用对象，能力层不 import 调度层，避免越层依赖；
        真被取消时返回带 "cancelled": True 的结果，由上层决定怎么处理。
        """
        # L1：模型原生工具调用，模型自己决定调什么、调完自己组织答案
        if get_settings().llm_supports_tool_call:
            try:
                return self._run_native_loop(user_text, session_id, should_cancel)
            except Exception:
                pass

        # L2：模型规划工具和参数，执行和收尾由我们接手
        try:
            spec = ToolPlanner().plan(user_text, intent)
            ok, reason = ToolGuard().check(spec)
            if ok:
                out = services.get("tool_executor").execute(spec.tool_name, spec.arguments, session_id)
                result = ToolCallResult(spec.tool_name, out["status"], out["data"])
                return {"results": [result], "final_text": ""}
        except Exception:
            pass

        # L3：规则表硬上，参数靠正则抠，模型再怎么抽风都有底
        results = []
        rule_name = RULE_TOOL_MAP.get(intent)
        if rule_name:
            args = _build_rule_args(intent, user_text, fallback_city)
            out = services.get("tool_executor").execute(rule_name, args, session_id)
            results.append(ToolCallResult(rule_name, out["status"], out["data"]))
        return {"results": results, "final_text": ""}

    def _run_native_loop(self, user_text: str, session_id: str = "", should_cancel=None) -> dict:
        """L1 循环：模型带工具列表补全，要调工具就执行后把结果喂回去，
        直到它给出最终回答或轮数用完。"""
        registry = services.get("tool_registry")
        executor = services.get("tool_executor")
        catalog = registry.catalog()
        cfg = load_app_config()["tools"]
        rounds = cfg.get("max_tool_loop_rounds", _MAX_LOOP_ROUNDS)

        messages = [
            {
                "role": "system",
                "content": (
                    "需要查天气、搜资料或画图时调用对应工具；"
                    "用户想让你写日记时调用 diary_write 工具；"
                    "工具结果拿到后再回答用户。"
                ),
            },
            {"role": "user", "content": user_text},
        ]
        results = []
        guard = ToolGuard()

        for _ in range(rounds):
            # 取消检查点：每一轮开跑前先看看这轮还该不该继续，别白烧模型调用
            if should_cancel is not None and should_cancel():
                return {"results": results, "final_text": "", "cancelled": True}
            resp = get_llm().chat_with_tools(messages, catalog)

            # 模型说完了，把它的答案带回去
            if resp["type"] == "final":
                return {"results": results, "final_text": resp.get("content", "")}

            # 模型要调工具：过安检再执行
            spec = ToolCallSpec(
                tool_name=resp.get("name", ""),
                arguments=resp.get("arguments") or {},
            )
            ok, reason = guard.check(spec)
            if ok:
                out = executor.execute(spec.tool_name, spec.arguments, session_id)
            else:
                out = {"status": "error", "data": reason}
            results.append(
                ToolCallResult(spec.tool_name, out["status"], out["data"])
            )

            # 工具结果用普通文本回填，不走 tool role，兼容 GLM 这类接口
            #
            # ===== 这是一个有意取舍，不是写漏了，别顺手"修"成标准写法 =====
            # 标准写法是 {"role": "tool", "tool_call_id": <id>, "content": ...}，
            # 但当前接的 GLM 接口不吃 role:"tool"，传过去会直接报错，
            # 所以这里把结果降级成一条 role:"user" 的普通文本塞回去。
            #
            # 代价（已知，接受）：
            #   1. 丢掉了 call_id。模型看不出"这条结果对应哪次调用"，
            #      多轮多工具时它可能重复调同一个工具，或把结果张冠李戴。
            #   2. llm_client.py:128 只取首个 tool_calls（resp["tool_calls"][0]），
            #      模型一次想调多个工具时，后面的会被丢掉。
            # 这两条对现有场景（单用户、一轮基本一个工具）够用，所以留着。
            #
            # 将来要换成标准 tool role，动这三处：
            #   1. 这里：messages.append 改成 role:"tool" + tool_call_id；
            #   2. llm_client.py：chat_with_tools 要把 call_id 从响应里取出来往回传
            #      （现在 resp 里根本没有这个字段），并且别再只取 [0]；
            #   3. 确认目标模型真的支持 tool role —— 换成 OpenAI 系就可以，GLM 不行。
            messages.append(
                {"role": "assistant", "content": f"[调用工具 {spec.tool_name}]"}
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"工具 {spec.tool_name} 返回结果：{out['data']}，请据此回答用户。",
                }
            )

        # 轮数用完模型还没给答案，工具结果交给上层转述
        return {"results": results, "final_text": ""}
