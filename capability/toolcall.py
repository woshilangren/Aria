"""能力层 - 工具调用管线

两级降级：模型自己带工具调（L1）→ 规则表硬编码（L3）。
模型不配合、接口不支持、参数瞎填，都有的接，保证用户要的东西尽量能拿到。

为什么没有 L2（D3）：L2 曾是"模型不支持 function calling 时，让模型用文本规划出
工具名和参数、代码手动执行"。意图只有 4 个固定值（weather/search/image/diary），
L3 的规则表 + 正则已经能覆盖，L2 那一层是白花一次模型调用，还多开一处
"重跑有副作用的工具"的口子（见 I3）。
**代价（作者 triage 要求备注）**：删掉之后，"模型不支持 function calling"的场景
只剩 L3 规则表，规则表没有 LLM 规划的参数灵活性——意图一多、参数一复杂就不够用。
当前 4 个固定意图下够用；将来意图扩到两位数，再把规划层加回来（届时必须
连同 I3 的幂等账本一起接，否则又是三倍画图）。

面向模型的字符串里不出现"工具"二字（C3）：GLM 兼容只要求把 role 从 "tool"
换成 "user"（CLAUDE.md 取舍 #5），**从来不要求写"工具"两个字**——这两件事以前被
混在一起了。模型看到「工具 X 返回结果：…」就会把"工具"当成她嘴里的正常词，
真机 turn 7 那句「工具这边什么都没查到」就是这么来的。
"""

import json
import re

from config.settings import get_settings, load_app_config
from shared.singletons import get_llm, services
from shared.types import ToolCallSpec, ToolCallResult

# 意图到工具的固定对应关系，最底层的保命通道
RULE_TOOL_MAP = {
    "weather": "weather_query",
    "search": "web_search",
    "image": "image_gen",
    "diary": "diary_write",
}

# 工具循环最多转几轮，防止模型一直要调工具停不下来
_MAX_LOOP_ROUNDS = 4

# 有副作用的调用：花了钱（画图）或写了库（日记）。**一条消息里绝不跑第二遍**。
# registry 的 meta 目前只有 name/description/parameters，没有 side_effect 字段，
# 所以名单先维护在这里；建议 registry 补上 side_effect 后改成从 meta 读（见报告）。
_SIDE_EFFECT_TOOLS = frozenset({"image_gen", "diary_write"})

# 抠城市用的正则。
# I4：第一条以前是 ([\u4e00-\u9fa5]{2,8}?)，惰性匹配对"北京到上海的天气"会一路吃到
# "北京到上海"当城市名。收紧两处——① 长度收到 2~4（国内城市名最长就是"乌鲁木齐"
# "呼和浩特"这类 4 字）；② 逐字排除连接字/语气字，撞上就说明前面那截不是地名，
# 让正则从后面重新找，"北京到上海的天气"因此能正确抠出"上海"。
# 注意排除集里**不能有"都""那"**——成都、那曲是真城市名，排掉就抠不出来了。
_CITY_EXCLUDE = "到从在和跟与去的了过吗呢啊吧这就还"
_CITY_PATTERNS = (
    re.compile(
        r"((?:(?![" + _CITY_EXCLUDE + r"])[\u4e00-\u9fa5]){2,4}?)(?:市)?的?天气"
    ),
    re.compile(r"(?:在|去|到)([\u4e00-\u9fa5]{2,4}?)(?:玩|旅游|出差)"),
)

# 时间词不可能出现在城市名里，抠城市前先从整句里剔掉——"北京今天天气"不剔就会
# 抠出"北京今天"。注意这一步的整句替换**只服务于抠城市**：query / prompt 用的还是
# _strip_commands 的首尾剥离结果，不会被毁（I4 的分工）。
_TIME_NOISE = ("今天", "明天", "后天", "昨天", "前天", "现在", "当前", "目前",
               "这两天", "这几天", "最近", "一会儿", "等下")

# 句首的祈使/时间前缀（I4：只剥首尾，句子中间一个都不动）。
# 按长度从长到短剥，"查一下"才不会被"查"先拆散。
_HEAD_WORDS = (
    "帮我", "麻烦", "劳驾", "请问", "查询", "查一下", "查查", "查下", "看看",
    "看下", "问一下", "告诉", "一下", "查", "现在", "今天", "明天", "后天",
    "当前", "目前", "画一张", "画个", "画幅", "生成一张", "生成个", "来一张",
    "帮我画", "画一下", "画",
)

# 句尾的祈使后缀。刻意只有一个："一下"是唯一长在句尾还没信息的祈使词，
# 语气词（吧/呗/啊）不剥——剥了等于替她改用户原话，得不偿失。
_TAIL_WORDS = ("一下",)

_HEAD_SORTED = tuple(sorted(_HEAD_WORDS, key=len, reverse=True))

# 单字动词后面接这些字时，那是个正常的词不是祈使前缀（I4）。
# "画家今天在北京"以前被整句 replace 毁成"家今天在北京"，毁完的串还直接当了
# search query 发出去；"调查""动画"同理（那两个字不在句首，首尾剥离本身就已经救回）。
_COMPOUND_TAILS = {
    "画": "家作师展卷面片集法框意匠报册夹廊谜",
    "查": "证处封阅核勘",
}


def _strip_commands(text: str) -> str:
    """只剥句首/句尾的祈使词，剩下的大概率就是正主。

    以前对整句做 30 个子串 replace："画家"→"家"、"调查"→"调"、"动画"→"动"，
    毁完的串还被直接当 search query / image prompt 发出去（I4）。祈使词天然长在
    句首（"帮我查一下X"），剥首尾就够；剥不动的就原样留着——query 里多一个"查"字，
    比把"画家"毁掉代价小得多。
    """
    out = (text or "").strip()
    changed = True
    while changed and out:
        changed = False
        for word in _HEAD_SORTED:
            if not out.startswith(word):
                continue
            rest = out[len(word):]
            # 单字动词撞上正常词（画家/查证）就不剥，留着比毁掉强
            if len(word) == 1 and rest[:1] in _COMPOUND_TAILS.get(word, ""):
                continue
            out = rest.strip()
            changed = True
            break
        for word in _TAIL_WORDS:
            if len(out) > len(word) and out.endswith(word):
                out = out[:-len(word)].strip()
                changed = True
    return out


def _extract_city(text: str) -> str:
    """从"上海天气""在东京玩"这类句子里抠城市名，抠不到就空手而归。"""
    src = text or ""
    for word in _TIME_NOISE:
        src = src.replace(word, "")
    for pattern in _CITY_PATTERNS:
        m = pattern.search(src)
        if m:
            return m.group(1)
    return ""


def _build_rule_args(intent: str, user_text: str, fallback_city: str) -> dict:
    """按意图给规则通道凑工具参数。凑不出像样的就返回空 dict（= 这级别硬上）。"""
    cleaned = _strip_commands(user_text)
    if intent == "weather":
        city = _extract_city(cleaned) or (fallback_city or "").strip()
        if not city:
            # I4：抠不到城市就返回空参数，让 L3 的安检把这次执行拦下来。
            # 以前这里硬编默认"上海"——自信地答错城市比承认不知道更伤"像人"，
            # 也违反 ARCHITECTURE.md 第四节明写的"不瞎查默认城市"。
            # 追问归上层：pipeline._toolcall 已经在"没城市"时走
            # InfoGapCoordinator.ask("city")，能力层不许反向 import 调度层。
            return {}
        return {"city": city}
    if intent == "search":
        return {"query": cleaned or user_text, "top_k": 3}
    if intent == "image":
        return {"prompt": cleaned or user_text}
    if intent == "diary":
        return {}  # 写日记不需要参数，会话和日期运行时自己知道
    return {}


def _canonical_args(arguments: dict) -> str:
    """把参数字典压成可比较的字符串，给幂等账本当键用。

    键排序 + default=str：模型两次填同一组参数但顺序不同，也得算同一次调用；
    万一参数里混进不可序列化的东西，宁可降级成 repr 也不能让账本这一层抛异常
    （它站在防重复花钱的路上，自己不能先炸）。
    """
    try:
        return json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(arguments)


class ToolGuard:
    """工具调用前的安检：工具名存不存在、必填参数齐不齐。"""

    def check(self, spec: ToolCallSpec) -> tuple:
        """返回 (ok, 内部码)。

        内部码是**给模型看的系统级信号，不是台词**（C3）：全用中性代号
        （E_UNKNOWN_FN / E_MISSING_ARG），一个中文字都不带。以前这里返回
        「没有这个工具：X」「缺少必填参数：Y」，会被当成结果回填给模型，
        模型顺口就把"工具"说给了用户。这些码也绝不进 persona_wrap 的转述清单
        ——安检没过 = 这次调用根本没执行，压根不是"结果"（见 _run_native_loop）。
        """
        meta = services.get("tool_registry").get_meta(spec.tool_name)
        if meta is None:
            return False, f"E_UNKNOWN_FN:{spec.tool_name}"
        required = (meta.get("parameters") or {}).get("required") or []
        for key in required:
            if key not in (spec.arguments or {}):
                return False, f"E_MISSING_ARG:{spec.tool_name}.{key}"
        return True, ""


class ToolCallOrchestrator:
    """工具调用的总指挥，L1 挂了找 L3，一层层往下兜（L2 已删，见模块 docstring）。"""

    def run(
        self,
        user_text: str,
        intent: str,
        fallback_city: str = "",
        session_id: str = "",
        should_cancel=None,
    ) -> dict:
        """跑完两级降级，返回 {"results": [ToolCallResult...], "final_text": str}。

        final_text 只有 L1 通道才有：模型自己把答案说完了。
        session_id 是运行时上下文，注入式工具（比如写日记）靠它知道在给谁干活。
        should_cancel 是调用方注入的"该不该停"回调（返回 True 就尽早收工）。
        它只是个普通可调用对象，能力层不 import 调度层，避免越层依赖；
        真被取消时返回带 "cancelled": True 的结果，由上层决定怎么处理。
        """
        # I3：本条消息的幂等账本 + 已产出结果，从 L1 一路带到 L3。
        # 以前 L1 抛异常时 except: pass 把已经跑出来的 results 全丢掉，下一级重新
        # execute 一遍，L2 再抛又落 L3 再 execute ——一条消息最多三次 image_gen
        # （画图要钱）。现在这两个是**跨级共享的可变对象**，抛异常也带得下去：
        # 结果不丢，副作用不重跑。
        executed = set()
        results = []

        # L1：模型原生工具调用，模型自己决定调什么、调完自己组织答案
        if get_settings().llm_supports_tool_call:
            try:
                return self._run_native_loop(
                    user_text, session_id, should_cancel, executed, results
                )
            except Exception:
                pass  # 已执行的 results / executed 不丢，带着往 L3 走

        # L3：规则表硬上，参数靠正则抠，模型再怎么抽风都有底
        rule_name = RULE_TOOL_MAP.get(intent)
        if rule_name:
            args = _build_rule_args(intent, user_text, fallback_city)
            ok, _reason = ToolGuard().check(
                ToolCallSpec(tool_name=rule_name, arguments=args)
            )
            # 过不了安检就一个结果都不给：上层 persona_wrap 会落 no_result 兜底话。
            # 典型场景是天气抠不到城市（I4 已删掉硬编的"上海"）——拿空城市去查、
            # 或者瞎填一个城市，都比"我不知道你说哪儿"更伤。
            if ok:
                self._execute_once(rule_name, args, session_id, executed, results)
        return {"results": results, "final_text": ""}

    @staticmethod
    def _execute_once(tool_name: str, arguments: dict, session_id: str,
                      executed: set, results: list):
        """执行一次工具并记进幂等账本。被账本拦下返回 None（不产出新结果）。

        I3 两道闸：① 同样的 (名字, 规范化参数) 一条消息里只跑一次；
        ② 有副作用的（画图 / 写日记）**参数不同也不跑第二次**——ToolExecutor
        内部本来就有 max_retry 次重试，这里再重试等于重复花钱/重复写库。
        闸②正好兜住 CLAUDE.md 取舍 #5 记下的那笔代价：结果以 role:"user" 回填、
        call_id 被丢弃，模型看不出"哪条结果对应哪次调用"，4 轮循环里再要一次图
        完全可能。
        """
        key = (tool_name, _canonical_args(arguments))
        if key in executed:
            return None
        if tool_name in _SIDE_EFFECT_TOOLS and any(n == tool_name for n, _k in executed):
            return None
        executed.add(key)
        out = services.get("tool_executor").execute(tool_name, arguments, session_id)
        result = ToolCallResult(tool_name, out["status"], out["data"])
        results.append(result)
        return result

    def _run_native_loop(self, user_text: str, session_id: str = "", should_cancel=None,
                         executed: set = None, results: list = None) -> dict:
        """L1 循环：模型带工具列表补全，要调工具就执行后把结果喂回去，
        直到它给出最终回答或轮数用完。

        executed / results 由 run() 建好传进来（I3）：本函数中途抛异常时，
        已经跑出来的结果留在 results 里被 run() 带去 L3，不会被丢掉重跑。
        """
        registry = services.get("tool_registry")
        catalog = registry.catalog()
        cfg = load_app_config()["tools"]
        rounds = cfg.get("max_tool_loop_rounds", _MAX_LOOP_ROUNDS)
        executed = set() if executed is None else executed
        results = [] if results is None else results

        messages = [
            {
                "role": "system",
                "content": (
                    "你能查天气、搜资料、画图、写日记。需要哪样就直接去办，"
                    "办完拿到结果再回答他；办不成或者信息不够，就照你自己的性子"
                    "跟他说一句，别解释过程、别念任何代号。"
                ),
            },
            {"role": "user", "content": user_text},
        ]
        guard = ToolGuard()

        for _ in range(rounds):
            # 取消检查点：每一轮开跑前先看看这轮还该不该继续，别白烧模型调用
            if should_cancel is not None and should_cancel():
                return {"results": results, "final_text": "", "cancelled": True}
            resp = get_llm().chat_with_tools(messages, catalog)

            # 模型说完了，把它的答案带回去
            if resp["type"] == "final":
                return {"results": results, "final_text": resp.get("content", "")}

            # 模型要调东西：过安检再执行
            spec = ToolCallSpec(
                tool_name=resp.get("name", ""),
                arguments=resp.get("arguments") or {},
            )
            ok, reason = guard.check(spec)
            if not ok:
                # 安检没过 = 这次调用**根本没执行**，所以不进 results（不进转述清单）。
                # 内部码只回填给模型让它自己纠，一个字都不会到她嘴边（C3）。
                messages.append({"role": "assistant", "content": "（这步没成）"})
                messages.append({
                    "role": "user",
                    "content": (
                        f"（刚才那步没办成，内部码 {reason}。这是系统代号，别念给他听；"
                        "能补救就换个法子补救，不能就照你自己的性子回一句。）"
                    ),
                })
                continue

            result = self._execute_once(
                spec.tool_name, spec.arguments, session_id, executed, results
            )
            if result is None:
                # 幂等账本拦下的重复调用：结果上面已经回填过了，催它直接说话
                messages.append({
                    "role": "user",
                    "content": "（这件事你刚已经办过了，结果就在上面——别再办一次，直接跟他说。）",
                })
                continue

            # 结果用普通文本回填，不走 tool role，兼容 GLM 这类接口
            #
            # ===== 这是一个有意取舍，不是写漏了，别顺手"修"成标准写法 =====
            # 标准写法是 {"role": "tool", "tool_call_id": <id>, "content": ...}，
            # 但当前接的 GLM 接口不吃 role:"tool"，传过去会直接报错，
            # 所以这里把结果降级成一条 role:"user" 的普通文本塞回去。
            # **兼容只要求换 role，从来不要求文本里写"工具"两个字**（C3 的纠偏）：
            # 回填措辞是沉浸口径的「（你刚查到的：…）」，她才不会跟着说出"工具"。
            #
            # 代价（已知，接受）：
            #   1. 丢掉了 call_id。模型看不出"这条结果对应哪次调用"，
            #      多轮多工具时它可能重复调同一个，或把结果张冠李戴。
            #      ——重复调这一半现在由 I3 的幂等账本兜住（_execute_once），
            #      张冠李戴仍有可能，对现有场景（单用户、一轮基本一个）够用。
            #   2. llm_client.py:128 只取首个 tool_calls（resp["tool_calls"][0]），
            #      模型一次想调多个时，后面的会被丢掉。
            #
            # 将来要换成标准 tool role，动这三处：
            #   1. 这里：messages.append 改成 role:"tool" + tool_call_id；
            #   2. llm_client.py：chat_with_tools 要把 call_id 从响应里取出来往回传
            #      （现在 resp 里根本没有这个字段），并且别再只取 [0]；
            #   3. 确认目标模型真的支持 tool role —— 换成 OpenAI 系就可以，GLM 不行。
            messages.append({"role": "assistant", "content": "（去办了一下）"})
            messages.append({
                "role": "user",
                "content": f"（你刚查到的：{result.data}）据此跟他说，别解释过程。",
            })

        # 轮数用完模型还没给答案，结果交给上层转述
        return {"results": results, "final_text": ""}
