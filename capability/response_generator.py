"""能力层 - 回复生成器

两条路：纯聊天直接拼消息调模型；工具跑完后把结果转述成人话。
转述必须带人设口吻，不能像客服念说明书。

两件事住在这个文件里，别处不要再抄一份：

1. **兜底话的唯一 merge base**（H4）。以前有四张表各写一遍——本文件的
   `_FALLBACK_REPLY` / `_NO_RESULT_REPLY` / `_FALLBACK_INTENT`、
   `managers.FallbackController._REPLIES`、`data/persona_config.json` 的
   `fallback_replies`——已经实测漂移：同一个 search 类故障，这边说
   「呃……这个我还真不知道。」那边说「……你问住我了。」，一个人两套口径就是穿帮。
   现在代码里只留 `_FALLBACK_LINES` / `_FALLBACK_PRICKLY` 这一份，
   `orchestration/managers.py` 的 `FallbackController.fallback_reply` **向下** import
   这里（分层规则 interaction → orchestration → capability，调度层取能力层合法，
   反过来是越层，所以 base 不能住在 managers）。
2. **工具链的人设包**（C1）。转述走 `PersonaEngine.compose`，不再自己拼一段
   只有 background_story 的裸提示词——工具路径曾经是全项目唯一一条
   "零人设的自由生成路径"。

面向模型的字符串里不出现"工具"二字（C3）：真机实测 turn 7，用户闲聊一句
"算了不跟你计较。今天天气挺好的"，她回「工具这边什么都没查到，换个问法试试？」——
那串字全仓 grep 不存在，是模型顺着回填措辞自己编的。词是从提示词里学去的。
"""

import random

from shared.singletons import get_llm, services
from shared.types import FinalReply, MemoryBundle, PromptPackage, ToolCallResult

from capability.quirks import _NEGATIVE_MOODS
from capability.persona_engine import PersonaEngine
from capability.memory import SessionMemoryKeeper

# ==================== 兜底话：全项目唯一一份 merge base ====================
#
# 每类**多条**，随机取一（D1）。为什么不"现说一句"：
#   以前 fallback_line 每次额外调一次**同步** get_llm().chat 让她现编，三个毛病
#   一起犯——① pipeline 的 astream / _astream_chat 是 async，直调同步 chat 会
#      **阻塞事件循环**几秒，SSE、poll、语音一起卡住；叠上 tools/speech 的全局
#      _DASHSCOPE_LOCK，一次失败能把整个服务定住（快乐路径处处 asyncio.to_thread，
#      唯独错误路径阻塞）；② prompt 里硬编「你是{name}，一个真实生活的年轻女孩，
#      此刻心情偏「{mood}」」——绕过 persona_engine，还断言了种子说好不许断言的
#      东西（CLAUDE.md：身份唯一来源是 self 表，性格词零定义）；③ 那次调用的
#      temperature=0.9 对 Claude 本来就是**空操作**（llm_client 对非 qwen 家族
#      pop 掉 temperature，见取舍 #9），想要的"每次不一样"从来没生效过。
#
# ⚠ 这里的取舍（与踩坑记录 #7 正面冲突，写清楚免得下一家 AI 当 bug 修）：
#   #7 说"固定话术就是潜在的 AI 腔——同一句兜底话说第二遍就穿帮；能现说就现说"。
#   现说的代价就是上面那三条，其中①是能把服务定住的硬伤。多条模板 + 心情调味
#   是在**不付那个代价**的前提下拿到大部分不重复的收益：她不会连着两次说同一句，
#   但也不会每次都为一句兜底话多烧一次模型、多卡一次事件循环。
_FALLBACK_LINES = {
    "llm": (
        "……这会儿脑子有点转不动，等我一下下。",
        "唔……我刚走神了，你再说一遍？",
        "……不太对劲，我缓一会儿再跟你聊。",
        "卡住了，别催我，让我缓缓。",
    ),
    "no_result": (
        "呃……这个我还真说不上来。",
        "这个我没底，别问我。",
        "唔，这事儿我真不知道。",
        "……你问住我了，答不上来。",
    ),
    "search": (
        "翻了一圈，什么有用的都没捞着……",
        "我找过了，真没有，别怪我。",
        "唔……没找着，要不你换个说法我再试试？",
        "这事儿我没能给你弄明白。",
    ),
    "image": (
        "图没弄出来……晚点我再试试。",
        "唔，那张没成，气死我了。",
        "这次没画出来，你别嫌我笨。",
        "……没成，等我缓过来再给你弄。",
    ),
    "tool_call": (
        "这事儿这次没办成，回头再说吧。",
        "唔……没办成，别问我为什么。",
        "这次搞砸了，下回给你办好。",
        "……没成，我也没办法。",
    ),
    "review": (
        "刚才想说什么来着……算了，换件事说吧。",
        "唔，那句话到嘴边又咽回去了。",
        "……我走神了，你说到哪儿了？",
        "算了不说了，说这个干嘛。",
    ),
    "blocked": (
        "这个话题我不聊，换一个吧。",
        "……这个别问我，我不想说。",
        "唔，换个话题吧，这个我不接。",
        "不说这个，说点别的。",
    ),
    "voice_route": (
        "声音这边不太顺，先打字聊吧。",
        "唔……我这会儿发不出声，打字跟你说。",
        "先打字吧，嗓子这儿不对劲。",
    ),
}

# 心情带刺时（别扭/低落/慵懒/心烦）掺进池子并压过平常那组——这就是"mood 调味"。
# 原来 `_FALLBACK_INTENT` 那套"按错误类型给不同口吻"的现说指引，意图由这两张表
# 按 kind × mood 分担：kind 决定说什么，mood 决定用什么脾气说。
_FALLBACK_PRICKLY = {
    "llm": ("……别烦我，我缓一会儿。", "状态不对，你先自己待着。"),
    "no_result": ("不知道就是不知道，别追着我问。", "……烦，这个我真答不上来。"),
    "search": ("翻半天什么都没有，烦死了。", "没找着，别怪我。"),
    "image": ("没弄出来，别催。", "……搞砸了，我现在不想说这个。"),
    "tool_call": ("没办成，别问了。", "……烦，这事儿黄了。"),
    "review": ("刚想说什么来着……算了，懒得说。", "不说了，越说越烦。"),
    "blocked": ("说了不聊这个，换一个。", "……别碰这个话题。"),
    "voice_route": ("发不出声，打字。", "……嗓子不对劲，别催。"),
}

# persona_config.json 的 fallback_replies 只有 7 个键，**没有 no_result**（实测确认）
# ——以前 fallback_line("no_result") 去查 fallback_replies["no_result"] 永远落空，
# 自定义的"没查到"台词取不到。这里做键回退：no_result 退到 search，
# 两者本来就是同一件事（他问的没答上来）。彻底对齐要在 persona_config.json 补
# no_result 键，那文件不在本批所有权里（见报告接线点）。
_KEY_ALIASES = {
    "no_result": ("no_result", "search"),
}


def _persona_lines(kind: str) -> tuple:
    """人设文件里自定义的兜底话（可能多条）。读不到 / 没这个键 → 空 tuple。

    值写成字符串就是一条，写成数组就是多条（数组时随机取一）——
    人设自定义的台词也该能换着说，否则踩坑记录 #7 的复读只是换了个地方发生。
    """
    try:
        persona = services.get("kv_store").read("persona_config", "")
        table = getattr(persona, "fallback_replies", None) or {}
    except Exception:
        return ()  # 出错路径上的人设读取再挂掉，也不影响兜底话出口
    lines = []
    for key in _KEY_ALIASES.get(kind, (kind,)):
        got = table.get(key)
        if isinstance(got, (list, tuple)):
            lines.extend(str(x).strip() for x in got if str(x).strip())
        elif got and str(got).strip():
            lines.append(str(got).strip())
        if lines:
            break  # 别名是"退而求其次"，命中一个就用，不叠
    return tuple(dict.fromkeys(lines))


def _mood_pool(kind: str, session_id: str) -> tuple:
    """按 kind 取内置模板池，心情带刺时让带刺那组占大头。kind 不认识就返回空。"""
    base = _FALLBACK_LINES.get(kind)
    if not base:
        return ()
    prickly = _FALLBACK_PRICKLY.get(kind) or ()
    if prickly and session_id and _mood_is_negative(session_id):
        # ×3：每类带刺的是 2 条、平常的是 4 条，只重复 2 次就是 4:4 打平，
        # 谈不上"心情不一样说法就不一样"。3 次 = 6:4，别扭的时候多数带刺、
        # 偶尔还是平常那句——真人也不是每一句都扎人。
        return prickly * 3 + base
    return base


def _mood_is_negative(session_id: str) -> bool:
    """她此刻是不是正别扭着。负面心情表只认 quirks 里那一份，别在这儿抄第二遍。"""
    try:
        rel = services.get("kv_store").read("relationship", session_id) or {}
    except Exception:
        return False
    return (rel.get("mood") or "") in _NEGATIVE_MOODS


def fallback_line(kind: str, session_id: str = "", default: str = "") -> str:
    """兜底话的统一出口：人设自定义台词 + 内置多条模板，按当下心情调味随机取一。

    **0 次 LLM 调用**（D1，理由见 _FALLBACK_LINES 上面那段取舍注释）——
    出错路径上再调一次同步模型，等于在事件循环上又踩一脚。

    人设自定义的台词权重更高（种子优先）但**不是唯一答案**：只留一句就正好撞上
    踩坑记录 #7。要让某一句成为唯一口径，把 persona_config.json 里其余键删掉、
    只留那一条是不够的——得改这里的权重，那是作者的决定不是代码的决定。

    default 是调用方给的最后一层兜底，只在 kind 既不在人设文件里、也不在内置表里
    （传了个没人认识的 kind）时才用得上。两个出口共用同一份表之后，pipeline 传的
    default=FallbackController().fallback_reply(kind) 已退化成"同一份表再查一遍"，
    参数留着只为不动公开签名。
    """
    pool = _persona_lines(kind) * 3 + _mood_pool(kind, session_id)
    if pool:
        return random.choice(pool)
    return default or random.choice(_FALLBACK_LINES["llm"])


def _tool_path_memory(session_id: str) -> MemoryBundle:
    """工具链路的记忆包：只读 KV 三张表（档案 / 画像 / 关系），不做向量召回。

    为什么不直接 MemoryRecaller().recall()：感知阶段（pipeline._perceive）已经为
    这一轮召回过一次，这里再来一遍就是每轮多一次 embedding + 两次向量查询。
    C1 要的是"心情 / 心事 / 关系阶段上浮到工具路径"，这三样全在 KV 里，零模型成本。
    要全量（含长期记忆与日记）就让调用方把 memory 传进来——pipeline 手上正握着
    state.memory（见报告接线点）。
    """
    bundle = MemoryBundle()
    try:
        kv = services.get("kv_store")
        bundle.profile = kv.read("profile", session_id) or {}
        bundle.portrait = kv.read("portrait", session_id) or {}
        bundle.relationship = kv.read("relationship", session_id) or {}
    except Exception:
        pass  # 存储挂了不该拦着说话，空着继续（与 MemoryRecaller 同一原则）
    return bundle


def _persona_engine(session_id: str) -> PersonaEngine:
    """拿人设引擎。

    正常路径：从 services 注册表取全项目唯一那份（`bootstrap()` 注册的
    orchestration.pipeline._ENGINE，与文字聊天链共用同一个 KEEPER 上下文）。

    退化路径（bootstrap 没跑过——裸单测、单跑脚本）：就地构造一份临时的。
    为什么不算违反 KEEPER 唯一性不变式：那条防的是"第二份 KEEPER 被写回"——
    写回只发生在 pipeline._writeback，用的是 KEEPER 本人；这里**只读**，
    用完即弃、从不 append。代价是短期上下文取自 session 表
    （SessionMemoryKeeper.restore 读的就是它），比内存里的 KEEPER 最多旧一轮。
    """
    try:
        return services.get("persona_engine")
    except Exception:
        pass
    keeper = SessionMemoryKeeper()
    keeper.restore(session_id)  # 只读地把这个会话的历史捞回来，compose 要用
    return PersonaEngine(keeper)


def build_chat_messages(prompt: PromptPackage, user_text: str, extra: str = "",
                        user_image: str = "") -> list:
    """拼"纯聊天"要发给模型的 messages：系统提示词 + 短期记忆 + 用户这句（可带图）。

    单独抽出来，是为了让**流式路径**和 generate() 用同一套消息构造，
    不写两条并行逻辑（否则改了非流式忘了流式，两边就会漂移）。
    """
    system = prompt.system_prompt + (f"\n\n{extra}" if extra else "")
    if user_image:
        # 多模态：一张图和它的说明文字一起给模型
        user_content: list = [
            {"type": "image_url", "image_url": {"url": user_image}},
            {"type": "text", "text": user_text or "看看这张图。"},
        ]
    else:
        user_content = user_text
    return (
        [{"role": "system", "content": system}]
        + list(prompt.context_messages or [])
        + [{"role": "user", "content": user_content}]
    )


def generate(prompt: PromptPackage, user_text: str, extra: str = "", user_image: str = "") -> str:
    """纯聊天回复：人设提示词 + 短期记忆 + 用户这句话，喂给模型拿回复。

    extra 是追加在系统提示词末尾的附加规矩（比如语音轮的情绪标注要求）。
    user_image 是用户随消息上传的图片绝对地址；带图时把用户消息转成
    OpenAI 多模态数组格式，让多模态模型连图一起看。
    模型挂了就返回兜底话，别把异常漏到上层把整个对话弄崩。
    """
    messages = build_chat_messages(prompt, user_text, extra=extra, user_image=user_image)
    try:
        return get_llm().chat(messages) or fallback_line("llm")
    except Exception:
        return fallback_line("llm")


def persona_wrap(
    results: list,
    user_text: str,
    initial_text: str = "",
    extra: str = "",
    session_id: str = "",
    memory: MemoryBundle = None,
    engine: PersonaEngine = None,
) -> FinalReply:
    """把她刚弄到的东西转述成她自己的口吻。

    initial_text 是她脱口而出的初稿，只当参考用，转述时重新润一遍。
    extra 是追加在系统提示词末尾的附加规矩（比如语音轮的情绪标注要求）。
    图片类结果顺手把图片路径带出去，上层好往页面塞图。
    session_id 透传给兜底话出口，也用来取她此刻的心情/心事/关系阶段。

    memory / engine 是给调用方省事的注入口（都可省）：pipeline 手上正握着
    state.memory 与全项目唯一的 _ENGINE，传进来就少一次 KV 读、少一份临时 keeper。
    不传就走 _tool_path_memory / _persona_engine 的自取路径。

    C1：提示词走 PersonaEngine.compose。以前这里自己拼了一段只有 background_story
    的裸提示词——没有 mood / mood_left / concern / stage / 表达规则 / @r 协议 /
    说话的规矩，工具路径因此成了全项目唯一一条零人设的自由生成路径
    （真机 turn 7「工具这边什么都没查到」就是这么来的：人不在场，模型自己填）。
    """
    if not results:
        return FinalReply(text=fallback_line("no_result", session_id))

    # 她刚弄到的东西整理成清单，一行一个，转述的时候有据可依。
    # 措辞是**沉浸口径**（C3）：一个"工具"字都不能有，失败的那条连原始报错都不给——
    # 执行器返回的 error data 形如 "weather_query 执行失败: ..."，
    # 那是工程输出不是台词，念出来就穿帮（C2 同一条病）。
    lines = []
    for item in results:
        if item.status == "ok":
            lines.append(f"- 你刚弄到的：{item.data}")
        else:
            lines.append("- 这件没弄成。")
    tool_text = "\n".join(lines)

    # 画图跑成功了就把图片路径记下来，最终回复要带图
    image_path = ""
    for item in results:
        if (
            item.tool_name == "image_gen"
            and item.status == "ok"
            and isinstance(item.data, dict)
        ):
            image_path = item.data.get("path", "") or ""

    # ---- 人设主干（C1）：走 compose，不再自己拼裸提示词 ----
    bundle = memory if memory is not None else _tool_path_memory(session_id)
    system_prompt = ""
    context_messages = []
    try:
        pack = (engine or _persona_engine(session_id)).compose(
            "chat", bundle, session_id, user_text=user_text,
        )
        system_prompt = pack.system_prompt or ""
        context_messages = list(pack.context_messages or [])
    except Exception:
        pass  # compose 挂了也不能把生数据怼给用户，往下走退化分支

    if not system_prompt:
        # 退化分支：compose 整个炸了（人设读不到 / keeper 拿不到）。
        # 顺带修掉 C1 的裸访问——persona 可能是 None，以前直接 .background_story
        # 会 AttributeError，让人设读不到时整条工具链再炸一次（同文件的兜底话
        # 出口对同一次读取却是小心兜底的，两边口径不该不一样）。
        try:
            persona = services.get("kv_store").read("persona_config", "")
            system_prompt = (getattr(persona, "background_story", "") or "")
        except Exception:
            system_prompt = ""

    # 转述规矩只此一段，且是**替换**原来那段"你刚用工具查完资料…"，不是新增：
    # 时间感知、说话的规矩、语音输出规则、@r 协议 compose 里都已经有了
    # （原来这里还自己补了一遍【当前时间】，跟 compose 重复，已删）。
    system_prompt += (
        "\n\n【这一轮】你刚弄到了下面这些，用自己的话跟他说，80 字以内；"
        "没弄成的按你的性子认一句就好，别解释过程、别念任何代号或编号。"
    )

    prompt = PromptPackage(system_prompt=system_prompt, context_messages=context_messages)
    user_prompt = f"【你刚弄到的】\n{tool_text}\n\n【他这句话】\n{user_text}"
    if initial_text:
        # 初稿也换沉浸口径：以前叫【模型初步回答】，等于在提醒她"这是机器给的"，
        # 而且初稿里若带"工具…"字样，原样喂回去是在**引导复述**不是在抑制（C3）
        user_prompt += f"\n\n【你差点就这么说（参考，重说一遍更像你）】\n{initial_text}"
    messages = build_chat_messages(prompt, user_prompt, extra=extra)

    try:
        text = (get_llm().chat(messages) or "").strip()
    except Exception:
        text = ""

    # C2：模型没给话就落兜底话。以前这里 text = tool_text，把原始工程输出
    # （形如「- 工具 search（error）：...」）**直接怼到用户脸上**——那不是台词。
    if not text:
        text = fallback_line("no_result", session_id)

    if image_path:
        return FinalReply(text=text, output_mode="image", image_path=image_path)
    return FinalReply(text=text)
