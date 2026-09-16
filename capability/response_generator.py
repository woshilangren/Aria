"""能力层 - 回复生成器

两条路：纯聊天直接拼消息调模型；工具跑完后把结果转述成人话。
转述必须带人设口吻，不能像客服念说明书。
"""

from shared.singletons import get_llm, services
from shared.types import FinalReply, PromptPackage, ToolCallResult
from tools.misc import ClockTool

# LLM 挂了时的兜底话（内置默认；persona_config.json 的 fallback_replies.llm 可覆盖）
_FALLBACK_REPLY = "……这会儿状态不太对，等一下再聊。"

# 没查到结果时的兜底话（fallback_replies.no_result 可覆盖）。
# 沉浸式措辞：她是"人"，嘴里不能出现"工具"这种词
_NO_RESULT_REPLY = "呃……这个我还真不知道。"

# 兜底话的"现说"指引（S12 媒体如实告知原则的延伸）：出状况时她用自己当下的
# 口吻承认，每次不一样——固定台词复读本身就是 AI 腔
_FALLBACK_INTENT = {
    "no_result": "他问的事你没弄明白——直说不知道，可以带点小情绪或不服气",
    "search": "你想给他查点东西但没查到有用的——承认这次没辙，别装知道",
    "image": "他让你弄张图但你没弄出来——说这次没成，别提任何工具字眼",
    "tool_call": "他想让你办的事没办成——直说这次没成，别提任何工具字眼",
    "review": "你刚才有句话没说出口——算了不说了，自然岔开",
    "blocked": "他碰了你不想聊的话题——按你的性子拒绝或岔开",
}


def fallback_line(kind: str, session_id: str = "", default: str = "") -> str:
    """兜底话的统一出口：能开口就现说一句（每次不一样），说不了落回固定台词。

    两层兜底：
    1. LLM 活着 → 以她的身份、按当下心情现说一句（绝不提工具/系统字眼）；
    2. 任何失败 → 人设自定义台词 → 内置默认。
    "llm" 类故障（模型本身挂了）直接跳过现说——那时生成必然失败，别白等。
    """
    if kind != "llm" and session_id:
        try:
            from capability import self_identity

            name = self_identity.display_name(session_id)
            rel = services.get("kv_store").read("relationship", session_id) or {}
            mood = rel.get("mood") or "平常"
            intent = _FALLBACK_INTENT.get(kind, "出了点小状况，自然承认")
            line = (get_llm().chat(
                [{"role": "user", "content": (
                    f"你是{name}，一个真实生活的年轻女孩，此刻心情偏「{mood}」。\n"
                    f"{intent}。一句话，30 字以内，口语，符合你此刻的心情；"
                    "绝不提工具、系统、搜索、网络这类字眼。只输出这一句。"
                )}],
                temperature=0.9,
                max_tokens=50,
            ) or "").strip()
            if line:
                return line[:60]
        except Exception:
            pass  # 现说失败落固定台词，兜底链不能断
    return _persona_fallback(kind, default or _FALLBACK_REPLY)


def _persona_fallback(kind: str, default: str) -> str:
    """优先取人设文件里自定义的兜底话，取不到落回内置默认。"""
    try:
        persona = services.get("kv_store").read("persona_config", "")
        custom = (getattr(persona, "fallback_replies", None) or {}).get(kind)
        if custom:
            return custom
    except Exception:
        pass
    return default


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
        return get_llm().chat(messages) or _persona_fallback("llm", _FALLBACK_REPLY)
    except Exception:
        return _persona_fallback("llm", _FALLBACK_REPLY)


def persona_wrap(
    results: list,
    user_text: str,
    initial_text: str = "",
    extra: str = "",
    session_id: str = "",
) -> FinalReply:
    """把工具跑出来的结果转述成角色的口吻。

    initial_text 是模型自己先说的版本，只当参考用，转述时重新润一遍。
    extra 是追加在系统提示词末尾的附加规矩（比如语音轮的情绪标注要求）。
    图片类结果顺手把图片路径带出去，上层好往页面塞图。
    session_id 透传给兜底话出口：没查到结果时她说的是"现说的"，不是复读。
    """
    if not results:
        return FinalReply(text=fallback_line("no_result", session_id, default=_NO_RESULT_REPLY))

    # 工具结果整理成清单，一行一个，转述的时候有据可依
    lines = []
    for item in results:
        lines.append(f"- 工具 {item.tool_name}（{item.status}）：{item.data}")
    tool_text = "\n".join(lines)

    # 画图工具跑成功了就把图片路径记下来，最终回复要带图
    image_path = ""
    for item in results:
        if (
            item.tool_name == "image_gen"
            and item.status == "ok"
            and isinstance(item.data, dict)
        ):
            image_path = item.data.get("path", "") or ""

    kv = services.get("kv_store")
    persona = kv.read("persona_config", "")

    system_prompt = (
        f"{persona.background_story}\n\n"
        "现在你刚用工具查完资料，把下面工具结果用自己的话告诉用户，"
        "不许暴露任何工具细节，80 字以内，语气要符合你的人设。"
        "如果结果里都是报错，就按你自己的性子认个错。"
    )
    # 工具链路没有 compose 的人设包，时间感知在这里单独补上，别问了天气忘了时辰
    clock = ClockTool()
    time_hint = f"【当前时间】{clock.now()}（{clock.period()}）"
    guidance = clock.time_guidance()
    if guidance:
        time_hint += f"\n{guidance}"
    system_prompt = f"{system_prompt}\n\n{time_hint}"
    # extra 是追加的附加规矩（比如语音轮的情绪标注要求），和 generate() 同一套约定
    if extra:
        system_prompt = f"{system_prompt}\n\n{extra}"
    user_prompt = f"【工具结果】\n{tool_text}\n\n【用户原话】\n{user_text}"
    if initial_text:
        user_prompt += f"\n\n【模型初步回答（仅参考，要改成你的口吻）】\n{initial_text}"

    try:
        text = get_llm().chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = (text or "").strip()
    except Exception:
        text = ""

    # 模型没给话就自己拼一段朴素的，起码信息不丢
    if not text:
        text = tool_text

    if image_path:
        return FinalReply(text=text, output_mode="image", image_path=image_path)
    return FinalReply(text=text)
