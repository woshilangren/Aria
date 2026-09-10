"""能力层 - 回复生成器

两条路：纯聊天直接拼消息调模型；工具跑完后把结果转述成人话。
转述必须带人设口吻，不能像客服念说明书。
"""

from shared.singletons import get_llm, services
from shared.types import FinalReply, PromptPackage, ToolCallResult
from tools.misc import ClockTool

# LLM 挂了时的兜底话（内置默认；persona_config.json 的 fallback_replies.llm 可覆盖）
_FALLBACK_REPLY = "……这会儿状态不太对，等一下再聊。"

# 工具没跑出来任何东西时的兜底话（fallback_replies.no_result 可覆盖）
_NO_RESULT_REPLY = "工具这边什么都没查到，换个问法试试？"


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


def generate(prompt: PromptPackage, user_text: str, extra: str = "", user_image: str = "") -> str:
    """纯聊天回复：人设提示词 + 短期记忆 + 用户这句话，喂给模型拿回复。

    extra 是追加在系统提示词末尾的附加规矩（比如语音轮的情绪标注要求）。
    user_image 是用户随消息上传的图片绝对地址；带图时把用户消息转成
    OpenAI 多模态数组格式，让多模态模型连图一起看。
    模型挂了就返回兜底话，别把异常漏到上层把整个对话弄崩。
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
    messages = (
        [{"role": "system", "content": system}]
        + list(prompt.context_messages or [])
        + [{"role": "user", "content": user_content}]
    )
    try:
        return get_llm().chat(messages) or _persona_fallback("llm", _FALLBACK_REPLY)
    except Exception:
        return _persona_fallback("llm", _FALLBACK_REPLY)


def persona_wrap(
    results: list,
    user_text: str,
    initial_text: str = "",
    extra: str = "",
) -> FinalReply:
    """把工具跑出来的结果转述成角色的口吻。

    initial_text 是模型自己先说的版本，只当参考用，转述时重新润一遍。
    extra 是追加在系统提示词末尾的附加规矩（比如语音轮的情绪标注要求）。
    图片类结果顺手把图片路径带出去，上层好往页面塞图。
    """
    if not results:
        return FinalReply(text=_persona_fallback("no_result", _NO_RESULT_REPLY))

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
