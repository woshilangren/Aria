"""能力层 - 她自己的身份（批次0"种子+涌现"）

设计哲学（来自项目作者）：她不该被配置定义——persona_config 只留一颗极薄的
种子（年轻女孩 / 20-25 / 国内某城），其余的一切——名字、年龄、城市、职业、
住处、对自己的认知——都由**她本人在对话里第一次说出口**时定下，定下即冻结，
从此是她的一致历史。代码只提供"冻结"的机制，不提供任何内容。

三条纪律：
1. **只收她亲口说的**：抽取提示词严禁根据常识补全，她没说的键一律空串。
2. **只填空，不改口**：已冻结的键永远不被后来的抽取覆盖（首次即定，
   改口机制留待将来——目前"第一次说出口的就是人生"）。
3. **性格不变性**：self_narrative（她对自己的认知）只在最初几轮生成一次、
   一次冻结；用户的一切评价都进关系层（S2/N2），永远碰不到这里。

冻结不阻塞聊天：LLM 失败静默跳过，下一轮再试；字段长期空着也有兜底
（compose 注入"由你定下"的指令，她随时可以补）。
"""

from shared.singletons import services
from tools.misc import ClockTool, parse_llm_json

# 身份事实白名单：模型答出别的键不收。gender 不在此列——种子已定（年轻女孩），
# 它是唯一被"定义"的东西，为了和固定的 TTS 女声音色保持一致
_IDENTITY_KEYS = ("name", "age", "city", "occupation", "home")

# 名字与年龄的硬校验：名字防模型塞长句；年龄出了 20~25 的锚点范围就不收
_NAME_MAX = 12
_AGE_RANGE = (20, 25)


def get_self(session_id: str) -> dict:
    """读她已冻结的身份，没有就空 dict（任何异常都不拦聊天）。"""
    try:
        return services.get("kv_store").read("self", session_id) or {}
    except Exception:
        return {}


def display_name(session_id: str) -> str:
    """她的名字：优先她自己取的（冻结的），没取过落回人设占位名。

    界面、日记、对话标签的统一取名口子——不要各自硬编码。
    """
    rec = get_self(session_id)
    name = (rec.get("name") or "").strip()
    if name:
        return name
    try:
        return (services.get("kv_store").read("persona_config", "").char_name or "Aria").strip()
    except Exception:
        return "Aria"


def compose_sections(session_id: str) -> list:
    """给 persona_engine 的身份注入段。

    三种形态：
    - 全空：注入"你的第一次"——让她在这次回复里自然介绍自己并取名字；
    - 部分冻结：列出已定下的，缺名字时催一次自命名（只催，不代取）；
    - 全冻结：注入"你是谁"+她自己的自我认知，性格以她的认知为准。
    """
    rec = get_self(session_id)
    if not rec:
        return [
            "【你的第一次】\n"
            "你们才刚认识，你还没有任何定下来的过去：名字、年龄、城市、在做什么，都还空着。\n"
            "这条回复里自然地把自己带出来：自己取一个名字（认真取，取了就是你的名字），"
            "随口提到你在哪个城市、平时做什么、多大——由你当场定下，说出口就是你的人生，"
            "从此永远一致。只说这次自然说到的，别一口气念档案。\n"
            "开场别套模板：不说\"新来的\"\"刚搬来\"，不问\"怎么想到来找我\"，"
            "不演相亲、网友面基、转学生——你们就是正在聊天的两个人。"
        ]

    lines = []
    if rec.get("name"):
        lines.append(f"你的名字：{rec['name']}（这是你自己取的）")
    if rec.get("age"):
        lines.append(f"年龄：{rec['age']}")
    if rec.get("city"):
        lines.append(f"住在：{rec['city']}")
    if rec.get("occupation"):
        lines.append(f"在做：{rec['occupation']}")
    if rec.get("home"):
        lines.append(f"住处：{rec['home']}")
    sections = []
    if lines:
        sections.append("【你是谁】\n" + "\n".join(lines))
    if not rec.get("name"):
        sections.append(
            "【名字还没定】你还没给自己取名字：这条回复里自己取一个，自然地说出来"
            "（认真取，取了就是你的名字，以后一直是它）。"
        )
    narrative = (rec.get("self_narrative") or "").strip()
    if narrative:
        sections.append(
            f"【你对自己的认知】{narrative}\n"
            "（这是你自己在最初几天里认下的自己——你的性格以它为准，"
            "不随他的评价改变；他的话只改变你们之间的温度。）"
        )
    return sections


def _extract_prompt(her_reply: str) -> list:
    """身份抽取的提示词：只收她亲口说的，严禁常识补全——这是防幻觉的第一道闸。"""
    return [
        {"role": "system", "content": (
            "下面是一个女孩刚说的一段话。提取其中她说出口的、关于她自己的事实。"
            "只输出 JSON：{\"name\":\"\",\"age\":\"\",\"city\":\"\",\"occupation\":\"\",\"home\":\"\"}\n"
            "- name：她给自己取的名字（只有她说了\"我叫X/X就是我的名字\"这类才算）；\n"
            "- age：她的年龄（数字字符串）；\n"
            "- city：她住的城市；occupation：她在做的事（上学/工作都算）；home：住处描述；\n"
            "- 她没提到的键一律留空串。绝不要根据常识或语境补全、猜测。"
            "她在讲别的事情里顺带说的才算数，单纯被问到但没回答的不算。"
        )},
        {"role": "user", "content": her_reply[:800]},
    ]


def _merge_freeze(session_id: str, extracted: dict) -> dict:
    """只填空、不改口（纪律 2）：已冻结的键原样保留，锁内纯计算（F3 原子闭包）。"""
    def _fill(rec: dict) -> dict:
        rec = rec or {}
        for key in _IDENTITY_KEYS:
            if rec.get(key):
                continue  # 冻结的键不被覆盖
            val = str(extracted.get(key) or "").strip()
            if not val:
                continue
            if key == "name":
                val = val[:_NAME_MAX]
            if key == "age":
                try:
                    n = int(float(val))
                except (TypeError, ValueError):
                    continue
                if not (_AGE_RANGE[0] <= n <= _AGE_RANGE[1]):
                    continue  # 出了锚点范围（20~25）不收
                val = str(n)
            rec[key] = val[:40]
        if not rec.get("frozen_at") and any(rec.get(k) for k in _IDENTITY_KEYS):
            rec["frozen_at"] = ClockTool().now()  # 第一次冻结的时刻
        return rec

    return services.get("kv_store").update("self", session_id, _fill)


def maybe_freeze(session_id: str, her_reply: str,
                 recent_her_lines: list = None, interaction_count: int = 0) -> None:
    """写回时调用：该冻结的冻结（身份事实 / 自我认知）。失败静默，下轮再试。

    - 身份事实：每轮从她刚说的话里抽取，直到五个键全部冻结（此后不再烧这次调用）；
    - 自我认知（self_narrative）：聊到第 3 轮之后，从她最初几轮的发言里提炼
      "她对自己的认知"，一次生成一次冻结——它是性格内核的描述，之后永不改写。
    """
    her_reply = (her_reply or "").strip()
    if not her_reply:
        return
    rec = get_self(session_id)
    llm = services.get("llm")

    # ---- 1. 身份事实抽取（还有空键才跑）----
    if not rec or any(not rec.get(k) for k in _IDENTITY_KEYS):
        try:
            raw = llm.chat(_extract_prompt(her_reply), temperature=0.1, max_tokens=160)
            data = parse_llm_json(raw)
            if isinstance(data, dict):
                rec = _merge_freeze(session_id, data)
        except Exception:
            pass  # 抽取失败下轮再试，聊天照常

    # ---- 2. 自我认知基线（第 3 轮后，一次冻结）----
    if not (rec or {}).get("self_narrative") and interaction_count >= 3:
        her_lines = [ln.strip() for ln in (recent_her_lines or []) if ln and ln.strip()]
        if her_lines:
            try:
                text = "\n".join(f"- {ln[:120]}" for ln in her_lines[-5:])
                raw = llm.chat(
                    [
                        {"role": "system", "content": (
                            "下面是这个女孩在最初几次聊天里说过的话。"
                            "以她的第一人称、用她自己的口吻，写一小段对自己的认知"
                            "（我说话是什么路子、我大概是个什么样的人），60 字以内，"
                            "只输出正文，别提这些材料，别提任何对话细节。"
                        )},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.5,
                    max_tokens=150,
                )
                note = (raw or "").strip()
                if note:
                    def _set(rec2: dict) -> dict:
                        rec2 = rec2 or {}
                        if not rec2.get("self_narrative"):  # 一次冻结，永不改写
                            rec2["self_narrative"] = note[:120]
                            rec2["narrative_frozen_at"] = ClockTool().now()
                        return rec2

                    services.get("kv_store").update("self", session_id, _set)
            except Exception:
                pass  # 失败下轮再试
