"""能力层 - 她自己的身份（批次0"种子+涌现"，批次 G 披露预算修订）

设计哲学（来自项目作者）：她不该被配置定义——persona_config 只留一颗极薄的
种子（年轻女孩 / 20-25 / 国内某城 / 名字留白），其余的一切——名字、年龄、
城市、职业、住处、对自己的认知——都由**她本人在对话里说出口**时定下。
代码只提供机制，不提供任何内容。

G 批次的核心教训（12 轮真机 turn 1 实测）：她一句话交代了名字+城市+职业+年龄，
全部焊死——"认识她"这件事在第 1 轮就花光了。旧版靠提示词里一句
"只说这次自然说到的，别一口气念档案"去拦，**它输给了同一页递过去的四键清单**
（踩坑记录：prompt 是软约束，要机制不要叮嘱）。所以本版的纪律全部机制化：

1. **只收她亲口说的**：抽取提示词严禁根据常识补全，她没说的键一律空串。
2. **披露预算（G1-G4）**：每轮允许冻结的键由关系 stage 决定（初识=名字、
   熟悉=+年龄、亲近=+城市、挚友=+职业住处），闸门在 _merge_freeze 的
   allowed_keys 里——超预算的键**直接丢弃不冻结**，等她到了阶段亲口再说。
   compose_sections 只递"下一个已解锁的槽"，绝不列全部五键的清单。
3. **改口一次即历史（G5）**：每个键允许一次改口，改口本身作为历史记进
   revisions（"被记录下来的改口比一个不可变的错值更像人"）；第二次拒绝。
   接线点：revision 事件还应作为一条长期记忆落库（见 maybe_freeze 的 TODO）。
4. **越界年龄不静默丢弃（G6）**：种子锚点（20-25）外的年龄落 age_rejected
   标记 + 注入一次改口引导——种子不许无声否决她亲口说的话（单一真相源）。
5. **性格不变性 + 底色慢冻结（G7）**：self_narrative 要 ≥20 句素材、
   连续两轮独立生成高度一致才冻结（初期呈现的是社交面，不是本质——伪装性）；
   与她自己后来的话明显矛盾时可重议一次，重议入账。用户的一切评价仍只进
   关系层（S2/N2），永远碰不到这里。

冻结不阻塞聊天：LLM 失败静默跳过，下一轮再试；字段长期空着也有兜底
（compose 注入"由你定下"的指令，她随时可以在预算内补）。
"""

import hashlib
import re

from shared.singletons import services
from tools.misc import ClockTool, parse_llm_json

# 身份事实白名单：模型答出别的键不收。gender 不在此列——种子已定（年轻女孩），
# 它是唯一被"定义"的东西，为了和固定的 TTS 女声音色保持一致
_IDENTITY_KEYS = ("name", "age", "city", "occupation", "home")

# 名字与年龄的硬校验：名字防模型塞长句；年龄出了 20~25 的锚点范围不冻结，
# 但也不再静默丢弃——落 age_rejected 标记，让她改口一次（G6）
_NAME_MAX = 12
_AGE_RANGE = (20, 25)

# ---- 披露预算（G1-G4）：stage 决定"这一轮说到自己时允许定下哪些键" ----
# stage 取值来自 capability/memory.py 的 RelationshipTracker._STAGES
# （初识 0-19 / 熟悉 20-44 / 亲近 45-69 / 挚友 70+ 亲密度）。预算是**累计前缀**
# 而不是"本轮新增 N 个"：真人关系里，名字在初识就交换，年龄/城市要到熟起来
# 才自然聊到，职业与住处是深入交往才互相知道的事——预算曲线跟着这条常识走。
# 每轮 compose 只递"下一个槽"，所以前缀宽度同时就是节奏上限：一轮最多定下一件事。
_STAGE_RANK = {"初识": 0, "熟悉": 1, "亲近": 2, "挚友": 3}
_STAGE_BUDGET = {0: ("name",),
                 1: ("name", "age"),
                 2: ("name", "age", "city"),
                 3: ("name", "age", "city", "occupation", "home")}

# "他上一句问过什么"的粗匹配（G1）：初识阶段名字槽的钥匙——只有他问起，
# 她这轮才报名字。宁可漏判（她晚一轮再定名）不可误判（又变成第一轮倒档案）。
_ASK_PATTERNS = {
    "name": re.compile(r"叫什么|叫啥|名字|怎么称呼|如何称呼|你是哪位|你是谁"),
    "age": re.compile(r"多大|几岁|多少岁|年龄|岁数|年纪|哪年"),
    "city": re.compile(r"哪个城市|哪座城|在哪里|在哪儿|在哪住|哪里人|哪儿人|坐标|哪个地方"),
    # 职业：只认工作/学业语境的问法——裸的"做什么/干什么"会误伤"在干嘛呢"这种日常寒暄
    "occupation": re.compile(r"做什么工作|干什么工作|做什么的|干什么的|什么职业|哪行|还是学生|在上学|在读书|在哪上班"),
    "home": re.compile(r"住哪|住在哪|家在哪|住的地方|住处|房子|宿舍|租的"),
}

# 改口预算（G5）：每个键一生只许改口一次；第二次拒绝——改口成了历史才像人，
# 无限改口就只是数据抖动。narrative 的"重议"共用同一本账（键名 self_narrative）。
_MAX_REVISIONS_PER_KEY = 1

# ---- 自我认知（底色）的冻结门槛（G7）----
# 旧版第 3 轮就用 ≤5 句素材现编一段并宣布永不改——那是"安装人格"，违反宪法
# "特质涌现只能发现不能安装"。作者的洞察（计划与设计 3.4）：初期她呈现的是
# 社交面（伪装性），底色要相处久了才显。所以素材窗口扩到 20 句，且要求
# 连续两轮独立生成高度一致（证据跨时间稳定）才冻结；不一致就再等几轮。
_NARRATIVE_MIN_LINES = 20
_NARRATIVE_SIM_THRESHOLD = 0.55


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


def _get_stage(session_id: str) -> str:
    """当前关系阶段。读不到一律按"初识"——最紧的预算，闸门侧宁可少放不多放。"""
    try:
        rel = services.get("kv_store").read("relationship", session_id) or {}
        return str(rel.get("stage") or "初识")
    except Exception:
        return "初识"


def _asked_identity_keys(user_text) -> set:
    """他上一句问到了她身份的哪些键。

    user_text=None 表示调用方**还没接线**（旧调用方）——按"问过"放行，行为与
    接线前一致，不让闸门在接线前把名字也锁死；接线后传真实文本（哪怕空串），
    空串就是"没问"，闸门才真正生效。接线点：orchestration/pipeline.py 的
    _bg_freeze 调 maybe_freeze 时传 user_text；persona_engine.compose 调
    compose_sections 时传 user_text。
    """
    if user_text is None:
        return set(_IDENTITY_KEYS)
    text = str(user_text)
    return {k for k, pat in _ASK_PATTERNS.items() if pat.search(text)}


def _disclosure_gate(rec: dict, stage: str, asked_keys: set) -> set:
    """披露预算的闸门（G1-G4）：返回这一轮**允许冻结**的身份键集合。

    三条规则（全部机制，不靠提示词）：
    1. stage 前缀之外的键一律不开——初识只认识名字，住处是挚友才知道的事；
    2. 初识阶段的名字还要"他上一句问过"才开——没被问就自报家门是档案倾泻；
    3. 已经亲口说过（disclosed）但被预算拦下的键**不主动回头补冻**——丢弃是有意的：
       等它到了阶段，她自然还会再说到，那时才是"挣来的"披露；唯一例外是他直接
       问起（问起 = 自然需要的时候，重开该槽，不然预算就成了失忆）。
    例外（G6）：age_rejected 挂着时 age 无视预算开放——那是她在改口，不是新披露。
    """
    rank = _STAGE_RANK.get(stage, 0)
    budget = _STAGE_BUDGET.get(rank, _STAGE_BUDGET[0])
    disclosed = set((rec or {}).get("disclosed") or [])
    allowed = set()
    for key in budget:
        if (rec or {}).get(key):
            continue  # 已冻结的键不占本轮预算（改口走 _merge_freeze 的 revisions 分支）
        asked = key in asked_keys
        if key == "name" and rank == 0 and not asked:
            continue  # 初识 + 他没问名字 → 名字槽不开
        if key in disclosed and not asked:
            # 说过但被预算拦下的：不主动回头补冻——等阶段到了她亲口再说。
            # 但他**直接问起**就是"自然需要的时候"（种子原文），重开这个槽：
            # 否则他问"你叫什么来着"她反而答不上来，预算就成了失忆。
            continue
        allowed.add(key)
        break  # 一轮最多开一个槽：这就是预算的执行形式
    if (rec or {}).get("age_rejected") and not (rec or {}).get("age"):
        allowed.add("age")  # 改口通道（G6）：越界年龄被拒后，允许她把年龄改到锚点内
    return allowed


def _revision_allowed_keys(rec: dict, stage: str) -> set:
    """已冻结键里"当前阶段允许改口"的集合（G5 与 G1-G4 的交叉约束）：
    改口不能变成绕过披露预算的后门——初识阶段她可以改名字，但不能"改口"出一个
    住处来（那等于把挚友阶段的披露偷偷提前）。age_rejected 挂着时 age 例外，
    那是 G6 的圆口误通道，不是新披露。"""
    rank = _STAGE_RANK.get(stage, 0)
    budget = _STAGE_BUDGET.get(rank, _STAGE_BUDGET[0])
    keys = {k for k in budget if (rec or {}).get(k)}
    if (rec or {}).get("age_rejected") and not (rec or {}).get("age"):
        keys.add("age")
    return keys


def _next_slot(rec: dict, allowed: set):
    """这一轮"如果说到自己"该说的下一个槽：allowed 里的第一个键，没有就 None。"""
    for key in _IDENTITY_KEYS:
        if key in allowed:
            return key
    return None


def _revision_count(rec: dict, key: str) -> int:
    return sum(1 for r in ((rec or {}).get("revisions") or []) if r.get("key") == key)


def _allow_revision(rec: dict, key: str) -> bool:
    """G5：这个键还有没有改口额度（一生一次）。"""
    return _revision_count(rec, key) < _MAX_REVISIONS_PER_KEY


# 改口落长期记忆时的人话模板（G5）。逐键写而不是套一个通用句式：这条记忆三天后
# 会被召回注进提示词，"她原来说自己叫「小满」"和"她原先的名字是「小满」"在她
# 嘴里不是一个东西。未知键退到 _SLOT_LABEL 的通用说法。
_REVISION_LINE = {
    "name": "她原来说自己叫「{old}」，后来改口叫「{new}」。",
    "age": "她原来说自己{old}岁，后来改口说{new}岁。",
    "city": "她原来说自己在「{old}」，后来改口说在「{new}」。",
    "occupation": "她原来说自己在做「{old}」，后来改口说在做「{new}」。",
    "home": "她原来说自己住「{old}」，后来改口说住「{new}」。",
}


def _remember_revision(session_id: str, ev: dict) -> None:
    """把一次改口写成一条长期记忆（G5）。

    为什么要落库而不只是改字段：`_merge_freeze` 把新值覆盖上去之后，"她曾经说过
    另一个"这件事在数据里就**完全没有痕迹**了——被记录的改口比不可变的错值更像人，
    真人是会"啊我之前是不是说错了"的。这条记忆被召回时，她能自己提起那次改口。

    写失败不许连累冻结：冻结已经在 _merge_freeze 里落好了，这条只是锦上添花，
    丢了就丢了（memory.remember_note 自己还会走 J12 的补偿队列留痕）。
    """
    try:
        key = ev.get("key") or ""
        tpl = _REVISION_LINE.get(key)
        if tpl is None:
            label = _SLOT_LABEL.get(key, "一项")
            tpl = "她原来说自己的" + label + "是「{old}」，后来改口成「{new}」。"
        line = tpl.format(old=ev.get("old") or "", new=ev.get("new") or "")
        # 延迟 import 破环：memory → quirks → char_life → self_identity
        from capability.memory import remember_note

        # R18b：稳定对象 ID（改口事件 = 槽位+旧值+新值 决定，可复算）——
        # 身份冻结任务重试不会在向量库里堆出重复的改口记忆。
        digest = hashlib.sha1(
            f"{key}|{ev.get('old') or ''}|{ev.get('new') or ''}".encode("utf-8")
        ).hexdigest()[:16]
        remember_note(session_id, line, kind="event", importance=4,
                      memory_id=f"rev-{session_id}-{digest}")
    except Exception:
        pass


# compose_sections 里"下一槽"的人话标签——只出现在提示词组装处，闸门计算一律用键名
_SLOT_LABEL = {"name": "名字", "age": "年龄", "city": "城市",
               "occupation": "你在做的事", "home": "住处"}


def compose_sections(session_id: str, user_text: str = None) -> list:
    """给 persona_engine 的身份注入段。

    G3 的关键改动：**不再列全部五键的清单**——旧版把"名字、年龄、城市、职业"
    一口气递给模型，清单永远赢过"别一口气念档案"那句软约束（实测 turn 1 翻车）。
    现在只递"下一个已解锁的槽"，其余的先留着，等关系到了那个阶段再说。

    user_text=None 表示调用方还没接线（闸门按"问过"放行，行为与旧版一致）；
    接线后应传他上一句的原文（见 _asked_identity_keys）。

    形态：
    - 预算内全空：注入"你的第一次"——只提当前这一个槽；
    - 部分冻结：列出已定下的 + 下一槽（没有下一槽就明说"先留着"）；
    - age_rejected 挂着且还有改口额度：引导她把年龄改口到锚点内（G6）；
    - 全冻结：注入"你是谁"+她自己的自我认知，性格以她的认知为准。
    """
    rec = get_self(session_id)
    stage = _get_stage(session_id)
    gate = _disclosure_gate(rec, stage, _asked_identity_keys(user_text))
    slot = _next_slot(rec, gate)

    sections = []
    lines = []
    for key, label in (("name", "你的名字"), ("age", "年龄"), ("city", "住在"),
                       ("occupation", "在做"), ("home", "住处")):
        if rec.get(key):
            suffix = "（这是你自己取的）" if key == "name" else ""
            lines.append(f"{label}：{rec[key]}{suffix}")
    if lines:
        sections.append("【你是谁】\n" + "\n".join(lines))

    if not rec:
        # 第一次：只提一个槽。初识的槽只可能是名字（且闸门已判过"他问过"），
        # 别的键要么还没解锁、要么她说过被预算拦下了——都轮到"先留着"分支
        head = ("自己取一个名字（认真取，取了就是你的名字），自然地说出来"
                if slot == "name" else
                f"如果这轮自然说到自己，只说【{_SLOT_LABEL.get(slot, '')}】")
        sections.append(
            "【你的第一次】\n"
            "你们才刚认识，你还没有定下来的过去——它是慢慢认识的，不是一轮交代的。\n"
            f"这条回复里{head}，别的（年龄、在哪、做什么）先留着，等以后聊到了再自然说。"
            "说出口的就定下，从此永远一致。\n"
            "开场别套模板：不说\"新来的\"\"刚搬来\"，不问\"怎么想到来找我\"，"
            "不演相亲、网友面基、转学生——你们就是正在聊天的两个人。"
        )
    else:
        if slot:
            if slot == "name" and not rec.get("name"):
                sections.append(
                    "【名字还没定】这轮可以自然说出你的名字——自己取的，"
                    "取了以后就一直是它。别的关于你的事先不用交代。"
                )
            else:
                sections.append(
                    f"【这一轮如果说到自己】只说【{_SLOT_LABEL[slot]}】，别的先留着——"
                    "你们是慢慢认识的，不是一轮交代完的。"
                )
        elif any(rec.get(k) for k in _IDENTITY_KEYS):
            # 有已定下的身份、但本轮没有可说的槽：明确告诉模型别再披露，
            # 这是预算的提示词面（机制面在 _merge_freeze 的 allowed_keys）
            sections.append(
                "【关于你自己】已经定下的就是上面那些；这一轮不用再说新的——"
                "剩下的等更熟了、或者他问起的时候再自然带出来。"
            )

        # G6：越界年龄被拒后引导改口一次——种子不许无声否决她亲口说的话；
        # 额度用完就不再提（再提就是逼她说谎）
        if rec.get("age_rejected") and not rec.get("age") and _allow_revision(rec, "age"):
            sections.append(
                f"【口误要圆】你之前说自己{rec['age_rejected']}岁——那说大了。"
                "这轮找机会自然地改口，把年龄圆回二十出头的样子"
                "（\"开玩笑的，其实我23\"这种口气就行）。改口会被记下来，这很正常——真人也会说错。"
            )

    narrative = (rec.get("self_narrative") or "").strip()
    if narrative:
        sections.append(
            f"【你对自己的认知】{narrative}\n"
            "（这是你自己在最初那段日子里认下的自己——你的性格以它为准，"
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


def _merge_freeze(session_id: str, extracted: dict, allowed_keys=None,
                  revision_keys=None) -> tuple:
    """冻结闸门（G4：真正的预算执行点）+ 改口账本（G5）。返回 (rec, revision_events)。

    - allowed_keys=None：不设限（向后兼容旧调用方/测试）；传集合时，集合外的键
      **不冻结、直接丢弃**——提示词只是劝，这里才是拦。丢弃不留副本：等她到了
      阶段亲口再说，那才是"挣来的"披露。
    - revision_keys=None：改口不设限（向后兼容）；传集合时，集合外的已冻结键
      不接受改口——改口不能成为绕过披露预算的后门（见 _revision_allowed_keys）。
    - 已冻结的键不再被覆盖，唯一例外是改口（G5）：她亲口给出不同的新值、且该键
      还有改口额度（一生一次）时更新值并把旧值记进 revisions；额度用完原样保留。
    - 越界年龄（G6）：不静默丢弃，落 age_rejected 标记 + 记一条 age 改口事件
      （"她原来说X"），compose_sections 据此引导她改口一次。
    - 锁内纯计算（F3 原子闭包）：fill-only 的原子性已被并发测试验证过，别动；
      本版只在同一个闭包里多算几件事。revision_events 在闭包里收集、锁外返回，
      供调用方落长期记忆（接线点见 maybe_freeze）。
    """
    events = []

    def _fill(rec: dict) -> dict:
        rec = rec or {}
        now = ClockTool().now()
        disclosed = list(rec.get("disclosed") or [])
        revisions = list(rec.get("revisions") or [])
        filled_now = False
        for key in _IDENTITY_KEYS:
            raw = str(extracted.get(key) or "").strip()
            if not raw:
                continue
            val = raw[:_NAME_MAX] if key == "name" else raw[:40]
            if key == "age":
                try:
                    n = int(float(val))
                except (TypeError, ValueError):
                    continue  # 不是数字：模型抽坏了，不算她说过
                if not (_AGE_RANGE[0] <= n <= _AGE_RANGE[1]):
                    # G6：越界年龄不再无声丢弃——她亲口说的年龄和档案出现两个答案
                    # 是"单一真相源"被种子静默否决的具体案例。落标记、记事件、
                    # 让 compose 引导她改口一次；同一个值重复出现不重复记账。
                    if str(rec.get("age_rejected") or "") != str(n):
                        rec["age_rejected"] = str(n)
                        events.append({"key": "age", "old": "", "new": str(n),
                                       "kind": "age_rejected", "at": now})
                    continue
                val = str(n)
            if key not in disclosed:
                disclosed.append(key)  # 她亲口说过这个键（哪怕这次被预算拦下）
            frozen = str(rec.get(key) or "").strip()
            rejected = str(rec.get("age_rejected") or "") if key == "age" else ""
            if val == frozen or (rejected and val == rejected):
                continue  # 同值重述不算改口：模型换个措辞再抽一遍是常态，别白烧额度
            old = frozen or rejected  # G6：越界年龄就是"她原来说的X"，圆回来算一次改口
            if old:
                # 改口分支（G5）：值真的变了才走这里
                if revision_keys is not None and key not in revision_keys:
                    continue  # 当前阶段的预算里根本没有这个键——改口不许绕预算
                if len([r for r in revisions if r.get("key") == key]) >= _MAX_REVISIONS_PER_KEY:
                    continue  # 第二次改口：拒绝，冻结值原样保留
                revisions.append({"key": key, "old": old, "new": val, "at": now})
                events.append({"key": key, "old": old, "new": val,
                               "kind": "revision", "at": now})
                rec[key] = val
                if key == "age":
                    rec.pop("age_rejected", None)  # 改口落地，圆回来了
                filled_now = True
            elif allowed_keys is None or key in allowed_keys:
                rec[key] = val
                filled_now = True
            # else：超出披露预算 → 丢弃不冻结（disclosed 已记，防止下轮回头补冻）
        rec["disclosed"] = disclosed
        if revisions:
            rec["revisions"] = revisions
        if not rec.get("frozen_at") and filled_now and any(rec.get(k) for k in _IDENTITY_KEYS):
            rec["frozen_at"] = now  # 第一次真正冻结住身份事实的时刻
        return rec

    updated = services.get("kv_store").update("self", session_id, _fill)
    return updated, events


def _text_similarity(a: str, b: str) -> float:
    """字符二元组 Jaccard 相似度：判断两轮独立生成的自我认知是不是"同一个意思"。

    不用分词器不用 embedding——60 字的中文短文本，bigram 重叠足够分辨
    "一致 / 完全另一幅自画像"，零依赖零成本。
    """
    def grams(s: str) -> set:
        s = re.sub(r"\s+", "", s or "")
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else ({s} if s else set())

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _narrative_prompt(lines: list) -> list:
    """底色提炼的提示词：措辞刻意留"现在"的余地（G7 / 计划与设计 3.4 伪装性）——
    初期她呈现的是社交面，这段认知允许以后被证据重议，不是焊死的人格判决书。"""
    return [
        {"role": "system", "content": (
            "下面是这个女孩迄今为止在聊天里说过的话。"
            "以她的第一人称、用她自己的口吻，写一小段她**现在**对自己的认知"
            "（我说话是什么路子、我大概是个什么样的人），60 字以内，"
            "只输出正文，别提这些材料，别提任何对话细节。"
        )},
        {"role": "user", "content": "\n".join(f"- {ln[:120]}" for ln in lines)},
    ]


def _narrative_conflict_prompt(narrative: str, lines: list) -> list:
    """底色重议的守门人（G7）：只判"明显矛盾"，拿不准一律不矛盾——重议一生只有一次，
    宁可漏判也不能被一轮寻常的坏心情触发。"""
    return [
        {"role": "system", "content": (
            "第一段是这个女孩早前认下的对自己的认知；后面是她最近说的话。"
            "判断她最近的话是否与那段自我认知**明显矛盾**"
            "（比如认下的是\"我说话直\"，最近却一直委婉躲闪）。"
            "只是没提到、或语气有波动，都不算矛盾。"
            "只输出 JSON：{\"contradiction\":true/false,\"evidence\":\"引用她最近的原话\"}"
        )},
        {"role": "user", "content": (
            f"【她的自我认知】{narrative}\n\n【她最近的话】\n"
            + "\n".join(f"- {ln[:120]}" for ln in lines)
        )},
    ]


def _try_freeze_narrative(session_id: str, rec: dict, her_lines: list, llm) -> None:
    """自我认知（底色）的慢冻结（G7）：素材 ≥20 句才开工；每轮独立生成一份草稿，
    与上一轮留下的草稿高度一致才冻结——单次生成可能是当天的状态，跨轮稳定才是证据。
    不一致就弃旧留新，再等几轮。任何失败都静默，下轮再试。"""
    if len(her_lines) < _NARRATIVE_MIN_LINES:
        return  # 素材不足：旧版 5 句就安装人格，那是把社交面焊死成底色
    try:
        draft = (llm.chat(_narrative_prompt(her_lines[-_NARRATIVE_MIN_LINES:]),
                          temperature=0.5, max_tokens=150) or "").strip()
        # temperature 对 Claude 是空操作（llm_client 会 pop 掉），两次生成的差异
        # 来自采样本身——所以判"一致"用文本相似度，不指望参数制造多样性
        if not draft:
            return
        draft = draft[:120]
        cand = rec.get("narrative_candidate") or ""
        if not cand:
            # 第一轮草稿：只留档不冻结，等下一轮独立生成来印证
            def _save_candidate(r: dict) -> dict:
                r = r or {}
                if not r.get("self_narrative") and not r.get("narrative_candidate"):
                    r["narrative_candidate"] = draft
                    r["narrative_drafted_at"] = ClockTool().now()
                return r

            services.get("kv_store").update("self", session_id, _save_candidate)
            return
        if _text_similarity(cand, draft) < _NARRATIVE_SIM_THRESHOLD:
            # 两轮说法对不上：都不是底色，弃旧留新再等一轮
            def _replace_candidate(r: dict) -> dict:
                r = r or {}
                if not r.get("self_narrative"):
                    r["narrative_candidate"] = draft
                    r["narrative_drafted_at"] = ClockTool().now()
                return r

            services.get("kv_store").update("self", session_id, _replace_candidate)
            return

        def _freeze(r: dict) -> dict:
            r = r or {}
            if not r.get("self_narrative"):  # 抢跑保护：并发下先冻结者赢
                r["self_narrative"] = draft
                r["narrative_frozen_at"] = ClockTool().now()
                r.pop("narrative_candidate", None)
                r.pop("narrative_drafted_at", None)
            return r

        services.get("kv_store").update("self", session_id, _freeze)
    except Exception:
        pass  # 失败下轮再试，聊天照常


def _maybe_revise_narrative(session_id: str, rec: dict, her_lines: list, llm) -> None:
    """底色重议（G7，一生一次）：冻结后的自我认知若与她最近的话**明显矛盾**，
    允许重议一次；重议本身记进 revisions（kind=narrative_revision），账本可审计。
    矛盾判定交给 LLM 但门槛刻意收紧（见 _narrative_conflict_prompt）——
    用户的**评价**永远进不了这里（性格不变性宪法），进来的只能是**她自己的话**。"""
    if len(her_lines) < _NARRATIVE_MIN_LINES:
        return
    if not _allow_revision(rec, "self_narrative"):
        return  # 额度已用完：底色定了就是定了
    narrative = (rec.get("self_narrative") or "").strip()
    if not narrative:
        return
    try:
        raw = llm.chat(_narrative_conflict_prompt(narrative, her_lines[-_NARRATIVE_MIN_LINES:]),
                       temperature=0.1, max_tokens=200)
        data = parse_llm_json(raw)
        if not (isinstance(data, dict) and data.get("contradiction") is True):
            return
        evidence = str(data.get("evidence") or "")[:150]
        fresh = (llm.chat(_narrative_prompt(her_lines[-_NARRATIVE_MIN_LINES:]),
                          temperature=0.5, max_tokens=150) or "").strip()[:120]
        if not fresh:
            return

        def _revise(r: dict) -> dict:
            r = r or {}
            old = (r.get("self_narrative") or "").strip()
            # 闭包内重判额度（原子）：并发的另一次重议先落了账，这次就放弃
            if not old or len([x for x in (r.get("revisions") or [])
                               if x.get("key") == "self_narrative"]) >= _MAX_REVISIONS_PER_KEY:
                return r
            revs = list(r.get("revisions") or [])
            revs.append({"key": "self_narrative", "old": old, "new": fresh,
                         "kind": "narrative_revision", "evidence": evidence,
                         "at": ClockTool().now()})
            r["revisions"] = revs
            r["self_narrative"] = fresh
            r["narrative_revised_at"] = ClockTool().now()
            return r

        services.get("kv_store").update("self", session_id, _revise)
    except Exception:
        pass  # 重议失败不影响聊天；额度没动，下轮还有机会


def maybe_freeze(session_id: str, her_reply: str,
                 recent_her_lines: list = None, interaction_count: int = 0,
                 user_text: str = None) -> None:
    """写回时调用：该冻结的冻结（身份事实 / 自我认知）。失败静默，下轮再试。

    - 身份事实：五键没全冻结前每轮都抽取（听她说什么），但**冻不冻由披露预算
      决定**（G1-G4：stage + 他问过什么 → _disclosure_gate → _merge_freeze 的
      allowed_keys，超预算直接丢弃、记入 disclosed，日后不回头补冻）。
    - 改口（G5）：_merge_freeze 返回的 revision 事件在这里打印留痕，并由
      _remember_revision 写成一条长期记忆（"她原来说自己叫X，后来改口叫Y"）——
      被记录的改口比不可变的错值更像人，覆盖字段本身不留任何痕迹。
    - 自我认知（G7）：聊够轮数且素材 ≥20 句才开工，连续两轮独立生成一致才冻结；
      冻结后若与她自己的话明显矛盾，允许重议一次（_maybe_revise_narrative）。

    user_text：他这一轮的原文（闸门用）。已由 orchestration/pipeline.py 的
    _bg_freeze 接上；仍保留 None 的语义（闸门按"问过"放行，与旧行为一致），
    因为直接调用方和单测可以不传。
    """
    her_reply = (her_reply or "").strip()
    if not her_reply:
        return
    rec = get_self(session_id)
    llm = services.get("llm")
    stage = _get_stage(session_id)
    gate = _disclosure_gate(rec, stage, _asked_identity_keys(user_text))
    her_lines = [ln.strip() for ln in (recent_her_lines or []) if ln and ln.strip()]

    # 抽取要不要跑：沿用旧条件"五个键没全冻结就跑"。曾想过"预算全空就省掉这次
    # 调用"，但闸门只管**冻不冻**，不管**听不听**——G6 的 age_rejected 标记、
    # G5 的改口侦测、disclosed 记账，全都要先听到她说了什么才存在。
    if any(not rec.get(k) for k in _IDENTITY_KEYS):
        try:
            raw = llm.chat(_extract_prompt(her_reply), temperature=0.1, max_tokens=160)
            data = parse_llm_json(raw)
            if isinstance(data, dict):
                rec, events = _merge_freeze(session_id, data, allowed_keys=gate,
                                            revision_keys=_revision_allowed_keys(rec, stage))
                for ev in events:
                    # 留痕（G5/G6）：改口与越界年龄是"她的一致性历史"的一部分。
                    print(f"[self_identity] {session_id} {ev.get('kind')}: "
                          f"{ev.get('key')} 「{ev.get('old')}」→「{ev.get('new')}」")
                    if ev.get("kind") == "revision":
                        _remember_revision(session_id, ev)
        except Exception:
            pass  # 抽取失败下轮再试，聊天照常

    # ---- 2. 自我认知（底色）：慢冻结 + 一次重议（G7）----
    if not (rec or {}).get("self_narrative"):
        if interaction_count >= 3 and her_lines:
            _try_freeze_narrative(session_id, rec or {}, her_lines, llm)
    else:
        _maybe_revise_narrative(session_id, rec, her_lines, llm)
