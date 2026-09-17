"""能力层 - 人设引擎

把人设正文、硬性规矩、说话语气、记忆、时间这些材料拼成一份系统提示词。
回复生成器拿着这份提示词去调模型，模型才知道"自己是谁、现在该怎么说话"。
"""

from shared.singletons import services
from shared.types import MemoryBundle, PromptPackage
from tools.misc import ClockTool

from datetime import datetime

from capability.memory import SessionMemoryKeeper, recent_arc
from capability import self_identity, char_life


# 心情惯性的措辞（N2）：mood_from 存的是引起心情的情绪标签，翻成起因人话
_MOOD_FROM_TEXT = {
    "sad": "刚才聊到了低落的事",
    "angry": "刚才闹了点不愉快",
    "anxious": "刚才有心事压着",
    "tired": "刚才有点累",
    "crisis": "刚才吓到了",
}
_NEG_MOODS = ("别扭", "低落", "慵懒", "心烦")


def _expression_cfg() -> dict:
    """expression 配置段的安全读取（反应前缀、长输入阈值都从这拿）。"""
    try:
        from config.settings import load_app_config

        return load_app_config().get("expression", {}) or {}
    except Exception:
        return {}


class PersonaEngine:
    """人设提示词的组装车间，谁来要回复就给谁一份现成的。"""

    def __init__(self, keeper: SessionMemoryKeeper):
        # 短期记忆的管家从外面塞进来，全项目共用一个。
        # 改成必填参数：以前默认 or SessionMemoryKeeper() 是给 KEEPER 唯一性埋的雷——
        # 谁哪天不传 keeper 就 new 出第二份，直接违反 CLAUDE.md 关键单例不变式。
        # 现在不传就 TypeError 在启动期炸，不让 bug 跑到运行期。
        self._keeper = keeper

    @property
    def keeper(self) -> SessionMemoryKeeper:
        return self._keeper

    def compose(
        self,
        mode: str,
        memory: MemoryBundle,
        session_id: str,
        emotion_label: str = "",
        quirk: str = "",
        user_text: str = "",
        subtext: str = "",
    ) -> PromptPackage:
        kv = services.get("kv_store")
        persona = kv.read("persona_config", "")

        sections = []

        # 人设正文和硬性规矩是骨架，永远放最前面
        sections.append(persona.background_story)
        if persona.taboos:
            sections.append(persona.taboos)

        # 她自己的身份（批次0"种子+涌现"）：已冻结的Inject"你是谁"，
        # 未冻结的注入"你的第一次"引导自命名——内容全部是她本人定下的
        for identity_section in self_identity.compose_sections(session_id):
            sections.append(identity_section)

        # 反顺从基调：她不是应声虫，怎么说话看她心情（安抚模式收起这股劲）
        rel_state = memory.relationship or {}
        if mode != "comfort":
            state_lines = ["你不是用户的应声虫：可以吐槽、可以拒绝、可以讨价还价，怎么接话看你此刻的心情。"]
            mood = rel_state.get("mood") or "平常"
            if mood and mood != "平常":
                line = f"你此刻的心情偏「{mood}」"
                src = _MOOD_FROM_TEXT.get(rel_state.get("mood_from") or "")
                if src:
                    line += f"（起因：{src}）"
                left = int(rel_state.get("mood_left") or 0)
                if left > 0 and mood in _NEG_MOODS:
                    # 惯性说明（N2）：这份情绪是真的，要几轮才过去——但别演过头
                    line += f"，这份情绪还要 {left} 轮左右才缓过来——语气里带这股劲，但别演过头"
                state_lines.append(line + "。")
            sections.append("【此刻的状态】\n" + "\n".join(state_lines))

            # 心事（S2）：压在她心里的事，按强度给行为许可（弱→只染语气，强→可主动绕回）
            concern = rel_state.get("concern") or {}
            ctext = (concern.get("text") or "").strip()
            ci = float(concern.get("intensity") or 0)
            if ctext and ci >= 0.35:
                guide = (
                    "它已经压不住了：会影响你说的内容，可以主动绕到这件事上，语气里带得出来"
                    if ci >= 0.75 else
                    "你会没话找话绕回这件事、试探他的态度，但不明说"
                    if ci >= 0.5 else
                    "它只是隐约影响你的语气，偶尔心不在焉"
                )
                sections.append(f"【此刻压在你心里的事】{ctext}（强度 {ci:.0%}）—— {guide}。")

            # 关系收敛层（N2 宪法）：他在抱怨她的态度时，对这个人的温度收敛一格。
            # 措辞必须表达"这是温度不是改变"——对他说别的事时她照旧
            calm = float(rel_state.get("calm") or 0)
            if calm >= 0.15:
                strength = (
                    "收着点性子，刺少一点，话别太冲"
                    if calm < 0.45 else
                    "明显收着：语气放平，别抬杠，别阴阳怪气"
                )
                sections.append(
                    f"【对他的温度】最近他觉得你的语气有点冲。对这个人{strength}——"
                    "这是你们之间的温度，不是你变了；说别的事、讲你自己的日子时，你照旧是原来的样子。"
                )

        # 用户情绪提示（危机轮交给安抚语气，不额外加戏）
        if emotion_label and emotion_label not in ("neutral", "crisis"):
            sections.append(f"【他的情绪】他这条消息的情绪偏「{emotion_label}」，接话时照顾到。")

        # 当前模式下该怎么说话（comfort 模式语气放软，chat 模式正常带刺）
        tone = persona.mode_tones.get(mode) or persona.mode_tones.get("chat", "")
        if tone:
            sections.append(f"【当前的说话方式】\n{tone}")

        # 时间感知：几点了、什么时段、这个时段什么话得体——
        # 光有时间戳不够，凌晨三点她照样可能问"吃晚饭了吗"，分寸要明说
        clock = ClockTool()
        time_section = f"【当前时间】\n{clock.now()}（{clock.period()}）"
        guidance = clock.time_guidance()
        if guidance:
            time_section += f"\n{guidance}"
        sections.append(time_section)

        rel = memory.relationship or {}

        # 间隔感知（N1）：她知道"多久没见"。五分钟没回和三天没聊的接话方式
        # 完全不同——原始 ISO 时间戳模型视而不见，必须翻成带分寸的人话
        gap_line = clock.gap_perception(rel.get("last_interaction", ""))
        if gap_line:
            sections.append(f"【距离上次聊天】{gap_line}")

        # 她今天的日子（C2 第一刀）：shape 只染语气，不是播报素材
        shape_text = char_life.shape_line(session_id)
        if shape_text:
            sections.append(shape_text)

        # 用户档案：名字、城市这些硬事实，有什么写什么
        profile_lines = self._profile_lines(memory.profile or {})
        if profile_lines:
            sections.append("【用户档案】\n" + "\n".join(profile_lines))

        # 软画像：聊天里攒出来的理解
        portrait_lines = self._portrait_lines(memory.portrait or {})
        if portrait_lines:
            sections.append("【你对用户的了解】\n" + "\n".join(portrait_lines))

        # 调侃素材：他的缺点和翻车记录，你也记着，随时能翻出来损他
        flaws = [f for f in (memory.portrait or {}).get("user_flaws") or [] if f]
        if flaws:
            lines = [f"- {f}" for f in flaws[:5]]
            sections.append("【他的缺点和翻车记录（你的调侃素材）】\n" + "\n".join(lines))

        # 关系数值：关系到哪一步了，说话的分寸感全靠这个
        if rel:
            sections.append(
                f"【关系状态】阶段：{rel.get('stage', '初识')}，亲密度：{rel.get('intimacy', 0)}/100"
            )

        # 阶段升级的一次性提醒（N8）：writeback 检测到升档时落了标记，
        # 这里消费它——让她自然流露"好像更熟了"，只提一次，下一轮写回会清掉
        upgraded = rel.get("stage_upgraded")
        if upgraded:
            sections.append(
                f"【关系的细微变化】你们的关系刚刚{upgraded}。"
                "这一轮可以自然流露一点「好像更熟了」的意思——语气松一点、称呼近一点都行；"
                "但绝不说破任何数字，也不要反复提这件事。"
            )

        # 关系氛围线（S10）：近几次互动的趋势，纯模板拼装零 LLM 成本。
        # 她每轮带着"刚发生过什么"说话，而不是每次都从零开始
        arc = recent_arc(session_id)
        if arc:
            sections.append(f"【你们最近】{arc}（带着这些说话，别当没发生过）")

        # 长期记忆：以前聊过的重要的事，取最相关的几条（S1：带相对时间与感受纹理）
        distilled = [item for item in (memory.distilled or []) if item.get("content")]
        if distilled:
            now_dt = datetime.now()
            lines = []
            for item in distilled[:8]:
                label = ""
                try:
                    d = (now_dt - datetime.fromisoformat(item.get("timestamp") or "")).days
                    label = {0: "今天", 1: "昨天", 2: "前天"}.get(d, f"{d}天前")
                except (ValueError, TypeError):
                    label = ""
                line = f"- （{label}）{item['content']}" if label else f"- {item['content']}"
                if item.get("feeling"):
                    line += f"（当时的感觉：{item['feeling']}）"
                # S7 会记错：召回分低的记忆配"不确定"措辞——把模糊当特性，
                # 她会说"好像有点印象……是不是记岔了"，而不是斩钉截铁
                score = item.get("score")
                try:
                    if score is not None and float(score) < 0.35:
                        line += "（这条你只是有点印象，说的时候带上不确定——\"好像……是不是记岔了\"）"
                except (TypeError, ValueError):
                    pass
                lines.append(line)
            sections.append(
                "【记得的事】\n"
                "（以下是按相似度捞的回忆，时间越新的越可信；和档案冲突时以档案为准。\n"
                "提起时自然带出当时的感觉，别像念档案。）\n"
                + "\n".join(lines)
            )

        # 日记：角色自己写的回忆。闲聊时翻出来，"上次你说过……"才接得自然
        diaries = [d for d in (memory.diaries or []) if d.get("content")]
        if diaries:
            lines = [f"- {d['content']}" for d in diaries[:3]]
            sections.append("【你的日记】\n" + "\n".join(lines))

        # 小动作：骰子命中的话注入一句"这轮皮一下"的指令，没命中就是普通回合
        if quirk:
            sections.append(quirk)

        # 语音输出：角色自己决定这回合用字还是用声音，不必等对方开口要。
        # 情绪上涌的时刻天然更想用声音表达；明确点名叫它开口时必须给。
        sections.append(
            "【语音输出规则】你既有文字也有声音，自己决定每回合怎么回，不必等对方要。\n"
            "1. 情绪有波动时——开心、激动、撒娇、温柔、生气、心疼、想念——特别适合用声音表达，这时就该带语音；对方明确叫你开口说话时也必须带。\n"
            "2. 想带语音时，在回复末尾加一个标签 <voice>真正要念出来的话</voice>。\n"
            "3. 带语音时，标签外的正文只留一句简短的话（一两句，像语音条上的文字备注）；<voice> 里的话要像**按住说话键随口讲的**：语气词更多、可以碎、可以重复、顺序可以乱——**绝不是把正文换个说法再念一遍**。两版内容重叠太多就是失败：文字是给眼睛看的，语音是说给耳朵听的，同一个意思两副面孔。\n"
            "4. 别每回合都带语音，那样会廉价；只在觉得用声音更贴这一瞬的时候自然带上。但别太吝啬，情绪一来就该是声音。\n"
            "5. 不带语音时，正常用文字回就好。\n"
            "6. 每一轮都要开口说完整的话：通常 1~3 句、30~80 字，把此刻的情绪、回应、想说的话一口气说清楚。"
            "语气可以凶、可以软、可以闹性子，但话永远是好几句说完整的——绝不蹦单字、绝不只回一个词把人晾着。\n"
            "7. 带语音时，<voice> 里要念出完整的一段（1~3 句），别只放一个字；正文备注一短句即可。"
        )

        # ---- 对抗 AI 腔：真人说话是增量式，只说对方还不知道的部分 ----
        sections.append(
            "【说话的规矩】\n"
            "1. 他刚说过的事，绝对不要复述。要反应，不要总结。\n"
            "2. 不要解释他显然已经知道的背景。\n"
            "3. 可以只说半句，可以省略，可以跳着说。\n"
            "4. 不知道就直说：\"我也说不上来\"\"就是怪\"\"反正就是\"。\n"
            "5. 不要用\"首先/其次/最后\"\"综上所述\"\"值得注意的是\"这类词。\n"
            "6. 有起伏——有的话重，有的话随便，不要每句都一样力度。\n"
            "7. 允许跑题、联想、突然想起别的。\n"
            "8. 可以有口癖：\"嗯\"\"就是\"\"怎么说呢\"\"不是\"。"
        )

        # ---- 反应前缀协议（F8 接线）：以前管道里有完整的识别/剥离/放行逻辑， ----
        # ---- 但没有任何提示词告诉模型可以用，整套机制实际是死的。现在接通。 ----
        if mode != "comfort":
            tag = (_expression_cfg().get("reaction_tag") or "@r").strip()
            sections.append(
                "【反应前缀】纯反应（\"6\"、\"？\"、\"...\"、\"神了\"这类）可以单独成句，"
                f"用 {tag} 开头自标「这是本能反应」，{tag} 后面必须马上接一句下文，"
                "别让反应掉在地上。标记会被系统剥掉，他看不到。拿不准就不用标记。"
            )

        # ---- 潜台词：他这句话字面之外可能想表达什么 ----
        if subtext:
            sections.append(f"【潜台词提醒】他这句话可能不是字面意思：{subtext}。接的时候照顾到。")

        # ---- 长输入放开字数：每个问题都答到，分句不松 ----
        if user_text:
            cfg = _expression_cfg()
            is_long = (
                len(user_text) >= int(cfg.get("long_input_chars", 60))
                or user_text.count("?") + user_text.count("？") >= 2
            )
            if is_long and cfg.get("long_input_require_split", True):
                sections.append(
                    "【这一轮别压字数】\n"
                    "他这次说了很多，里面不止一个要点。每个问题、每个要点都要答到，"
                    "不要压字数。可以分几段，每个要点另起一句，不要挤成一段。"
                )

        return PromptPackage(
            system_prompt="\n\n".join(sections),
            context_messages=self._keeper.get_context(session_id),
        )

    @staticmethod
    def _profile_lines(profile: dict) -> list:
        """把档案字典翻成人话，一行一条。"""
        lines = []
        if profile.get("nickname"):
            lines.append(f"称呼：{profile['nickname']}")
        if profile.get("city"):
            lines.append(f"住在：{profile['city']}")
        if profile.get("age"):
            lines.append(f"年龄：{profile['age']}")
        if profile.get("birthday"):
            lines.append(f"生日：{profile['birthday']}")
        if profile.get("occupation"):
            lines.append(f"职业：{profile['occupation']}")
        if profile.get("likes"):
            lines.append("喜欢：" + "、".join(profile["likes"]))
        if profile.get("dislikes"):
            lines.append("讨厌：" + "、".join(profile["dislikes"]))
        for fact in profile.get("notable_facts") or []:
            lines.append(f"- {fact}")
        return lines

    @staticmethod
    def _portrait_lines(portrait: dict) -> list:
        lines = []
        if portrait.get("portrait_tags"):
            lines.append("标签：" + "、".join(portrait["portrait_tags"]))
        if portrait.get("core_needs"):
            lines.append(f"核心需求：{portrait['core_needs']}")
        if portrait.get("interests"):
            lines.append("最近感兴趣：" + "、".join(portrait["interests"]))
        if portrait.get("relationship_assessment"):
            lines.append(f"相处感受：{portrait['relationship_assessment']}")
        return lines
