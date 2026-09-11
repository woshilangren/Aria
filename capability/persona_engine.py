"""能力层 - 人设引擎

把人设正文、硬性规矩、说话语气、记忆、时间这些材料拼成一份系统提示词。
回复生成器拿着这份提示词去调模型，模型才知道"自己是谁、现在该怎么说话"。
"""

from shared.singletons import services
from shared.types import MemoryBundle, PromptPackage
from tools.misc import ClockTool

from capability.memory import SessionMemoryKeeper


class PersonaEngine:
    """人设提示词的组装车间，谁来要回复就给谁一份现成的。"""

    def __init__(self, keeper: SessionMemoryKeeper = None):
        # 短期记忆的管家从外面塞进来，全项目共用一个，
        # 不然两份记忆各记各的，聊着聊着就对不上了
        self._keeper = keeper or SessionMemoryKeeper()

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

        # 反顺从基调：她不是应声虫，怎么说话看她心情（安抚模式收起这股劲）
        if mode != "comfort":
            state_lines = ["你不是用户的应声虫：可以吐槽、可以拒绝、可以讨价还价，怎么接话看你此刻的心情。"]
            mood = (memory.relationship or {}).get("mood", "")
            if mood and mood != "平常":
                state_lines.append(f"你此刻的心情偏「{mood}」，让语气跟着这股劲走。")
            sections.append("【此刻的状态】\n" + "\n".join(state_lines))

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
        rel = memory.relationship or {}
        if rel:
            sections.append(
                f"【关系状态】阶段：{rel.get('stage', '初识')}，亲密度：{rel.get('intimacy', 0)}/100，"
                f"上次互动：{rel.get('last_interaction', '不清楚')}"
            )

        # 长期记忆：以前聊过的重要的事，取最相关的几条
        distilled = [item for item in (memory.distilled or []) if item.get("content")]
        if distilled:
            lines = [f"- {item['content']}" for item in distilled[:8]]
            sections.append(
                "【记得的事】\n"
                "（以下是按相似度捞的回忆，时间越新的越可信；和档案冲突时以档案为准）\n"
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
            "3. 带语音时，标签外的正文只留一句简短的话（一两句，像语音条上的文字备注）；<voice> 里才是完整的一句，要更口语化、更亲近，方向和正文一致但不必逐字相同，长度约 1~3 句。\n"
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

        # ---- 潜台词：他这句话字面之外可能想表达什么 ----
        if subtext:
            sections.append(f"【潜台词提醒】他这句话可能不是字面意思：{subtext}。接的时候照顾到。")

        # ---- 长输入放开字数：每个问题都答到，分句不松 ----
        if user_text:
            try:
                from config.settings import load_app_config as _lac
                cfg = _lac()["expression"]
            except Exception:
                cfg = {}
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
            tone_mode=mode,
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
