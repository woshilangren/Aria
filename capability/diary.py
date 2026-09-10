"""能力层 - 日记模块

收工后把当天对话整理成一篇日记，用它自己的口吻写，存进向量库。
之后闲聊就能自然想起"上次你说过……"。触发入口有两个：
手动喊它写（s11 接）、闲置太久自动写（s12 接），本模块只管写这件事本身。
"""

from datetime import datetime, timedelta

from config.settings import load_app_config
from shared.singletons import services
from tools.misc import ClockTool


class DiaryWriter:
    """角色的日记本：当天对话 -> 角色口吻日记 -> 向量库，超过上限自动翻掉最旧的。"""

    # 结构性要求写死在这；"是谁、用什么口吻写"跟人设走——
    # persona_config.json 的 char_name/background_story 定调，diary_notes 补充口吻要求。
    # 这样换人设只改配置文件，日记口吻自动跟着换，代码不用动
    _PROMPT_BODY = (
        "请用第一人称写今天的日记。\n"
        "要求：\n"
        "- 用你自己的口吻写，符合你的人设\n"
        "- 100~150 字：今天聊了什么、对方的状态、你自己的小心思\n"
        "- 直接输出日记正文，不要任何解释、引号或标题"
    )

    def __init__(self):
        self._max_entries = load_app_config()["memory"].get("diary_max_entries", 30)

    def write_diary(self, session_id: str, date: str = "", skip_if_exists: bool = False) -> str:
        """给某一天写日记，返回日记内容；当天没聊过天或写失败都返回空串。

        date 留空时取"最后一条聊天记录的日期"而不是今天——闲置触发往往
        跨了零点（23:50 聊完，00:20 才触发），跟着最后聊天时间走才不会写错天。
        skip_if_exists 给自动巡检用：那天的日记已经在册就别让模型白写一遍
        （id 带日期，重写只是覆盖，但每次都是一次实打实的 LLM 调用）。
        """
        try:
            kv = services.get("kv_store")
            if not date:
                recent = kv.read("session", session_id) or []
                date = (recent[-1].get("time") or "")[:10] if recent else ""
            if not date:
                return ""
            if skip_if_exists and self._has_diary(date):
                return ""

            start = f"{date}T00:00:00"
            next_day = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
            end = next_day.strftime("%Y-%m-%dT00:00:00")
            chats = kv.chats_between(session_id, start, end)
            if not chats:
                return ""

            diary = self._compose(chats)
            if not diary:
                return ""
            if not self._save(session_id, date, diary):
                return ""  # 存都存不进去，就当没写成，别让上层空欢喜
            self._trim_old_diaries()
            return diary
        except Exception:
            return ""  # 日记写不成不吵不闹，聊天照常

    def run_as_tool(self, session_id: str = "", **_kwargs) -> str:
        """给工具管线用的入口，手动"写日记"就从这进。

        签名故意做得能吃：session_id 由执行链注入（不是模型填的），
        模型要是手痒多给了别的参数，**_kwargs 一律吃掉不接。
        """
        return self.write_diary(session_id)

    def _diary_system(self) -> str:
        """日记的系统提示词：是谁、用什么口吻写，全跟人设正主文件走。"""
        who = "你是用户的 AI 伙伴。"
        notes = ""
        try:
            persona = services.get("kv_store").read("persona_config", "")
            if persona and getattr(persona, "char_name", ""):
                who = f"你是{persona.char_name}。{persona.background_story}"
            notes = (getattr(persona, "diary_notes", "") or "").strip()
        except Exception:
            pass  # 人设读不到就按通用口吻写，日记照出
        parts = [who]
        if notes:
            parts.append(f"【口吻要求】{notes}")
        parts.append(self._PROMPT_BODY)
        return "\n".join(parts)

    def _compose(self, chats: list) -> str:
        """把一天的对话交给模型，写成角色口吻的日记。"""
        name = "AI"
        try:
            persona = services.get("kv_store").read("persona_config", "")
            name = (persona.char_name or "AI") if persona else "AI"
        except Exception:
            pass
        transcript = "\n".join(
            f"{'用户' if c.get('role') == 'user' else name}：{c.get('text', '')}"
            for c in chats
        )
        return services.get("llm").chat(
            [
                {"role": "system", "content": self._diary_system()},
                {"role": "user", "content": f"今天的对话记录：\n{transcript}"},
            ],
            temperature=0.8,
            max_tokens=400,
        )

    def _save(self, session_id: str, date: str, diary: str) -> bool:
        # id 带日期：同一天重写就是覆盖，不会堆出一堆重复篇目
        return services.get("vector_store").upsert_diary(
            f"diary-{date}",
            diary,
            {
                "kind": "diary",
                "date": date,
                "session_id": session_id,
                "timestamp": ClockTool().now(),
            },
        )

    def _has_diary(self, date: str) -> bool:
        """那天的日记是否已经在册（id 就是 diary-日期，查一下就行）。"""
        diary_id = f"diary-{date}"
        return any(e.get("id") == diary_id for e in services.get("vector_store").list_diaries())

    def _trim_old_diaries(self) -> None:
        """日记超过上限就翻掉最旧的，只留最近 N 篇。"""
        store = services.get("vector_store")
        entries = store.list_diaries()
        overflow = max(0, len(entries) - self._max_entries)
        for entry in entries[:overflow]:
            store.delete_diary(entry["id"])
