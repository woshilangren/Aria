"""能力层 - 日记模块

收工后把当天对话整理成一篇日记，用它自己的口吻写，存进向量库。
之后闲聊就能自然想起"上次你说过……"。触发入口有两个：
手动喊它写（s11 接）、闲置太久自动写（s12 接），本模块只管写这件事本身。

I2 修复后的三条口径：
- 幂等键 = "日期 + 是否已覆盖当天新内容"（不再是裸日期——上午写过、下午又聊了，
  重写覆盖同一篇，不然下午的对话永远进不了日记）；
- 日记 id 带 session_id（多会话时 A 的日记不再让 B 当天永远 skip），读取兼容旧 id；
- 翻篇显式按日期排序取最旧（不再依赖 list_diaries 的返回顺序——顺序一反，
  删掉的是最新日记且因幂等闸门不可恢复）。
D4：反流水账的两段式（先问 standout、再扩正文）合并成一次结构化调用，
键序（items 在前、diary 在后）保住 S9 的"先挑亮点再写"机制，成本减半。
"""

from datetime import datetime, timedelta

from config.settings import load_app_config
from shared.singletons import services
from tools.misc import ClockTool, parse_llm_json

from capability import self_identity


class DiaryWriter:
    """角色的日记本：当天对话 -> 角色口吻日记 -> 向量库，超过上限自动翻掉最旧的。"""

    # 结构性要求写死在这；"是谁、用什么口吻写"跟人设走——
    # persona_config.json 的种子定调，diary_notes 补充口吻要求。
    # S9 反流水账 + D4 单次合并：一个 JSON 同时给 standout（items）与正文（diary）。
    # **键序就是机制**：自回归生成下模型必须先逐条写下"哪里和平常不一样"，
    # 再写正文——结构一卡就写不出流水账。items 允许空数组（完全例行的日子
    # 存 0 条是信号不是失败），正文照写。
    _MERGED_PROMPT = (
        "回顾今天的对话，先挑出值得记的事，再写今天的日记。只输出 JSON：\n"
        "{\"items\":[{\"what\":\"发生了什么\",\"feeling\":\"当时你的感受"
        "（带身体或感官细节）\",\"why_special\":\"它比平时特别在哪\"}],"
        "\"diary\":\"日记正文\"}\n"
        "- items：和平常不一样、值得你记下来的事，最多 2 条，挑最特别的；"
        "完全例行的一天就给空数组，别硬凑；\n"
        "- diary：100~150 字，第一人称，你自己的口吻；有 items 就以它们为中心写，"
        "别的例行琐事不用提；没有就平实地写今天；\n"
        "- 全部只能从今天的对话里来，绝不编造。JSON 必须合法，diary 里不要引号或标题。"
    )
    # 兜底调用（合并输出解析失败/模型挂）用的保守写法——纯正文，旧行为
    _PROMPT_BODY = (
        "请用第一人称写今天的日记。\n"
        "要求：\n"
        "- 用你自己的口吻写，符合你的人设\n"
        "- 100~150 字：今天聊了什么、他的状态、你自己的小心思\n"
        "- 直接输出日记正文，不要任何解释、引号或标题"
    )

    # K5：diary_max_entries 不再构造期快照——load_app_config() 是 lru_cache 的，
    # 运行时改参数靠 API 端点原地改缓存对象，快照会让配置热改对本模块无效。
    # 每次翻篇时现读，一次 dict 取值不值几个钱。

    def write_diary(self, session_id: str, date: str = "", skip_if_exists: bool = False) -> str:
        """给某一天写日记，返回日记内容；当天没聊过天或写失败都返回空串。

        date 留空时取"最后一条**真实对话**的日期"而不是今天——闲置触发往往
        跨了零点（23:50 聊完，00:20 才触发），跟着最后聊天时间走才不会写错天。
        I2-5：她自己的主动消息（intent='proactive'）不算"聊过"——主动消息也落
        session，不排除的话日期锚点和取材都会被她自己说的话带偏。
        skip_if_exists 给自动巡检用，幂等键是"日期 + 是否已覆盖当天新内容"
        （I2-1）：已在册且没有更新的对话才跳过；有新对话就重写覆盖同一篇 id。
        """
        try:
            kv = services.get("kv_store")
            if not date:
                recent = [r for r in (kv.read("session", session_id) or [])
                          if r.get("intent") != "proactive"]
                date = (recent[-1].get("time") or "")[:10] if recent else ""
            if not date:
                return ""

            start = f"{date}T00:00:00"
            next_day = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
            end = next_day.strftime("%Y-%m-%dT00:00:00")
            chats = [c for c in kv.chats_between(session_id, start, end)
                     if c.get("intent") != "proactive"]
            if not chats:
                return ""
            if skip_if_exists and self._entry_covers(
                self._find_diary(session_id, date), chats[-1].get("time") or ""
            ):
                return ""  # 在册且已覆盖到最后一条对话：别让模型白写一遍

            diary = self._compose(chats, session_id=session_id)
            if not diary:
                return ""
            if not self._save(session_id, date, diary):
                return ""  # 存都存不进去，就当没写成，别让上层空欢喜
            self._drop_legacy_diary(session_id, date)
            self._trim_old_diaries()
            return diary
        except Exception:
            return ""  # 日记写不成不吵不闹，聊天照常

    def covers(self, session_id: str, date: str) -> bool:
        """该会话那天的日记是否已覆盖当天全部真实对话（I2-1，给巡检当快路径）。

        判据 = 在册日记的 timestamp >= 当天最后一条非主动消息的时间。
        进程重启后巡检靠它恢复"已写过"状态，不依赖内存 _done。
        """
        entry = self._find_diary(session_id, date)
        if entry is None:
            return False
        try:
            kv = services.get("kv_store")
            start = f"{date}T00:00:00"
            end = (datetime.strptime(date, "%Y-%m-%d")
                   + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
            chats = [c for c in kv.chats_between(session_id, start, end)
                     if c.get("intent") != "proactive"]
            last_time = (chats[-1].get("time") or "") if chats else ""
        except Exception:
            last_time = ""
        return self._entry_covers(entry, last_time)

    def run_as_tool(self, session_id: str = "", **_kwargs) -> str:
        """给工具管线用的入口，手动"写日记"就从这进。

        签名故意做得能吃：session_id 由执行链注入（不是模型填的），
        模型要是手痒多给了别的参数，**_kwargs 一律吃掉不接。
        """
        return self.write_diary(session_id)

    # ---- id 与兼容（I2-2）----

    @staticmethod
    def _diary_id(session_id: str, date: str) -> str:
        # id 带 session_id + 日期：同一天重写就是覆盖，多会话互不遮挡
        return f"diary-{session_id}-{date}"

    def _find_diary(self, session_id: str, date: str):
        """找该会话那天的在册日记元数据；没有返回 None。

        I2-2 兼容：id 从 `diary-{date}` 改成 `diary-{session_id}-{date}` 之后，
        旧格式的日记**不许丢**——读取时新旧两种 id 都认。旧 id 只在元数据
        session 相符（或缺失）时算数：旧格式的锅就是不含 session，
        A 的日记让 B 当天永远 skip，认的时候要把它认回来。
        重写到新 id 后旧条目由 _drop_legacy_diary 随写随迁，避免同一天挂两篇。
        """
        new_id = self._diary_id(session_id, date)
        old_id = f"diary-{date}"
        try:
            entries = services.get("vector_store").list_diaries()
        except Exception:
            return None
        legacy = None
        for e in entries:
            if e.get("id") == new_id:
                return e
            if e.get("id") == old_id and (e.get("session_id") or session_id) == session_id:
                legacy = e
        return legacy

    def _drop_legacy_diary(self, session_id: str, date: str) -> None:
        """新 id 写成功后删掉同会话同天的旧格式条目（I2-2 的一次性迁移，随写随迁）。

        只认"元数据 session 相符（或缺失）"的旧条目——别的会话的旧日记不动，
        那是它自己下次重写时迁移。
        """
        old_id = f"diary-{date}"
        try:
            store = services.get("vector_store")
            for e in store.list_diaries():
                if e.get("id") == old_id and (e.get("session_id") or session_id) == session_id:
                    store.delete_diary(old_id)
                    break
        except Exception:
            pass  # 迁移失败不拦写日记，旧条目留着也只是多占一篇配额

    @staticmethod
    def _entry_covers(entry, last_chat_time: str) -> bool:
        """在册日记是否已覆盖到 last_chat_time（I2-1 的新幂等判据）。

        比不了时间（老日记缺 timestamp / 脏数据）时按"已覆盖"处理——退回旧行为
        （在册就跳过），宁可不重写也别让巡检每分钟白烧 LLM。
        """
        if not entry:
            return False
        ts = entry.get("timestamp") or ""
        if not last_chat_time or not ts:
            return True
        try:
            return (datetime.fromisoformat(ts[:19])
                    >= datetime.fromisoformat(last_chat_time[:19]))
        except ValueError:
            return True

    # ---- 生成 ----

    def _diary_system(self, session_id: str = "") -> str:
        """日记的系统提示词：只回答"是谁、用什么口吻写"。

        H5："是谁"不再硬编"年轻女孩"——身份唯一来源是 self 表，取名统一走
        self_identity.display_name()（它内部自带人设占位名兜底），种子文件补背景。
        D4：正文的格式要求从 system 挪进各调用的 user 消息——合并调用要 JSON、
        兜底调用要纯正文，两种格式要求不能同住一个 system 里打架。
        """
        try:
            who = f"你是{self_identity.display_name(session_id)}。"
        except Exception:
            who = "以你自己的身份和口吻写。"
        notes = ""
        try:
            persona = services.get("kv_store").read("persona_config", "")
            if persona and getattr(persona, "background_story", ""):
                who += f"\n{persona.background_story}"
            notes = (getattr(persona, "diary_notes", "") or "").strip()
        except Exception:
            pass  # 人设读不到就只带名字写，日记照出
        parts = [who]
        if notes:
            parts.append(f"【口吻要求】{notes}")
        return "\n".join(parts)

    def _compose(self, chats: list, session_id: str = "") -> str:
        """把一天的对话交给模型，写成她口吻的日记。

        D4：S9 两段式（先问 standout、再扩正文）合并成一次结构化调用——
        一个 JSON 同时给 items 与 diary，键序保住"先挑亮点再写正文"的
        反流水账机制，LLM 成本减半。解析失败/模型挂 → 保守写法兜底（旧行为）。
        两个分支共用同一份截断 transcript（以前保守分支不截断，长日子白烧 token）。
        """
        name = "我"
        try:
            name = self_identity.display_name(session_id) or "我"
        except Exception:
            pass
        transcript = "\n".join(
            f"{'他' if c.get('role') == 'user' else name}：{c.get('text', '')}"
            for c in chats
        )[-3000:]
        system = self._diary_system(session_id)
        llm = services.get("llm")
        try:
            raw = llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{self._MERGED_PROMPT}\n\n今天的对话：\n{transcript}"},
                ],
                temperature=0.7,
                max_tokens=700,
            )
            data = parse_llm_json(raw)
            diary = str((data if isinstance(data, dict) else {}).get("diary") or "").strip()
            if diary:
                return diary
        except Exception:
            pass  # 解析失败/模型挂 → 保守写法，行为不比改造前差
        return llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{self._PROMPT_BODY}\n\n今天的对话记录：\n{transcript}"},
            ],
            temperature=0.8,
            max_tokens=400,
        )

    # ---- 存取 ----

    def _save(self, session_id: str, date: str, diary: str) -> bool:
        # id 带 session_id + 日期（I2-2）：同一天同一会话重写就是覆盖，
        # 不会堆出重复篇目，也不会让别的会话被 skip
        return services.get("vector_store").upsert_diary(
            self._diary_id(session_id, date),
            diary,
            {
                "kind": "diary",
                "date": date,
                "session_id": session_id,
                "timestamp": ClockTool().now(),
            },
        )

    def _trim_old_diaries(self) -> None:
        """日记超过上限就翻掉最旧的，只留最近 N 篇。

        I2-3：显式按日期（同日按 timestamp、再按 id）排序后取最旧的——以前直接
        切 list_diaries() 的头，把"返回旧→新"当隐式契约：顺序一反，删掉的是
        **最新**日记，而且因为幂等闸门再也补不回来。契约要握在自己手里。
        """
        store = services.get("vector_store")
        try:
            entries = store.list_diaries()
        except Exception:
            return
        max_entries = int(
            load_app_config().get("memory", {}).get("diary_max_entries", 30) or 30
        )
        overflow = max(0, len(entries) - max_entries)
        if overflow <= 0:
            return
        ordered = sorted(
            entries,
            key=lambda e: (e.get("date") or "", e.get("timestamp") or "", e.get("id") or ""),
        )
        for entry in ordered[:overflow]:
            store.delete_diary(entry["id"])
