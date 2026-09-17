"""调度层 - 管理器（合并自原 managers.py 与 decisions.py）

这堆类都是"只做决定、不干活"的角色：
    - VoiceRouteManager：语音三路由的决策与热切换
    - InfoGapCoordinator：缺信息 → 追问 → 补充 → 入档 的循环协调
    - FallbackController：异常降级的决策中心
"""

from capability.memory import ProfileUpdater
from shared.singletons import services

# 语音路由的三个方案名
ROUTE_E2E = "e2e"            # 端到端：录音直进语音对话引擎
ROUTE_CASCADE = "cascade"    # 级联：转文字 → 文字对话 → 合成音频
ROUTE_REALTIME = "realtime"  # 实时专线：直连多模态模型，识别对话合成一线完成


class VoiceRouteManager:
    """语音双路由决策与切换。切换时不动会话上下文与亲密度。"""

    # 连续失败这么多次就自动降级
    _FAIL_LIMIT = 2

    def current_route(self, session_id: str) -> str:
        """拿会话当前的路由，没配置过就默认级联（级联最稳）。"""
        cfg = services.get("kv_store").read("route_config", session_id)
        return cfg.get("route", ROUTE_CASCADE)

    def switch(self, session_id: str, route: str) -> str:
        """热切换路由，切换不清聊天记录，用户无感。"""
        services.get("kv_store").write(
            "route_config",
            session_id,
            {"route": route, "fail_count": 0, "auto_degrade": True},
        )
        return route

    def report_failure(self, session_id: str) -> None:
        """上报一次失败，连续挂够次数且开了自动降级，就切到级联保命。"""
        kv = services.get("kv_store")
        cfg = kv.read("route_config", session_id) or {}
        count = int(cfg.get("fail_count", 0)) + 1
        cfg["fail_count"] = count
        if cfg.get("auto_degrade", True) and count >= self._FAIL_LIMIT:
            cfg["route"] = ROUTE_CASCADE
            cfg["fail_count"] = 0
        kv.write("route_config", session_id, cfg)


class InfoGapCoordinator:
    """缺信息时的追问循环：问一句 → 用户补 → 写进档案。"""

    # 各字段的追问话术；persona_config.json 的 ask_templates 同键可覆盖，换人设不用改代码
    _ASK_TEMPLATES = {
        "city": "对了，想帮你查天气的话，得先知道你在哪个城市？",
        "birthday": "你生日是什么时候？我想记下来。",
        "nickname": "该怎么称呼你？总得有个叫法吧。",
        "occupation": "你平时是做什么的？说说呗。",
    }

    def ask(self, missing_field: str, session_id: str) -> str:
        """生成追问话术：人设文件里自定义了就用自定义的，否则用内置模板。"""
        templates = dict(self._ASK_TEMPLATES)
        try:
            persona = services.get("kv_store").read("persona_config", "")
            templates.update(getattr(persona, "ask_templates", None) or {})
        except Exception:
            pass  # 人设读不到就用内置默认，追问照常
        return templates.get(
            missing_field, f"对了，你的{missing_field}是什么？告诉我一声。"
        )

    def supply(self, missing_field: str, value: str, session_id: str) -> bool:
        """用户补充的信息写进档案，写成功返回真。"""
        return ProfileUpdater().set_field(session_id, missing_field, value)


class FallbackController:
    """异常降级决策中心：什么错、重试过几次，给个说法。"""

    # 各错误类型的内置兜底话；persona_config.json 的 fallback_replies 同键可覆盖。
    # 兜底话本身就跑在出错路径上，人设读不到必须无缝落回内置默认，不能再抛异常。
    # 沉浸式措辞：她是"人"，台词里绝不出现"工具"这类词（AI 腔 + 穿帮，双杀）
    _REPLIES = {
        "llm": "……这会儿状态不太对，等一下再聊。",
        "voice_route": "语音这边不太顺畅，先打字聊吧。",
        "search": "呃……这个我还真不知道，你问住我了。",
        "image": "图没弄出来，晚点再试试。",
        "tool_call": "这事儿这次没办成，回头再说吧。",
        "blocked": "这个话题我不聊，换一个吧。",
        "review": "刚才想说什么来着……算了，换件事说吧。",
    }

    def fallback_reply(self, error_type: str) -> str:
        """按错误类型给一句符合人设的兜底话。"""
        replies = dict(self._REPLIES)
        try:
            persona = services.get("kv_store").read("persona_config", "")
            replies.update(getattr(persona, "fallback_replies", None) or {})
        except Exception:
            pass  # 出错路径上的人设读取再挂掉，也不影响兜底话出口
        return replies.get(error_type, replies["llm"])
