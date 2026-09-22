"""写回协调器（R17a 从 orchestration/pipeline.py 机械提取）。

**本批是纯提取**：代码逐行搬运、行为逐行保持（含 R02 已禁止原稿写回、
R14 已修的正文规则），不恢复任何已修缺陷，也不做语义切换——降级轮副作用闸、
commit_turn 接线是 R17b 的活。

为什么收成协调器：写回是"她这轮说过的话落到哪几处"的唯一裁决点
（KEEPER / chat_log / 蒸馏 / 关系 / 收手环 / 后台待办），原先长在 pipeline
的方法体里，测试与后续的提交契约（R15c/R17b）都够不着它。协调器把依赖
（KEEPER、取消句柄）做成注入——全项目仍只有 pipeline 顶层那**一份**
KEEPER（8.3 不变式），这里不新建、不复制。

分层：orchestration → capability / tools / shared，全部向下，无环
（pipeline import 本文件，本文件**不** import pipeline——KEEPER 靠注入）。
"""

from __future__ import annotations

import re

from capability.memory import (
    ConversationDistiller,
    PortraitBuilder,
    RelationshipTracker,
)
from capability import self_identity
from config.settings import load_app_config
from shared.singletons import services
from shared.types import DeferredTask
from tools.speech import split_emotion, strip_emotion_marks

# R18a：后台待办的任务种类。载荷绑定 (turn_id, kind) 幂等键——同一轮的
# 同类任务重启/重试最多登记一次。
TASK_PORTRAIT = "portrait"
TASK_IDENTITY_FREEZE = "identity_freeze"
TASK_SUMMARY = "summary"

# 内存调度登记：task_key -> {"turn_id", "kind"}。R18a 明确**只做内存调度**，
# 进程退出即丢（持久化与恢复是 R19 的活）——这里绝不谎称已可靠投递。
_COMMIT_TASK_REGISTRY: dict = {}


def make_task_key(turn_id: str, kind: str, target: str = "") -> str:
    """派生任务的幂等键：源轮 + 种类 + 目标（R18b 会加目标版本/来源范围）。"""
    return f"{turn_id}:{kind}:{target}" if target else f"{turn_id}:{kind}"


def schedule_commit_tasks(receipt: dict, tasks: list) -> list:
    """R18a 新调度入口：**只认 committed 回执**——未提交/取消/失败/降级轮
    的待办一律不调度。返回本次实际要执行的 fn 列表（去重后）。

    不能继续把"旧 handle 被新轮取消"当作丢弃旧已提交任务的理由：
    调度资格只看回执，不看 handle 状态（取消语义归 CommitGate）。
    """
    if receipt.get("status") != "committed":
        return []
    turn_id = receipt.get("turn_id") or ""
    runnable = []
    for task in tasks:
        key = make_task_key(turn_id, task.kind)
        if key in _COMMIT_TASK_REGISTRY:
            continue  # 幂等：同轮同类任务只登记一次
        _COMMIT_TASK_REGISTRY[key] = {"turn_id": turn_id, "kind": task.kind}
        runnable.append(task.fn)
    return runnable


def pending_commit_tasks() -> dict:
    """内存登记表只读视图（测试/观测用）。"""
    return dict(_COMMIT_TASK_REGISTRY)


def _expression_cfg() -> dict:
    """expression 配置段的安全读取：读不到给空 dict，调用方各自兜底。"""
    try:
        return load_app_config().get("expression", {}) or {}
    except Exception:
        return {}


def _strip_reaction_tag(text: str) -> str:
    """剥掉回复开头的反应前缀（@r 之类），落库和短期记忆里不许留标记。

    _ReplyStreamer 推送前会剥一次，但 state.final_reply 存的是模型原始输出——
    写回前必须再过一遍，否则标记会进 chat_log 和短期记忆污染后续上下文。
    """
    tag = (_expression_cfg().get("reaction_tag") or "@r").strip()
    t = (text or "").lstrip()
    if t.startswith(tag):
        return t[len(tag):].lstrip()
    return text


def _portrait_interval() -> int:
    """几轮聊天更新一次画像，从配置里读，读不到就 5 轮。"""
    try:
        return int(load_app_config()["memory"].get("session_update_interval", 5)) or 5
    except Exception:
        return 5


def _check_cancel(handle) -> None:
    """协作式取消检查点：被顶掉就抛 TurnCancelled 让上层收场。"""
    from orchestration.cancellation import TurnCancelled

    if handle is not None and handle.is_cancelled():
        raise TurnCancelled(f"turn {handle.turn_id} cancelled")


class WritebackCoordinator:
    """把"她这轮说过的话"落到该落的地方：KEEPER / chat_log / 蒸馏 / 关系 /
    收手环，并把画像/身份冻结/摘要的后台待办打包返回（由持有 running loop
    的一方调度，I11）。

    keeper 注入：必须是 pipeline 顶层的全项目唯一 KEEPER（8.3）。
    """

    def __init__(self, keeper):
        self._keeper = keeper

    def writeback(self, state, handle) -> list:
        """记忆写回。短期记忆、长期沉淀、亲密度、画像，一样别落。

        取消检查点插在每个子步骤之前：迟到的取消（被新轮顶掉）不会写出半截账。
        注意：一旦取消，本方法**什么都不写**——不写 KEEPER、不写聊天记录、
        不蒸馏、不涨亲密度、不刷画像。这偏离了设计文档里
        "chat_log 可补一条 cancelled 标记"的说法，是**有意选择**：
        写任何东西都会让"这轮当没发生过"不彻底，而 chat_log 的两个用途
        （写日记、重启后恢复短期记忆）都不需要一条中途被丢弃的痕迹。
        实际也很少出现"取消但没被新轮接管"——TurnRegistry.start 的会话级
        单轮不变式会让新轮先取消旧轮，旧轮本来就不该留痕。
        """
        session_id = state.session_id
        # I13 refuse-only 分支：refuse 路径（_RefuseTurn 被外层 except 接走）的写回，
        # final_reply / draft_reply 都空说明这轮没产出正文——但 was_poor=True 仍要
        # 落库，否则 force_if_last_poor（见 :247-250）在降级后下一轮永远不触发。
        # 机制只在"轮内重写成功"时生效，与 _think_needed 注释意图不符，而这
        # 恰恰是最需要它的时候。其他副作用一律不走：KEEPER / chat_log / 蒸馏 / 画像
        # 都是"这轮说过话"的产物，refuse 没说过。
        if not (state.final_reply or state.draft_reply):
            _check_cancel(handle)
            tracker = RelationshipTracker()
            tracker.update(
                state.session_id,
                state.emotion.emotion if state.emotion else "neutral",
                comfort_mode=state.comfort_mode,
                was_poor=state.was_poor,
                intensity=state.emotion.intensity if state.emotion else 0.5,
                extras=state.extras,
            )
            _check_cancel(handle)
            try:
                from capability.proactive import note_user_reply
                note_user_reply(state.session_id)
            except Exception:
                pass
            return []

        user_text = state.user_text
        # 普通聊天的回复还停在初稿里，先定稿再写回；标记落库前剥掉——
        # final_reply 存的是模型原始输出，不剥的话标记会进 chat_log 和短期记忆
        #
        # 剥离顺序**必须先剥情绪标记、再剥反应前缀**（F）：
        # _strip_reaction_tag 只看字符串开头，语音轮模型输出 "[开心]@r 你好" 时，
        # 开头是 '[' 而非 '@r'，反过来的顺序会让 @r 逃过剥离，随后 strip_emotion_marks
        # 只剥情绪标记、把 @r 留在正文 → 落库 "@r 你好" 而前端推送 "你好"，
        # @r 进 chat_log 和短期记忆还会污染后续上下文（角色读到自己上一轮的 @r）。
        reply_raw = state.final_reply or state.draft_reply
        # 先记录情绪前缀（只取开头第一个，剥之前拿）：语音轮要把它拼回 final_reply
        _, emo_tags, emo_descs = split_emotion(reply_raw)
        # 干净正文：先剥情绪标记、再剥反应前缀。剥完可能前头又露出反应前缀
        # （极端情况 "@r [开心] 你好"），所以在小循环里交替剥直到稳定，
        # 保证落库文本里既没有 @r 也没有 [情绪]。
        reply = reply_raw
        for _ in range(3):
            cleaned = _strip_reaction_tag(strip_emotion_marks(reply))
            if cleaned == reply:
                break
            reply = cleaned
        # 语音轮：final_reply 必须**保留情绪标签**——上层（renderer / voice 事件）
        # 拿它去合成语音、拆情绪。上面剥干净的是"落库用"的干净文本，
        # 这里把被剥掉的情绪前缀原样拼回 final_reply，别把标签一起削掉。
        if state.voice_mode:
            prefix = f"[{emo_tags[0]}]" if emo_tags else ""
            prefix += f"（{emo_descs[0]}）" if emo_descs else ""
            state.final_reply = prefix + reply
        else:
            state.final_reply = reply
        # 短期记忆和 chat_log 只收剥干净标记的正文（语音轮同样如此，情绪标签不进库）
        clean_reply = strip_emotion_marks(reply) if state.voice_mode else reply
        emotion = state.emotion

        # 短期记忆成对追加这一轮的问答（F12）：一次加锁写 user+assistant 两条，
        # 迟到的取消不会留下"只有问没有答"的半截记忆
        _check_cancel(handle)
        self._keeper.append_turn(session_id, user_text, clean_reply)

        # 聊天记录落库（chat_log 表）：日记生成、重启恢复都靠这份数据
        # 危机轮也要记——对话发生过就该留痕，只是不涨亲密度不沉淀记忆
        _check_cancel(handle)
        intent = state.intent
        emotion_label = emotion.emotion if emotion else ""
        kv = services.get("kv_store")
        kv.write(
            "session",
            session_id,
            {
                "role": "user",
                "text": user_text,
                "intent": intent.intent if intent else "",
                "emotion": emotion_label,
                "mode": "voice" if state.voice_mode else "text",
            },
        )
        if clean_reply:
            kv.write(
                "session",
                session_id,
                {
                    "role": "assistant",
                    "text": clean_reply,
                    "intent": intent.intent if intent else "",
                    "emotion": emotion_label,
                    "mode": state.output_mode,
                },
            )

        # 危机或婉拒的轮次不沉淀、不涨亲密度，这些轮次不算正常互动
        if emotion and emotion.is_crisis:
            return []  # 无后台待办

        _check_cancel(handle)
        distiller = ConversationDistiller()
        distiller.distill_turn(session_id, user_text)

        _check_cancel(handle)
        # 关系数值 + 心情状态机 + 阶段标记 + last_poor + 账本，一次原子闭包全部落好（F3 收口）。
        # 以前这里写两次库（tracker 一次、mood 一次），既可能被打断也可能互相覆盖。
        # last_poor（F8）必须传进闭包**一起写**，绝不能在 update 返回后再整包 kv.write：
        # 那正是 F3 要消灭的"陈旧整包覆盖"，并发 REST 的 intimacy 增量会被吞掉。
        tracker = RelationshipTracker()
        rel = tracker.update(
            session_id,
            emotion.emotion if emotion else "neutral",
            comfort_mode=state.comfort_mode,
            was_poor=state.was_poor,
            intensity=emotion.intensity if emotion else 0.5,
            extras=state.extras,
            # C6 反向标定：传她这轮**真说出口**的正文，安抚话能把误判的负面心情
            # 校回来。用 clean_reply 不用 final_reply——后者还带着语音轮的情绪标签。
            utterance_text=clean_reply,
        )

        # 用户开口 = 她的主动消息被回应了（N3 收手环）：清等待标记、重置连击。
        # 必须在 tracker 之后——放前面会在首次对话预创建一个空 relationship，
        # 让 tracker 的"首次初始化"分支失效（default_intimacy 被吞成 0，实测踩中）
        _check_cancel(handle)
        try:
            from capability.proactive import note_user_reply

            note_user_reply(session_id)
        except Exception:
            pass

        # 聊够几轮才重新画像，别一句话就给人家贴标签。
        # 画像与身份冻结都是 LLM 秒级调用——阻塞在 done 之前会让"她说完了"
        # 你还要等她做完笔记。这里**只把待办闭包打包返回**，由 astream（持有
        # running loop 的一方）create_task 调度——写回自己跑在工作线程里，
        # 线程内没有事件循环，在这里调度会退化成同步执行（I11，实测踩中）。
        deferred: list = []  # DeferredTask 载荷（R18a）：绑定 (turn_id, kind) 幂等键
        interval = _portrait_interval()
        if rel.get("interaction_count", 0) % interval == 0:
            context_snapshot = self._keeper.get_context(session_id)[-6:]

            def _bg_portrait(handle=handle, sid=session_id, ctx=context_snapshot):
                if handle is not None and handle.is_cancelled():
                    return  # 迟到的取消：什么都不写
                PortraitBuilder().refresh(sid, ctx)

            deferred.append(DeferredTask(kind=TASK_PORTRAIT, fn=_bg_portrait))

        # 她自己的身份冻结（批次0"种子+涌现"）：从她刚说的话里定下名字/年龄/
        # 城市/职业/住处（只收她亲口说的、只填空不改口），第 3 轮后提炼一次
        # 自我认知基线。LLM 失败静默跳过，下轮再试；下一轮 compose 注入"你是谁"
        her_lines = [
            m.get("content", "")
            for m in self._keeper.get_context(session_id)
            if m.get("role") == "assistant"
        ]
        freeze_text = re.sub(r"</?voice>", "", clean_reply or "")

        # user_text 是 G1-G4 披露预算闸门的输入（他这轮问过什么 → 她可以冻什么）。
        # 不传就是闸门按"问过"放行——compose 侧已经不催她倒档案了，冻结侧不接
        # 就只剩半边生效。
        def _bg_freeze(handle=handle, sid=session_id, text=freeze_text, lines=her_lines,
                       count=rel.get("interaction_count", 0), asked=user_text):
            if handle is not None and handle.is_cancelled():
                return  # 迟到的取消：什么都不写
            self_identity.maybe_freeze(sid, text, recent_her_lines=lines,
                                       interaction_count=count, user_text=asked)

        deferred.append(DeferredTask(kind=TASK_IDENTITY_FREEZE, fn=_bg_freeze))

        # J12：短期记忆的摘要不再内联在 append_turn 里（那是一次秒级 LLM 调用，
        # 阻塞在写回路径上）。append_turn 只登记待办，这里打包成后台闭包，
        # 由持有 running loop 的一方调度——工作线程内没有事件循环（I11）。
        if self._keeper.has_pending_summary(session_id):
            def _bg_summary(handle=handle, sid=session_id):
                if handle is not None and handle.is_cancelled():
                    return  # 迟到的取消：什么都不写
                self._keeper.run_pending_summary(sid)

            deferred.append(DeferredTask(kind=TASK_SUMMARY, fn=_bg_summary))
        return deferred
