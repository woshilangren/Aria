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
    compute_relationship_update,
)
from capability import self_identity
from config.settings import load_app_config
from shared.singletons import services
from shared.types import DeferredTask, PreparedTurn
from tools.speech import strip_emotion_marks

from orchestration.cancellation import COMMIT_GATE

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
    """R17b：文字轮的新契约写回——提交（CommitGate→commit_turn 事务）先行，
    提交成功才动 KEEPER 与派生任务；副作用按 disposition 分流。

    keeper 注入：必须是 pipeline 顶层的全项目唯一 KEEPER（8.3）。
    """

    _LEDGER_REASONS = {
        "happy": "聊得开心，关系热乎了一点",
        "sad": "他说了难过的事，信任多了一分",
        "crisis": "陪他熬过了一段艰难的时刻",
        "angry": "闹了点不愉快，热度降了一点",
        "tired": "平平常常聊了一会儿",
    }

    def __init__(self, keeper, gate=None):
        self._keeper = keeper
        self._gate = gate or COMMIT_GATE

    def _kv(self):
        return services.get("kv_store")

    def _default_intimacy(self) -> int:
        """首次初始化的种子默认亲密度（读人设文件，不经库锁）。"""
        try:
            return self._kv().read("persona_config", "").default_intimacy
        except Exception as exc:
            print(f"[writeback] 读初始亲密度失败（按 0 起）: {exc}")
            return 0

    def _mood_engine(self):
        from capability.quirks import MoodEngine

        return MoodEngine()

    def _relation_fn(self, turn: PreparedTurn):
        """normal 轮的关系纯计算（R15b），注入 commit_turn 事务内执行。"""
        def _apply(rel):
            new_rel, _ = compute_relationship_update(
                rel, emotion=turn.emotion or "neutral",
                comfort_mode=turn.comfort_mode, was_poor=turn.was_poor,
                intensity=turn.intensity, extras=turn.extras,
                utterance_text=turn.utterance_text,
                mood_engine=self._mood_engine(),
                default_intimacy=self._default_intimacy(),
            )
            return new_rel
        return _apply

    def commit_prepared(self, turn: PreparedTurn) -> dict:
        """R17b：先在门上裁决取消（取消先赢→数据事务根本不开始），未取消才
        跑 kv.commit_turn 数据事务。

        副作用表（8.11.2）按 disposition 分流：
        - normal：关系纯计算 + intimacy 账本（与正式记录同一事务）；
        - degraded / crisis：只落正式记录——无关系增量、无账本、无学习任务。
        """
        return self._gate.run_commit(turn.turn_id, fn=lambda: self._kv().commit_turn(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            request_id=turn.request_id,
            request_digest=turn.request_digest,
            disposition=turn.disposition,
            source_review_status=turn.source_review_status,
            reason_code=turn.reason_code,
            user_text=turn.user_text,
            assistant_text=turn.assistant_text,
            intent=turn.intent,
            emotion=turn.emotion,
            mode=turn.mode,
            relation_fn=self._relation_fn(turn) if turn.disposition == "normal" else None,
            relation_ledger=(self._LEDGER_REASONS.get(turn.emotion or "neutral", "又聊了一轮")
                             if turn.disposition == "normal" else ""),
            ledger_event_id=turn.turn_id,
        ))

    def post_commit(self, turn: PreparedTurn, receipt: dict) -> list:
        """提交成功后的内存与派生动作（8.11.3 原子边界 5：SQLite 成功才更新
        KEEPER）。返回经新入口去重后的可执行 fn 列表（由持有 loop 的一方调度）。

        - normal：KEEPER 追加 + 蒸馏 + 收手环 + 画像/冻结/摘要任务（新入口调度）；
        - degraded：KEEPER 追加 + 质量标志 bookkeeping + 收手环，**零学习任务**；
        - crisis：KEEPER 追加（对话发生过该留痕），零学习任务；
        - cancelled / failed：一律不动。
        """
        if receipt.get("status") != "committed":
            return []
        self._keeper.append_turn(turn.session_id, turn.user_text, turn.assistant_text)
        kv = self._kv()
        if turn.disposition == "normal":
            ConversationDistiller().distill_turn(turn.session_id, turn.user_text)
            try:
                from capability.proactive import note_user_reply

                note_user_reply(turn.session_id)
            except Exception:
                pass
            rel = kv.read("relationship", turn.session_id) or {}
            tasks: list = []
            interval = _portrait_interval()
            if rel.get("interaction_count", 0) % interval == 0:
                ctx = self._keeper.get_context(turn.session_id)[-6:]
                tasks.append(DeferredTask(
                    kind=TASK_PORTRAIT,
                    fn=lambda: PortraitBuilder().refresh(turn.session_id, ctx)))
            her_lines = [
                m.get("content", "")
                for m in self._keeper.get_context(turn.session_id)
                if m.get("role") == "assistant"
            ]
            count = rel.get("interaction_count", 0)
            tasks.append(DeferredTask(
                kind=TASK_IDENTITY_FREEZE,
                fn=lambda: self_identity.maybe_freeze(
                    turn.session_id, turn.assistant_text,
                    recent_her_lines=her_lines, interaction_count=count,
                    user_text=turn.user_text)))
            if self._keeper.has_pending_summary(turn.session_id):
                tasks.append(DeferredTask(
                    kind=TASK_SUMMARY,
                    fn=lambda: self._keeper.run_pending_summary(turn.session_id)))
            return schedule_commit_tasks(receipt, tasks)
        if turn.disposition == "degraded":
            # 副作用表：质量标志 + 收手环 bookkeeping——零关系增量、零学习任务
            RelationshipTracker().update_bookkeeping_only(
                turn.session_id, was_poor=turn.was_poor)
            try:
                from capability.proactive import note_user_reply

                note_user_reply(turn.session_id)
            except Exception:
                pass
            return []
        return []  # crisis
