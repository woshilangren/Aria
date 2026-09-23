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

    R18b：幂等升级为**存储原子裁决**——每个待办包一层领取守卫（claim→执行→
    失败退回），任务键带目标范围。重试/重放/重启后同键任务最多应用一次；
    并发领取由 derived_task_done 的主键裁决，不靠进程内约定。
    """
    if receipt.get("status") != "committed":
        return []
    turn_id = receipt.get("turn_id") or ""
    runnable = []
    for task in tasks:
        key = make_task_key(turn_id, task.kind, getattr(task, "target", ""))
        if key in _COMMIT_TASK_REGISTRY:
            continue  # 幂等：同轮同类任务只登记一次
        _COMMIT_TASK_REGISTRY[key] = {"turn_id": turn_id, "kind": task.kind}
        runnable.append(_claimed(key, task.fn))
    return runnable


def _claimed(key: str, fn) -> object:
    """给待办包领取守卫（R18b）：领取成功才执行，失败退回标记再抛。

    生成（LLM 抽取等）可重复，但"应用结果"必须幂等：同键任务第二次进来
    领取失败直接跳过，不重复命中候选、不重复修订身份、不重复追加事件记忆。

    R19b：应用成功后把 pending_writes 里同 ID 的登记行标 done——该任务不再
    会被消费者重试；应用失败行留在 pending（attempts 由消费者累计）。
    """

    def _run():
        from data.sqlite_store import get_db

        db = get_db()
        if not db.claim_task(key):
            return  # 已被应用过（重放/并发领取失败）：什么都不做
        try:
            fn()
        except BaseException:
            db.release_task(key)  # 领取≠完成：失败退回，下次还能重试
            raise
        db.finish_pending(key)

    return _run


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
        # R17c：emotion 为空 = 调用方明确"未知"（语音路由常有）——不许在门内
        # 伪造 neutral 领取关系更新资格：该轮照常落历史与学习，但无关系增量、
        # 无账本。
        normal = turn.disposition == "normal"
        has_emotion = bool((turn.emotion or "").strip())
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
            relation_fn=self._relation_fn(turn) if (normal and has_emotion) else None,
            relation_ledger=(self._LEDGER_REASONS.get(turn.emotion or "neutral", "又聊了一轮")
                             if (normal and has_emotion) else ""),
            ledger_event_id=turn.turn_id,
            pending_tasks=self._planned_tasks(turn, has_emotion),
        ))

    def _planned_tasks(self, turn: PreparedTurn, has_emotion: bool) -> list:
        """R19b：预先可知的后台任务清单——作为 commit_turn 的提交参数在同一
        短事务里 INSERT（禁止提交后单独 enqueue 冒充原子登记）。执行仍在提交后。

        与 post_commit 的 DeferredTask 用**同一个 task_id**（make_task_key）：
        执行成功 → pending 行标 done；失败/进程死 → 行留在 pending 给 R19c
        消费者按预算重试。
        """
        if turn.disposition != "normal":
            return []
        tasks = []
        try:
            rel = self._kv().read("relationship", turn.session_id) or {}
            # 正常轮提交即 interaction_count+1（R15b 纯计算），画像节拍按
            # 提交后的计数判定——与 post_commit 原判定口径一致
            new_count = rel.get("interaction_count", 0) + 1
            if new_count % _portrait_interval() == 0:
                tasks.append(self._pending_desc(
                    turn, TASK_PORTRAIT,
                    payload={"session_id": turn.session_id,
                             "window_scope": "learnable_last_6"}))
        except Exception as exc:
            print(f"[writeback] 画像任务登记预判失败（本轮不登记画像任务）: {exc}")
        tasks.append(self._pending_desc(
            turn, TASK_IDENTITY_FREEZE,
            payload={"session_id": turn.session_id, "has_emotion": has_emotion}))
        try:
            if self._keeper.has_pending_summary(turn.session_id):
                tasks.append(self._pending_desc(
                    turn, TASK_SUMMARY,
                    payload={"session_id": turn.session_id}))
        except Exception as exc:
            print(f"[writeback] 摘要任务登记预判失败（本轮不登记摘要任务）: {exc}")
        return tasks

    @staticmethod
    def _pending_desc(turn: PreparedTurn, kind: str, payload: dict) -> dict:
        return {
            "task_id": make_task_key(turn.turn_id, kind),
            "kind": kind,
            "source_turn_id": turn.turn_id,
            "payload": payload,
        }

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
            # R17d：提交前不落的隐式业务更新在这里补账——只有提交成功的正常轮
            # 才有资格记召回热度、消耗生活素材；取消/降级/危机轮什么都留不下。
            # 回执 already_committed 的重试轮 post_commit 直接早退（上面），
            # 所以每种效果天然最多一次。
            if turn.memory_ids:
                try:
                    from data.sqlite_store import get_db

                    get_db().touch_memories(list(turn.memory_ids))
                except Exception as exc:
                    print(f"[writeback] 召回热度补记失败（不影响本轮）: {exc}")
            if turn.burn_topic:
                try:
                    from capability import char_life

                    char_life.commit_topic(turn.session_id, turn.burn_topic)
                except Exception as exc:
                    print(f"[writeback] 生活素材烧计数失败（不影响本轮）: {exc}")
            rel = kv.read("relationship", turn.session_id) or {}
            tasks: list = []
            interval = _portrait_interval()
            # R14c：画像/身份的学习素材只取**可学习的已提交正常轮**——
            # KEEPER 上下文里混着降级轮的兜底正文（模板话），拿去做人格证据
            # 等于把"这轮没接上"炼成她的性格。走 learnable 读口直查持久层，
            # legacy 行（分类字段为空）按可学习对待。
            learnable = kv.recent_chat(turn.session_id, 100, learnable=True)
            if rel.get("interaction_count", 0) % interval == 0:
                ctx = [
                    {"role": r.get("role"), "content": r.get("text", ""),
                     "id": r.get("id")}  # R11b：消息 id 供证据归属
                    for r in learnable[-6:]
                ]
                tasks.append(DeferredTask(
                    kind=TASK_PORTRAIT,
                    fn=lambda: PortraitBuilder().refresh(turn.session_id, ctx)))
            her_lines = [
                r.get("text", "")
                for r in learnable
                if r.get("role") == "assistant"
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
