"""各层之间传来传去的数据类型，全定义在这。

字段名就是契约：上层生产、下层消费，随便改一个字段名就会断链路。
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class DeferredTask:
    """后台待办载荷（R18a）：绑定任务种类与可调用体。

    kind ∈ portrait / identity_freeze / summary——调度入口以
    (turn_id, kind) 造幂等键；fn 是无参闭包（后台线程执行）。
    只属于**已拿到 committed 回执的轮**：未提交/取消/降级轮不产生载荷。
    """

    kind: str
    fn: Callable


@dataclass
class CommitReceipt:
    """一轮提交的回执（R15d，8.11.3 的正式类型）。

    status ∈ committed / already_committed / cancelled / conflict /
    processing / failed。处理中（committing）不许伪造 committed_at；
    already_committed 必须指向原回执的 message_ids 与 committed_at。
    取消与提交通过**同一原子门**裁决后产生本类型（CommitGate）。
    """

    status: str
    session_id: str
    request_id: str
    turn_id: str
    disposition: str = "normal"
    committed_at: Optional[str] = None
    message_ids: list = field(default_factory=list)
    reason_code: str = ""


class ExternalServiceError(RuntimeError):
    """外部服务失败（R27b 窄错误载体），由各适配器在**最窄的边界**转换而来。

    - source：哪个服务（llm / asr / tts / vector …）；reason_code：有限原因码；
    - retryable：是否值得自动重试；detail 只进日志，**不进任何对外字段**。

    下游只对这一类异常进入备用/模板降级（兜底哲学：兜底只适用外部服务失败）。
    NameError / TypeError / AttributeError 等本地缺陷**绝不许**包装成它——
    那类错误走 PipelineInternalError 的路：向上冒泡、生命周期失败。
    """

    def __init__(self, source: str, reason_code: str,
                 retryable: bool = True, detail: str = ""):
        super().__init__(f"[{source}/{reason_code}]")
        self.source = source
        self.reason_code = reason_code
        self.retryable = retryable
        self.detail = detail


class PipelineInternalError(RuntimeError):
    """本地编程缺陷冒泡到请求边界的载体（R27b）。

    对用户只返回安全的 internal_error；生命周期失败：不重跑模型/付费工具、
    不作为 normal/degraded 学习轮保存。`handle()` 的 error 消费分支必须
    保留它——不能重新包成一个看似成功的 FinalReply 兜底（那会把 bug 变成
    "她说了句奇怪的话"，和 A2 事故是同一类病）。
    """


@dataclass
class MemoryBundle:
    """一次回忆打捞上来的所有东西。"""

    profile: dict = field(default_factory=dict)        # 硬事实档案
    portrait: dict = field(default_factory=dict)       # 软画像
    relationship: dict = field(default_factory=dict)   # 关系数值
    distilled: list = field(default_factory=list)      # 长期记忆条目
    diaries: list = field(default_factory=list)        # 语义召回的日记条目


@dataclass
class IntentResult:
    """意图识别结果。"""

    intent: str          # chat / comfort / weather / search / image / info_supply
    confidence: float = 1.0


@dataclass
class EmotionResult:
    """情绪分析结果。"""

    emotion: str = "neutral"
    intensity: float = 0.3
    is_crisis: bool = False


@dataclass
class PerceptionExtras:
    """感知的附加产出（S2/N2，与意图/情绪/潜台词同一次 LLM 调用带出）。

    concern_*：会压在她心里、影响之后几轮的念头（文本 + 本轮强度增减）；
    feedback：他在抱怨她的态度/语气时为 "tone_down"，进关系收敛层（calm），
    绝不改性格内核——"收敛不是改变"是批次的宪法。
    same_concern / resolved（C7）：心事是**实体**不是自由文本，这两个是它的换挡
    信号——延续同一件（强度走惯性）还是关旧开新，以及这轮有没有把它化解掉。
    """

    concern_text: str = ""
    concern_delta: float = 0.0
    feedback: str = ""
    # 缺字段时 same_concern 默认 true：延续是保守读法，强度惯性接续总比凭空
    # 开一个新实体稳（否则同一件事每轮都被当成新心事，强度永远从头涨）。
    same_concern: bool = True
    resolved: bool = False


@dataclass
class ToolCallSpec:
    """一次工具调用计划：调哪个工具、带什么参数。"""

    tool_name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class ToolCallResult:
    """工具执行结果。"""

    tool_name: str
    status: str  # ok / error
    data: object = None


@dataclass
class PromptPackage:
    """生成回复前备齐的一整套材料。"""

    system_prompt: str
    context_messages: list = field(default_factory=list)  # 短期记忆，[{"role","content"},...]


@dataclass
class FinalReply:
    """最终回复：文字为主，带图的时候补图片路径。

    R14a 正文契约（8.11.1 的最小载体）：
    - `text` = canonical_text（唯一正式正文）：已剥净 <voice>/情绪标记等协议，
      不再包含"语音版"内容——两段拼回正文的做法在此终结；
    - `voice_text` = 语音派生：模型自标的 <voice> 内容，允许口语/节奏与正文
      不同；**不能核验同义时，调用方直接拿 text 做 TTS**（契约明文允许）；
    - `review_status` / `reason_code`：原生成内容的审核结果（accepted /
      rejected / unavailable）与有限原因码——与生命周期无关，R27 原因码体系复用。
    draft / 工具结果不在这张表里：它们不具备持久化资格（8.11.1）。
    """

    text: str
    output_mode: str = "text"  # text / image
    image_path: str = ""
    voice_text: str = ""          # 语音派生正文；空 = 无独立语音表达
    review_status: str = "accepted"
    reason_code: str = ""
    # R14b 发布确认（8.11.1 规则 6）：发送与接收是不同事实。sent = 交互适配器
    # 已把回复交给发送层；partial/failed/unknown 由适配器按实际填写。
    # handle() 在内存里攒出 FinalReply 不算发布——这个字段只能由适配器写。
    delivery: str = "unknown"


@dataclass
class InputMessage:
    """一条收进来的用户消息。"""

    text: str
    input_mode: str = "text"   # text / voice
    session_id: str = "default"
    image_url: str = ""        # 用户随消息上传的图片（可选），多模态模型跟着一起看
