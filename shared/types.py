"""各层之间传来传去的数据类型，全定义在这。

字段名就是契约：上层生产、下层消费，随便改一个字段名就会断链路。
"""

from dataclasses import dataclass, field
from typing import Optional


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
    """最终回复：文字为主，带图的时候补图片路径。"""

    text: str
    output_mode: str = "text"  # text / image
    image_path: str = ""


@dataclass
class InputMessage:
    """一条收进来的用户消息。"""

    text: str
    input_mode: str = "text"   # text / voice
    session_id: str = "default"
    image_url: str = ""        # 用户随消息上传的图片（可选），多模态模型跟着一起看
