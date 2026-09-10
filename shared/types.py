"""各层之间传来传去的数据类型，全定义在这。

字段名就是契约：上层生产、下层消费，随便改一个字段名就会断链路。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonaPrompt:
    """拼好的一份提示词，喂给 LLM 用。"""

    text: str


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
    tone_mode: str = "chat"


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
    timestamp: str = ""
    image_url: str = ""        # 用户随消息上传的图片（可选），多模态模型跟着一起看
