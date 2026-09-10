"""数据结构定义：人设、记忆、关系、聊天记录这些东西长什么样，全在这。

各层之间传数据就靠这些类，字段名定死了，谁都不能乱改——改了前后就对不上了。
"""

from dataclasses import dataclass, field


@dataclass
class PersonaConfig:
    """完整人设：从 data/persona_config.json 读进来的那张总表。"""

    char_id: str
    char_name: str
    self_introduction: str
    age: str
    hobbies: list
    background_story: str
    default_intimacy: int
    daily_reset: bool
    mode_tones: dict
    memory_config: dict
    taboos: str = ""  # 硬性规矩，说话不许越过的线
    # 下面三个是可选的口吻接口：换人设时按需填，不填走代码里的内置默认。
    # 目的是把"角色怎么说"全部收进 persona_config.json，改口吻不用改代码。
    fallback_replies: dict = field(default_factory=dict)  # 异常兜底话术，键见 FallbackController
    ask_templates: dict = field(default_factory=dict)  # 缺信息追问话术，键：city/birthday/nickname/occupation
    diary_notes: str = ""  # 写日记时的口吻补充要求


@dataclass
class RelationshipState:
    """关系状态：亲密度这些数值，聊一次变一点。"""

    intimacy: int = 0
    affection: int = 0
    trust: int = 0
    mood_baseline: str = "平静"
    interaction_count: int = 0
    stage: str = "初识"
    last_interaction: str = ""


@dataclass
class MemoryItem:
    """一条被沉淀下来的长期记忆。"""

    memory_id: str
    session_id: str
    kind: str  # hard_fact / preference / event / daily_summary
    content: str
    importance: int = 3  # 1~5，越大越重要
    timestamp: str = ""


@dataclass
class DialogueRecord:
    """一句话的聊天记录，往对话历史里存的那种。"""

    role: str  # user / assistant
    text: str
    intent: str = ""
    emotion: str = ""
    mode: str = "text"


@dataclass
class ImageAsset:
    """一张生成过的图：路径加说明，方便回看。"""

    image_path: str
    caption: str = ""
    created_at: str = ""
