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
    subtext_hints: dict = field(default_factory=dict)  # 潜台词词典：他自己的反话/弦外之音，模型猜不出的那部分
    diary_notes: str = ""  # 写日记时的口吻补充要求


@dataclass
class RelationshipState:
    """关系状态：亲密度这些数值，聊一次变一点。"""

    intimacy: int = 0
    affection: int = 0
    trust: int = 0
    mood: str = "平静"  # MoodEngine 状态机输出（有累积/衰减语义），前端"当前情绪"读这个
    interaction_count: int = 0
    stage: str = "初识"
    last_interaction: str = ""


@dataclass
class MemoryItem:
    """一条被沉淀下来的长期记忆。

    感受字段（S1，批次6）：生成时想一次就**冻死**，召回时只做受控调制、
    绝不重新生成——同一件事两次回忆给出两种感受 = 自我打脸。
    - feeling：第一人称纹理短语（"脸一直发烫，想找个地缝钻进去"），不是标签；
    - appraisal：她的归因（她为什么这么感觉）；
    - valence/arousal：效价 -1..1 / 唤醒度 0..1，给召回衰减与排序算的标量；
    - peak_moment：感受峰值时刻。⚠ **它现在什么都不做**——设计意图是当衰减锚点
      （空则退回 timestamp），但 `upsert_memory` 落 Chroma 的 metadata 里没有这个键，
      召回侧的衰减只读 `meta["timestamp"]`，所以"峰值时刻影响遗忘速度"这套机制
      **不存在**。值只在内存对象里活着（以及 J12 的 vector_memory 重试 payload 里）。
      要么补写 metadata + 补用（会改变记忆衰减行为），要么删字段 + 删这句——
      两条都是作者的取舍，别由 agent 替作者定（见 计划与设计.md 待定项）。
      在拍板之前，这里不许再把它写成已经生效的机制。
    全部可缺省：旧记忆/感受生成失败时照存，只是没有纹理。
    """

    memory_id: str
    session_id: str
    kind: str  # hard_fact / preference / event / daily_summary / user_flaw
    content: str
    importance: int = 3  # 1~5，越大越重要
    timestamp: str = ""
    feeling: str = ""
    appraisal: str = ""
    valence: float = 0.0
    arousal: float = 0.3
    peak_moment: str = ""


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
