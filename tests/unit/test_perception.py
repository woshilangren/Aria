"""感知模块单测：`_load_subtext_hints` 词典命中 + 模型掉线时的词表兜底。

不联网：需要 LLM 时用"一律抛错"的替身逼出兜底路径；危机词路径根本不碰模型。
"""

import types

import pytest

from capability.perception import PerceptionPipeline, _load_subtext_hints


class _FailingLLM:
    """chat / chat_with_tools / astream_chat 全抛错，逼感知走词表兜底。"""

    def chat(self, *args, **kwargs):
        raise RuntimeError("test LLM offline")

    def chat_with_tools(self, *args, **kwargs):
        raise RuntimeError("test LLM offline")

    async def astream_chat(self, *args, **kwargs):
        raise RuntimeError("test LLM offline")
        yield ""  # pragma: no cover - 让它是异步生成器


class _FakeKV:
    """最小 kv_store 替身：read() 返回预置的 persona 对象。"""

    def __init__(self, persona=None, boom=False):
        self._persona = persona
        self._boom = boom

    def read(self, store, key, default=None):
        if self._boom:
            raise RuntimeError("kv down")
        return self._persona

    def write(self, store, key, value):
        return True


def _persona(hints):
    """构造一个带 subtext_hints 的假人设对象。"""
    return types.SimpleNamespace(subtext_hints=hints)


@pytest.fixture
def install_kv(register_services):
    def _install(persona=None, boom=False):
        register_services(kv_store=_FakeKV(persona, boom))

    return _install


@pytest.fixture
def offline_pipeline(register_services):
    """注册"离线"替身，返回一个会走词表兜底的感知器。"""
    register_services(llm=_FailingLLM(), kv_store=_FakeKV(_persona({})))
    return PerceptionPipeline()


# ----------------------- _load_subtext_hints -----------------------
def test_subtext_hints_hit(install_kv):
    install_kv(persona=_persona({"我没事": "其实很难过", "随便": "希望你替我做决定"}))
    out = _load_subtext_hints("我没事")
    assert "「我没事」" in out
    assert "其实很难过" in out
    # 没命中的那一条不许混进来
    assert "随便" not in out


def test_subtext_hints_no_match(install_kv):
    install_kv(persona=_persona({"我没事": "其实很难过"}))
    assert _load_subtext_hints("今天天气不错") == ""


def test_subtext_hints_persona_missing(install_kv):
    install_kv(persona=None)
    assert _load_subtext_hints("我没事") == ""


def test_subtext_hints_kv_error(install_kv):
    install_kv(boom=True)
    assert _load_subtext_hints("我没事") == ""


# ------------------------- 词表兜底路径 -----------------------------
def test_fallback_weather_intent(offline_pipeline):
    intent, emotion, subtext, _extras = offline_pipeline.run("今天天气怎么样")
    assert intent.intent == "weather"
    assert intent.confidence == 0.6
    assert emotion.emotion == "neutral"
    assert subtext == ""


def test_fallback_comfort_intent_and_sad_emotion(offline_pipeline):
    intent, emotion, _subtext, _extras = offline_pipeline.run("我好难过啊")
    assert intent.intent == "comfort"
    assert emotion.emotion == "sad"


def test_fallback_image_intent(offline_pipeline):
    intent, _emotion, _subtext, _extras = offline_pipeline.run("画一张小猫")
    assert intent.intent == "image"


def test_crisis_takes_priority_without_model(offline_pipeline):
    intent, emotion, subtext, _extras = offline_pipeline.run("我真的不想活了")
    assert intent.intent == "comfort"
    assert emotion.is_crisis is True
    assert emotion.intensity == 1.0
    assert subtext == ""


def test_empty_text_returns_default(offline_pipeline):
    intent, emotion, subtext, _extras = offline_pipeline.run("   ")
    assert intent.intent == "chat"
    assert emotion.emotion == "neutral"
    assert subtext == ""


# --------------------- SafetyReviewer 出戏话（C4）---------------------
@pytest.mark.parametrize("text", [
    "工具这边什么都没查到",      # 真机 turn 7 原句，这条模式就是为它加的
    "工具没有返回结果",
    "接口未响应",
    "系统查不到数据",
    "我调用了工具去查",
    "调用接口失败了",
    "搜索失败了，换个说法",
    "查询出错了",
    "作为一个AI我不能这么想",    # 原有字面量不许因为加了正则就失效
    "我是程序，没有感觉",
    "我是模型",
])
def test_review_blocks_system_voice(text):
    from capability.perception import SafetyReviewer

    ok, reason = SafetyReviewer().review(text, "output")
    assert ok is False
    assert "出戏" in reason


@pytest.mark.parametrize("text", [
    "他就是个工具人",
    "我翻翻工具箱",
    "工具人没什么用",        # 把正则间距放宽去够 turn 7 那句，就会开始误杀这一句
    "工具箱没带上",
    "这系统没什么毛病",
    "我没查到你说的那本书",   # 主语是"我"不是工具，是她自己在说话
    "这个工具挺好用的",
    "帮你查一下天气",
    "今天天气不错",
])
def test_review_does_not_block_legitimate_wording(text):
    """误杀的代价是**整轮降级**（_ReviewReject → refuse），远高于漏杀，所以宁可窄。"""
    from capability.perception import SafetyReviewer

    ok, reason = SafetyReviewer().review(text, "output")
    assert ok is True, f"误杀了正常话：{reason}"


def test_review_meta_check_only_on_output():
    """input 模式只查违禁词，不查出戏话——出戏是**她**说的问题，不是他说的。"""
    from capability.perception import SafetyReviewer

    ok, _ = SafetyReviewer().review("工具这边什么都没查到", "input")
    assert ok is True
