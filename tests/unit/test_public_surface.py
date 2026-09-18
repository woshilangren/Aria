"""装配完整性冒烟（批次 B2 + B7）：文档承诺的公开面必须真的在、而且真的干活。

两种断言，抓两类不同的事故：

1. `test_public_surface`（B2）——**方法还在不在**。
   `_log1p_safe` 缩进事故（A1）产出的是语法完全合法的 Python：import 照常成功、
   7 个类方法被吞成模块级函数 return 之后的死代码，ruff/compileall 都抓不到，
   134 个测试照样全绿。只有 hasattr 一测就红。
2. `test_store_facades_return_usable_containers`（B7）——**方法还干不干活**。
   附七 P0-1：批次 K7 往 `data/stores.py` 插 `chat_log_hour_distribution` 时，把
   `last_chat_per_session` 的 `return get_db().last_chat_per_session()` 一行孤儿化
   留在新方法里，于是 `last_chat_per_session` 方法体空了 → 返回 None → 调用方
   `proactive.py:263` 拿到 None 后迭代直接 TypeError → **闲置日记巡检整体停摆**。
   hasattr 抓不到（方法还在），只有"真调一次看返回类型"能抓。

两条与 ruff F821 互补：ruff 抓"用了没定义的名字"（A2/A3），这两条抓"承诺过的方法
消失或空转"（A1/P0-1）。**同一类病：往类里插代码把邻居弄坏。**
"""

import pytest

from data.stores import SessionStore
from orchestration.pipeline import DialoguePipeline
from tools.storage import KVStoreTool, VectorStoreTool


@pytest.mark.parametrize(
    "cls,names",
    [
        (
            VectorStoreTool,
            [
                "ensure_ready",
                "upsert_memory",
                "search_memory",
                "upsert_diary",
                "search_diary",
                "list_diaries",
                "delete_diary",
                "list_diaries_enriched",
                "list_memories",
                "delete_memory",
            ],
        ),
        (
            KVStoreTool,
            [
                "read",
                "write",
                "update",
                "log",
                "log_relationship_change",
                "recent_ledger",
                "chats_between",
                "last_chat_per_session",
                "proactive_after",
                "recent_chat",
                # J9/E1 的修剪口子。prune_candidates 尤其要盯着：巡检侧是用
                # getattr 探测它的，探不到就**静默跳过**——改名或删掉不会让任何
                # 测试变红，只会让候选池从此无界增长。这正是本护栏存在的理由。
                "prune_chat_log",
                "prune_ledger",
                "prune_candidates",
                "prune_uploads",
                "prune_image_assets",
            ],
        ),
        (DialoguePipeline, ["handle", "astream"]),
        # data 层 store：KVStoreTool 的门面方法全部委托到这里。
        # 单测 tools 层查不出"门面在、委托断了"——proactive_after 就是这么坏的。
        (
            SessionStore,
            [
                "append",
                "get_recent",
                "get_chats_between",
                "last_chat_per_session",
                "chat_log_hour_distribution",
                "proactive_after",
            ],
        ),
    ],
)
def test_public_surface(cls, names):
    missing = [n for n in names if not callable(getattr(cls, n, None))]
    assert not missing, f"{cls.__name__} 缺 {missing}"


def test_store_facades_return_usable_containers(tmp_path, monkeypatch):
    """存储门面的读方法在空库上必须返回可用的空容器，绝不能返回 None。

    调用方（`proactive.py` 的闲置巡检、`api.py` 的记忆/日记端点）拿到结果就直接
    迭代或取长度，返回 None 会当场 TypeError——而被兜底 except 吞掉之后，
    表现就是"某个子系统静默停摆"，和 A1 一样查不出来。
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from config.settings import get_settings, load_app_config

    get_settings.cache_clear()
    load_app_config.cache_clear()

    import data.sqlite_store as sqlite_mod

    sqlite_mod._db = None  # 指向本用例的 tmp_path，别复用上一个用例的库

    sid = "b7-nonexistent"

    # ---- KVStoreTool 门面（tools 层）----
    kv = KVStoreTool()
    for name, got in [
        ("recent_ledger", kv.recent_ledger(sid)),
        ("chats_between", kv.chats_between(sid, "2020-01-01T00:00:00", "2020-01-02T00:00:00")),
        ("last_chat_per_session", kv.last_chat_per_session()),
        ("proactive_after", kv.proactive_after(sid, "2020-01-01T00:00:00")),
        ("recent_chat", kv.recent_chat(sid)),
        ("read(profile)", kv.read("profile", sid)),
        ("read(relationship)", kv.read("relationship", sid)),
    ]:
        assert got is not None, f"KVStoreTool.{name} 返回了 None（方法体可能被邻居改动掏空）"
        assert isinstance(got, (list, dict)), f"KVStoreTool.{name} 应返回 list/dict，实测 {type(got)}"

    assert isinstance(kv.chat_log_hour_distribution(sid), dict), "小时分布应返回 dict"

    # ---- data.stores.SessionStore（附七 P0-1 的出事现场，直接测这一层）----
    store = SessionStore()
    assert store.last_chat_per_session() == [], "P0-1 的原始症状：这里曾经返回 None"
    assert store.get_recent(sid) == []
    assert store.get_chats_between(sid, "2020-01-01T00:00:00", "2020-01-02T00:00:00") == []
    assert isinstance(store.chat_log_hour_distribution(sid), dict)
    assert store.proactive_after(sid, "2020-01-01T00:00:00") == []

    # ---- VectorStoreTool：不调 ensure_ready()，全部走"集合没建出来"的降级路径 ----
    # 降级路径正是最常跑的那条（embedding key 没配 / Chroma 挂了），
    # 也是最容易被写坏后无人察觉的那条——静默返回空，看起来像"她还没有记忆"。
    from data.schemas import MemoryItem

    vs = VectorStoreTool()
    for name, got in [
        ("list_diaries", vs.list_diaries()),
        ("list_diaries_enriched", vs.list_diaries_enriched()),
        ("list_memories", vs.list_memories()),
        ("search_memory", vs.search_memory("随便搜点什么")),
        ("search_diary", vs.search_diary("随便搜点什么")),
    ]:
        assert got == [], f"VectorStoreTool.{name} 降级时应返回 []，实测 {got!r}"

    item = MemoryItem(memory_id="b7-1", session_id=sid, kind="event", content="内容")
    for name, got in [
        ("upsert_memory", vs.upsert_memory(item)),
        ("upsert_diary", vs.upsert_diary("diary-2020-01-01", "内容", {})),
        ("delete_diary", vs.delete_diary("diary-2020-01-01")),
        ("delete_memory", vs.delete_memory("some-id")),
    ]:
        assert got is False, f"VectorStoreTool.{name} 降级时应返回 False，实测 {got!r}"
