"""装配完整性冒烟（批次 B2）：文档承诺存在的公开方法必须真的在类上。

存在的理由：`_log1p_safe` 缩进事故（A1）产出的是语法完全合法的 Python——
import 照常成功、7 个类方法被吞成模块级函数 return 之后的死代码、
ruff/compileall 都抓不到，134 个测试照样全绿。只有 hasattr 一测就红。
这一层与 ruff F821 互补：ruff 抓"用了没定义的名字"，这里抓"承诺过的方法消失"。
"""

import pytest

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
            ],
        ),
        (DialoguePipeline, ["handle", "astream"]),
    ],
)
def test_public_surface(cls, names):
    missing = [n for n in names if not callable(getattr(cls, n, None))]
    assert not missing, f"{cls.__name__} 缺 {missing}"
