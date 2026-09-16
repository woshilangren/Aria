"""三处修复的复验（短）：初始亲密度 / 闲聊天气不触发工具 / 心事无AI字眼。"""

import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from shared.types import InputMessage  # noqa: E402

TURNS = [
    ("初见", "嗨，第一次跟你聊天"),
    ("闲聊天气（不得触发工具）", "今天天气挺好的，你在的那边呢"),
    ("抱怨她（收敛+心事转述）", "你怎么说话这么冲啊"),
]


async def main():
    from bootstrap import bootstrap

    bootstrap()
    from orchestration.pipeline import DialoguePipeline
    from shared.singletons import services

    kv = services.get("kv_store")
    pipeline = DialoguePipeline()
    for i, (label, text) in enumerate(TURNS, 1):
        reply = await pipeline.handle(InputMessage(text=text, session_id="default"))
        rel = kv.read("relationship", "default") or {}
        self_rec = kv.read("self", "default") or {}
        print(f"\n=== 第{i}轮 · {label} ===")
        print(f"[他] {text}")
        print(f"[她] {reply.text}")
        print(f"[状态] 名字={self_rec.get('name')} 亲密={rel.get('intimacy')} "
              f"心情={rel.get('mood')} 心事={rel.get('concern', {}).get('text', '')} "
              f"收敛={rel.get('calm', 0):.0%}")


if __name__ == "__main__":
    asyncio.run(main())
