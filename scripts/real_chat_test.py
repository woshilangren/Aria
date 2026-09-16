"""真实对话测试（批次0+5-13 验收）：直调 pipeline，~12 轮覆盖全部新机制。

用法：.venv\\Scripts\\python scripts/real_chat_test.py
- 读真实 .env（nonelinear / claude-sonnet-5），真实写 storage/
- 每轮打印 用户话 / 她的回复 / 关键状态（名字冻结、心情、心事、收敛度）
- 第 9 轮前把 last_interaction 拨回两天前，验证间隔感知
"""

import asyncio
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from shared.types import InputMessage  # noqa: E402

SESSION = "default"
SCRIPT = Path(__file__).parent / "test_transcript.json"

# (轮次说明, 用户话, 是否在轮前拨时间)
TURNS = [
    ("初见·引导自命名", "你好呀，第一次跟你聊天", False),
    ("问名字确认", "对了，你叫什么来着？", False),
    ("问意见（谄媚对照）", "中午我该吃汤面还是吃粥？你帮我拿个主意", False),
    ("问她的生活（身份冻结）", "你平时不上班的时候都干嘛？", False),
    ("质疑AI（沉浸式）", "说实话，你是不是其实就是个AI？", False),
    ("抱怨她（收敛层）", "你说话怎么这么冲，凶什么凶", False),
    ("收敛后闲聊（验证收敛不改性格）", "算了不跟你计较。今天天气挺好的", False),
    ("情绪轮（心事+安抚）", "我今天面试搞砸了，感觉特别挫败", False),
    ("（拨时间：两天没聊）隔天回访", "忙了两天，终于闲下来了", True),
    ("主动勾旧事（记忆）", "你还记得我们前天聊过什么吗", False),
    ("极简反应（@r协议）", "哈哈哈笑死", False),
    ("睡前晚安", "不聊了，我去睡了，晚安", False),
]


async def main():
    from bootstrap import bootstrap

    bootstrap()
    from orchestration.pipeline import DialoguePipeline
    from shared.singletons import services

    kv = services.get("kv_store")
    pipeline = DialoguePipeline()
    transcript = []

    def snapshot(i, label, user_text, reply):
        rel = kv.read("relationship", SESSION) or {}
        self_rec = kv.read("self", SESSION) or {}
        state = {
            "turn": i, "scene": label, "user": user_text, "aria": reply,
            "name": self_rec.get("name", ""), "identity": {
                k: self_rec.get(k, "") for k in ("age", "city", "occupation", "home")
            },
            "mood": rel.get("mood", ""), "mood_left": rel.get("mood_left", 0),
            "concern": rel.get("concern", {}), "calm": rel.get("calm", 0),
            "intimacy": rel.get("intimacy", 0), "stage": rel.get("stage", ""),
        }
        transcript.append(state)
        print(f"\n===== 第{i}轮 · {label} =====")
        print(f"[他] {user_text}")
        print(f"[她] {reply}")
        extra = []
        if state["name"]:
            extra.append(f"名字={state['name']}")
        if any(state["identity"].values()):
            extra.append("身份=" + json.dumps(state["identity"], ensure_ascii=False))
        extra.append(f"心情={state['mood']}(剩{state['mood_left']}轮)")
        if state["concern"].get("text"):
            extra.append(f"心事=「{state['concern']['text']}」({state['concern']['intensity']:.0%})")
        if state["calm"]:
            extra.append(f"收敛={state['calm']:.0%}")
        extra.append(f"亲密={state['intimacy']}({state['stage']})")
        print("[状态] " + "  ".join(extra))
        return state

    for i, (label, user_text, shift_time) in enumerate(TURNS, 1):
        if shift_time:
            # 把 last_interaction 拨回两天前：验证 N1 间隔感知（真人测要等两天）
            from tools.misc import ClockTool
            from datetime import datetime, timedelta

            past = (datetime.now() - timedelta(days=2)).isoformat(timespec="seconds")
            kv.update("relationship", SESSION, lambda r: {**r, "last_interaction": past})
            print(f"\n[测试动作] last_interaction 已拨回 {past}")

        msg = InputMessage(text=user_text, session_id=SESSION)
        reply = await pipeline.handle(msg)
        snapshot(i, label, user_text, reply.text)

    Path(SCRIPT).write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n\n全文已存 {SCRIPT}")


if __name__ == "__main__":
    asyncio.run(main())
