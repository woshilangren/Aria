"""调度层 - DialogueOrchestrator（对话编排器）

用 LangGraph 把 graph.py 里的节点连成一张状态图，
单轮对话从安全预检进来，到记忆写回出去，中间按意图和情绪分流。
"""

import asyncio

from langgraph.graph import END, StateGraph

from shared.types import FinalReply, InputMessage

from orchestration.graph import (
    DialogueState,
    KEEPER,
    _MAX_REWRITE,
    compose_node,
    generate_node,
    perceive_node,
    refuse_node,
    respond_node,
    rewrite_node,
    safety_precheck_node,
    safety_review_node,
    toolcall_node,
    writeback_node,
)
from orchestration.managers import FallbackController


class DialogueOrchestrator:
    """单轮对话主流程编排器。"""

    def __init__(self):
        self.graph = None
        self.build_graph()

    def build_graph(self) -> None:
        """构建 LangGraph 主流程图。

        主链：安全预检 → 感知（含并行记忆召回） → 人设组装 → 生成初稿 → 后置审核 → 记忆写回
        分流：意图是查天气/搜资料/画图 → 工具管线 → 转述
        例外：预检不过 → 婉拒；情绪危机 → 走安抚语气；审核不过 → 重写（最多 2 次）
        """
        builder = StateGraph(DialogueState)

        builder.add_node("safety", safety_precheck_node)
        builder.add_node("perceive", perceive_node)
        builder.add_node("toolcall", toolcall_node)
        builder.add_node("respond", respond_node)
        builder.add_node("compose", compose_node)
        builder.add_node("generate", generate_node)
        builder.add_node("review", safety_review_node)
        builder.add_node("rewrite", rewrite_node)
        builder.add_node("refuse", refuse_node)
        builder.add_node("writeback", writeback_node)

        builder.set_entry_point("safety")

        # 预检不过直接婉拒，后面的一切都不用跑了
        builder.add_conditional_edges(
            "safety",
            lambda s: "refuse" if not s.get("safety_passed", True) else "perceive",
        )

        # 感知节点里已并行做完记忆召回（state["memory"] 恒有值），这里直接分流：
        # 工具类意图去工具管线，其余进人设对话链
        builder.add_conditional_edges(
            "perceive",
            lambda s: "toolcall"
            if s["intent"].intent in ("weather", "search", "image", "diary")
            else "compose",
        )

        # 工具管线跑完转述；要是中途直接写了追问话术，跳过转述去写回
        builder.add_conditional_edges(
            "toolcall",
            lambda s: "writeback" if s.get("final_reply") else "respond",
        )

        # 转述完同样要过审
        builder.add_edge("respond", "review")

        # 人设对话链
        builder.add_edge("compose", "generate")
        builder.add_edge("generate", "review")

        # 审核三岔口：过审去写回；没过且还有重写额度就重写；超限就婉拒收场
        builder.add_conditional_edges(
            "review",
            lambda s: (
                "writeback"
                if s.get("review_passed", False)
                else ("rewrite" if s.get("rewrite_count", 0) < _MAX_REWRITE else "refuse")
            ),
        )
        builder.add_edge("rewrite", "review")

        # 收尾两兄弟
        builder.add_edge("writeback", END)
        builder.add_edge("refuse", END)

        self.graph = builder.compile()

    async def handle(self, message: InputMessage) -> FinalReply:
        """处理一条用户消息，返回最终回复（主入口）。

        重启后第一次聊天先把短期记忆从磁盘捞回来，再跑图。
        全图任何一环炸了都降级成兜底话，不让用户看到报错。
        """
        KEEPER.restore(message.session_id)

        state: DialogueState = {
            "user_text": message.text,
            "session_id": message.session_id,
            "voice_mode": message.input_mode == "voice",
            "user_image": message.image_url or "",
            "comfort_mode": False,
            "safety_passed": True,
            "rewrite_count": 0,
            "tool_results": [],
        }

        try:
            # LangGraph 的 invoke 是同步的，扔线程池里跑，别卡住事件循环
            result = await asyncio.to_thread(self.graph.invoke, state)
        except Exception:
            controller = FallbackController()
            return FinalReply(text=controller.fallback_reply("llm"))

        return FinalReply(
            text=result.get("final_reply", ""),
            output_mode=result.get("output_mode", "text"),
            image_path=result.get("image_path", ""),
        )
