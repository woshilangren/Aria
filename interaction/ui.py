"""交互层 - Gradio 聊天界面

浏览器里聊天用，文字对话为主，带图的回复顺带展示图片；
语音通话线路在这里切换：两段接力 / 直达专线 / 实时专线。
挂在 /ui 路径下，装载的事归 api.py 管。
"""

import uuid

import gradio as gr

from interaction.gateway import MessageReceiver, OutputRenderer
from orchestration.dialogue_orchestrator import DialogueOrchestrator

# 语音通话的三条线路，名字对齐 wiki：cascade=两段接力，e2e=直达专线，realtime=实时专线
ROUTE_LABELS = {
    "cascade": "两段接力（转文字再回答）",
    "e2e": "直达专线（语音对话引擎）",
    "realtime": "实时专线（带记忆）",
}
LABEL_TO_ROUTE = {v: k for k, v in ROUTE_LABELS.items()}


def build_gradio_app() -> gr.Blocks:
    """搭一个聊天页面：对话窗口 + 输入框 + 语音线路切换。"""
    orchestrator = DialogueOrchestrator()
    receiver = MessageReceiver()
    renderer = OutputRenderer()

    async def respond(message: str, history: list, session_id: str):
        """点发送之后的事：收消息 -> 过主对话图 -> 渲染回复 -> 刷回页面。"""
        msg = receiver.receive(message, session_id)
        reply = await orchestrator.handle(msg)
        data = renderer.render(reply, session_id)
        text = data["text"]
        if data["output_mode"] == "image" and data["image_path"]:
            text += f"\n\n（图：{data['image_path']}）"
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": text},
        ]
        return "", history

    def switch_route(label: str) -> str:
        """切语音线路。语音通话固定用 default 会话号，必须写它才对电话生效。"""
        from orchestration.managers import VoiceRouteManager

        VoiceRouteManager().switch("default", LABEL_TO_ROUTE.get(label, "cascade"))
        return label

    def load_page():
        """开页面的活：发会话号 + 把当前语音线路亮出来。"""
        from orchestration.managers import VoiceRouteManager

        try:
            current = VoiceRouteManager().current_route("default")
        except Exception:
            current = "cascade"
        return uuid.uuid4().hex, ROUTE_LABELS.get(current, ROUTE_LABELS["cascade"])

    with gr.Blocks(title="Aria") as demo:
        session_id = gr.State("default")
        chatbot = gr.Chatbot(label="Aria", height=480)
        box = gr.Textbox(placeholder="说点什么吧……", show_label=False)
        route_choice = gr.Radio(
            choices=list(ROUTE_LABELS.values()),
            value=ROUTE_LABELS["cascade"],
            label="语音通话线路（只影响语音电话，打字聊天不受影响）",
        )

        # 每个新开的页面发一个独立会话号，几个窗口同时聊也不串
        demo.load(load_page, None, [session_id, route_choice])
        box.submit(respond, [box, chatbot, session_id], [box, chatbot])
        route_choice.change(switch_route, route_choice, route_choice)

    return demo
