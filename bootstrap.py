"""组合根（Composition Root）。

这是全项目**唯一**允许"跨层向上 import"的地方——装配的职责本来就是
"知道所有层的存在"：它要 import tools / capability 里的具体实现，把实例
建好、注册进全局注册表。除此之外的任何模块都不许这么干。

把装配单独拎到这里之后：
- shared/singletons.py 退化成纯注册表，不再反向依赖 tools / capability，
  那个"懒加载环 + 分层倒置"就消掉了；
- 装配有了显式入口（`bootstrap()`），什么时候装配一目了然，顺手解决了
  "import 就产生副作用"——现在装配只发生在应用 startup（见 main.py 的 lifespan）。
"""

from shared.singletons import services


def bootstrap() -> None:
    """按依赖顺序装配所有服务，逐个注册进全局注册表。

    单个服务初始化失败只记账、不抛出：某个服务没起来不影响别的服务，
    真正用到它的时候再报错（保持既有容错语义）。
    """
    # 工具层基础件
    try:
        from tools.misc import Logger

        services.register("logger", Logger())
    except Exception as exc:
        services.mark_error("logger", str(exc))

    # kv_store 和 vector_store 各走各的 try：以前两个共用同一个 try，
    # vector_store 初始化一失败就把 kv_store 一起带走，等于全部持久化失效。
    try:
        from tools.storage import KVStoreTool

        services.register("kv_store", KVStoreTool())
    except Exception as exc:
        services.mark_error("kv_store", str(exc))

    try:
        from tools.storage import VectorStoreTool

        vector_store = VectorStoreTool()
        services.register("vector_store", vector_store)
        # 建集合（含"换模型 -> 删库重建"的维度守护）显式做一次。
        # 构造函数里只建 client/embedder，免得任何人 new 一下就悄悄触发重建。
        # 时机与改造前一致（都在 lifespan startup），所以不是行为变化。
        vector_store.ensure_ready()
    except Exception as exc:
        services.mark_error("vector_store", str(exc))

    # 拆成两个 try：以前两个共用同一个 try，registry 初始化一失败就把 executor 一起带走，
    # 等于工具表整体成功但 executor 不在工作位（key 是 `tool_registry` 拿不到真实原因）。
    # 这次照搬 kv_store / vector_store 的写法。
    try:
        from tools.registry import ToolRegistry

        services.register("tool_registry", ToolRegistry())
    except Exception as exc:
        services.mark_error("tool_registry", str(exc))

    try:
        from tools.registry import ToolExecutor

        services.register("tool_executor", ToolExecutor())
    except Exception as exc:
        services.mark_error("tool_executor", str(exc))

    # 日记作者：真身在 capability 层，工具名片已在 registry 登记，
    # 这里把实例注册好、把工具入口注入进去（延迟 import 避开循环依赖）
    try:
        from capability.diary import DiaryWriter

        services.register("diary_writer", DiaryWriter())
        services.get("tool_registry").register_handler(
            "diary_write", services.get("diary_writer").run_as_tool
        )
    except Exception as exc:
        services.mark_error("diary_writer", str(exc))

    # LLM：聊天功能的核心，失败要留清楚原因
    try:
        from tools.llm_client import LLMClient

        services.register("llm", LLMClient())
    except Exception as exc:
        services.mark_error("llm", str(exc))

    # 语音两件套，没配 key 也正常，用到再报错
    try:
        from tools.speech import ASRTool

        services.register("asr", ASRTool())
    except Exception as exc:
        services.mark_error("asr", str(exc))

    try:
        from tools.speech import TTSTool

        services.register("tts", TTSTool())
    except Exception as exc:
        services.mark_error("tts", str(exc))

    # 人设引擎：真身是 orchestration/pipeline.py 的模块级 _ENGINE（建在全项目唯一
    # 那份 KEEPER 上）。只有组合根允许向上 import 调度层，所以这一行只能在这里。
    # 注册上以后，工具链路的 response_generator._persona_engine() 就不用再走
    # "临时 new 一份只读 keeper"的退化分支——那条路的短期上下文取自 session 表，
    # 比内存里的 KEEPER 最多旧一轮，而且每轮多三次 KV 读（C1）。
    try:
        from orchestration.pipeline import _ENGINE

        services.register("persona_engine", _ENGINE)
    except Exception as exc:
        services.mark_error("persona_engine", str(exc))
