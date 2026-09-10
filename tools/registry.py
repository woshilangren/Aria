"""工具注册表和执行器。

注册表管"有哪些工具、参数长什么样"（给模型看的名片），
执行器管"按名字把工具真正跑起来"，带重试和日志。
"""

from config.settings import load_app_config
from shared.singletons import services
from tools.external import ExternalSearchTool, ImageGenTool, WeatherTool
from tools.misc import ClockTool, Logger


class ToolRegistry:
    """所有可调用的工具都先来这登记，模型看到的工具列表也从这出。

    工具分两种：内置的（实现就在 tools 层）和注入的（真身在 capability 层，
    启动时塞进来）。注册表只管"名片"和"调度"，不关心实现是谁写的。
    """

    def __init__(self):
        self._tools = {}
        self._handlers = {}  # 注入式工具的真身：capability 层提供的实现
        self._logger = Logger()
        # 内置工具，启动时一次登记完
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register(
            name="weather_query",
            description="查询某个城市当前的天气情况",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，比如：上海"}
                },
                "required": ["city"],
            },
        )
        self.register(
            name="web_search",
            description="联网搜索最新的资料或新闻",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "description": "要几条结果，默认 3"},
                },
                "required": ["query"],
            },
        )
        self.register(
            name="image_gen",
            description="根据文字描述生成一张图片",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "画面内容的描述"}
                },
                "required": ["prompt"],
            },
        )
        self.register(
            name="clock_now",
            description="获取当前的日期和时间",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        # 写日记不需要模型填任何参数：哪天写、写给谁，运行时都知道
        self.register(
            name="diary_write",
            description="把今天的对话整理成一篇日记保存起来（用户说写日记/记日记时调用）",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def register(self, name: str, description: str, parameters: dict) -> None:
        self._tools[name] = {"name": name, "description": description, "parameters": parameters}

    def register_handler(self, name: str, handler) -> None:
        """给工具挂上真正的实现。

        实现由 capability 层提供后注入进来，tools 层不用反向 import capability，
        依赖方向依旧是 capability → tools，分层不破。
        """
        self._handlers[name] = handler

    def get_handler(self, name: str):
        return self._handlers.get(name)

    def get_meta(self, name: str) -> dict:
        return self._tools.get(name)

    def catalog(self) -> list:
        """把工具表整理成 OpenAI 接口要的格式，直接塞给模型。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in self._tools.values()
        ]


class ToolExecutor:
    """按名字执行工具。失败重试几次，每次都记日志，别让上层摸黑。"""

    def __init__(self):
        self._weather = WeatherTool()
        self._search = ExternalSearchTool()
        self._image = ImageGenTool()
        self._clock = ClockTool()
        self._logger = Logger()

    def execute(self, tool_name: str, arguments: dict, session_id: str = "") -> dict:
        """跑一个工具，返回 {"status": ok/error, "data": ...}。

        session_id 是运行时上下文（当前是哪个会话），只给注入式工具用，
        模型不需要填也不该让它填，由调用链一路传下来。
        """
        cfg = load_app_config()["tools"]
        max_retry = cfg["max_retry"]

        # 注入式工具优先，没有注入的真身再找内置实现
        injected = self._injected_handler(tool_name)
        handler = injected if injected is not None else self._resolve(tool_name)
        if handler is None:
            return {"status": "error", "data": f"没有这个工具：{tool_name}"}

        last_error = None
        for attempt in range(max_retry):
            try:
                if injected is not None:
                    # 注入式约定：session_id 显式传入，模型瞎给的其他参数原样透传
                    kwargs = dict(arguments or {})
                    kwargs["session_id"] = session_id
                    data = injected(**kwargs)
                else:
                    data = handler(**(arguments or {}))
                self._logger.log_tool_call(tool_name, arguments, "ok")
                return {"status": "ok", "data": data}
            except Exception as exc:
                last_error = exc

        self._logger.log_tool_call(tool_name, arguments, "error", str(last_error))
        return {"status": "error", "data": f"{tool_name} 执行失败: {last_error}"}

    def _injected_handler(self, tool_name: str):
        """从注册表拿注入式工具的真身；注册表不在（比如单测环境）就当没有。"""
        try:
            return services.get("tool_registry").get_handler(tool_name)
        except Exception:
            return None

    def _resolve(self, tool_name: str):
        if tool_name == "weather_query":
            return self._weather.query
        if tool_name == "web_search":
            return self._search.search
        if tool_name == "image_gen":
            return self._image.generate
        if tool_name == "clock_now":
            return self._clock.now
        return None
