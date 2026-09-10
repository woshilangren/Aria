"""全局单例注册表：LLM、语音、工具表这些"全程只有一份"的东西都放这。

按名字存取，用之前先 init_all() 初始化一遍。哪个服务初始化失败不影响
其他服务起来，等真正用到它的时候再报错。
"""


class ServiceRegistry:
    """名字 -> 服务实例的注册表。"""

    def __init__(self):
        self._services: dict = {}
        self._errors: dict = {}

    def register(self, name: str, obj) -> None:
        self._services[name] = obj

    def get(self, name: str):
        if name not in self._services:
            reason = self._errors.get(name, "服务还没初始化")
            raise RuntimeError(f"服务 {name} 不可用: {reason}")
        return self._services[name]

    def check_status(self) -> dict:
        """报一下每个服务活没活着，给健康检查用。注册了算活着，初始化出过的错算挂了。"""
        status = {}
        for name in set(self._services) | set(self._errors):
            status[name] = name in self._services
        return status

    def init_all(self) -> None:
        """启动时把所有服务建一遍。单个失败只记账，不让整个程序挂掉。"""
        # 工具层基础件
        try:
            from tools.misc import Logger

            self.register("logger", Logger())
        except Exception as exc:
            self._errors["logger"] = str(exc)

        try:
            from tools.storage import KVStoreTool, VectorStoreTool

            self.register("kv_store", KVStoreTool())
            self.register("vector_store", VectorStoreTool())
        except Exception as exc:
            self._errors["kv_store"] = str(exc)

        try:
            from tools.registry import ToolExecutor, ToolRegistry

            self.register("tool_registry", ToolRegistry())
            self.register("tool_executor", ToolExecutor())
        except Exception as exc:
            self._errors["tool_registry"] = str(exc)

        # 日记作者：真身在 capability 层，工具名片已在 registry 登记，
        # 这里把实例注册好、把工具入口注入进去（延迟 import 避开循环依赖）
        try:
            from capability.diary import DiaryWriter

            self.register("diary_writer", DiaryWriter())
            self.get("tool_registry").register_handler(
                "diary_write", self.get("diary_writer").run_as_tool
            )
        except Exception as exc:
            self._errors["diary_writer"] = str(exc)

        # LLM：聊天功能的核心，失败要留清楚原因
        try:
            from tools.llm_client import LLMClient

            self.register("llm", LLMClient())
        except Exception as exc:
            self._errors["llm"] = str(exc)

        # 语音两件套，没配 key 也正常，用到再报错
        try:
            from tools.speech import ASRTool

            self.register("asr", ASRTool())
        except Exception as exc:
            self._errors["asr"] = str(exc)

        try:
            from tools.speech import TTSTool

            self.register("tts", TTSTool())
        except Exception as exc:
            self._errors["tts"] = str(exc)


# 全项目就这一份注册表
services = ServiceRegistry()


def get_llm():
    return services.get("llm")
