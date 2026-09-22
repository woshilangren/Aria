"""全局单例注册表：LLM、语音、工具表这些"全程只有一份"的东西都放这。

按名字存取。**这里只存不建**——装配归 bootstrap.py（组合根），什么时候装配
由应用启动时机决定（见 main.py 的 lifespan）。
本模块刻意不 import 任何上层实现，避免 shared 反向依赖 tools / capability。
"""


class ServiceRegistry:
    """名字 -> 服务实例的注册表。"""

    def __init__(self):
        self._services: dict = {}
        self._errors: dict = {}

    def register(self, name: str, obj) -> None:
        # R06b：成功注册即清掉同名的旧错误——"成功恢复要清掉旧错误"，
        # 否则一次失败的重挂着永久留在 _errors 里污染 get() 的报错理由
        self._errors.pop(name, None)
        self._services[name] = obj

    def fail(self, name: str, reason: str) -> None:
        """注册后初始化失败（R06b）：撤出注册表并记账。

        "注册对象"不等于"初始化成功"——先 register 再 ensure_ready 这类
        两段式服务，第二段挂了如果只 mark_error、实例还留在表里，
        health 就会假绿（服务看着活着，一用就炸）。
        """
        self._errors[name] = reason
        self._services.pop(name, None)

    def get(self, name: str):
        if name not in self._services:
            reason = self._errors.get(name, "服务还没初始化")
            raise RuntimeError(f"服务 {name} 不可用: {reason}")
        return self._services[name]

    def mark_error(self, name: str, reason: str) -> None:
        """记一笔某个服务初始化失败的原因，get() 时据此给出更清楚的理由。"""
        self._errors[name] = reason

    def check_status(self) -> dict:
        """报一下每个服务活没活着，给健康检查用。注册了算活着，初始化出过的错算挂了。"""
        status = {}
        for name in set(self._services) | set(self._errors):
            status[name] = name in self._services
        return status


# 全项目就这一份注册表
services = ServiceRegistry()


def get_llm():
    return services.get("llm")
