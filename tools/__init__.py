"""工具层（Tools）

职责：纯技术封装 —— 大模型、语音、外部 API、存储访问、时钟、日志。
不含任何业务含义；只调用数据层与外部服务，禁止调用上层。
ASR / LLM / TTS 三件套为全局单例，经 shared/singletons.ServiceRegistry 统一取用。
"""
