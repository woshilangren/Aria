"""向量化客户端：把文字变成一串数字（向量），Chroma 靠它做语义检索。

qwen3-vl-embedding 不支持 OpenAI 兼容接口，只能走 dashscope 的
MultiModalEmbedding，所以单独建这个文件，不塞进 llm_client。

R27c 失败契约：dashscope 调用边界上把网络/SDK/错误码失败统一转成
ExternalServiceError（本模块的调用方——storage 层的 upsert/search——按
RuntimeError 兜底，窄载体是它的子类，零破坏）；返回体的结构坏数据按
外部协议错误处理（bad_protocol，不重试）。try 块里**只有 SDK 调用本身**，
本地装配缺陷（比如 texts 传了 None）不在转换范围内，原样上抛。
"""

from typing import List

import dashscope

from config.settings import get_settings
from shared.types import ExternalServiceError

# dashscope 向量接口一次能吃的文本条数有限，稳妥起见一批 10 条
EMBED_BATCH_SIZE = 10


class QwenEmbedding:
    """百炼向量化模型的客户端：批量嵌入、单条查询都用它。"""

    def __init__(self):
        cfg = get_settings()
        self._model = cfg.embedding_model
        self._api_key = cfg.embedding_api_key
        self._dimension = cfg.embedding_dimension

    def _embed_once(self, texts: List[str]) -> List[List[float]]:
        """一小批文本 -> 一组向量。"""
        # 本地装配在边界之外：payload 构造出错是编程缺陷，不伪装成外部故障
        payload = [{"text": t} for t in texts]
        try:
            # —— SDK 调用边界：只有这一行，里面抛的都是 dashscope 的事 ——
            rsp = dashscope.MultiModalEmbedding.call(
                model=self._model,
                input=payload,
                api_key=self._api_key,
                dimension=self._dimension,
            )
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise ExternalServiceError(
                "embedding", "request_failed", retryable=True, detail=str(exc)
            ) from exc
        # 业务错误码显式消费（R27c）：非 200 不是异常也得当失败处理
        if rsp.status_code != 200:
            raise ExternalServiceError(
                "embedding", "request_failed", retryable=True,
                detail=f"向量化请求失败：{rsp.code} {rsp.message}",
            )
        try:
            # 返回的 embeddings 数组顺序和请求一致，按位置取向量就行
            return [item["embedding"] for item in rsp.output["embeddings"]]
        except (KeyError, TypeError, IndexError) as exc:
            # 协议坏数据（R27b/R27c）：返回体结构不是约定的形状——外部协议
            # 错误，重试大概率还是坏，标记不可重试；不与本地缺陷混为一谈
            raise ExternalServiceError(
                "embedding", "bad_protocol", retryable=False,
                detail=f"向量化返回体结构异常：{exc!r}",
            ) from exc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入：长列表按批切开，一段段喂给接口再拼回去。"""
        vectors: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors.extend(self._embed_once(texts[i : i + EMBED_BATCH_SIZE]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """单条嵌入，搜记忆前先把查询句变成向量。"""
        return self._embed_once([text])[0]
