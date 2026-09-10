"""向量化客户端：把文字变成一串数字（向量），Chroma 靠它做语义检索。

qwen3-vl-embedding 不支持 OpenAI 兼容接口，只能走 dashscope 的
MultiModalEmbedding，所以单独建这个文件，不塞进 llm_client。
"""

from typing import List

import dashscope

from config.settings import get_settings

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
        rsp = dashscope.MultiModalEmbedding.call(
            model=self._model,
            input=[{"text": t} for t in texts],
            api_key=self._api_key,
            dimension=self._dimension,
        )
        if rsp.status_code != 200:
            raise RuntimeError(f"向量化请求失败：{rsp.code} {rsp.message}")
        # 返回的 embeddings 数组顺序和请求一致，按位置取向量就行
        return [item["embedding"] for item in rsp.output["embeddings"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入：长列表按批切开，一段段喂给接口再拼回去。"""
        vectors: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors.extend(self._embed_once(texts[i : i + EMBED_BATCH_SIZE]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """单条嵌入，搜记忆前先把查询句变成向量。"""
        return self._embed_once([text])[0]
