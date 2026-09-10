"""Chroma 向量数据库的客户端，只负责一件事：把连接建好、把集合拿回来。

全项目共用一个客户端实例，别到处 new。
"""

import chromadb

from config.settings import get_settings


class ChromaClient:
    """包一层 chromadb，后面 VectorStoreTool 用它读写长期记忆。"""

    def __init__(self):
        self.client = None
        self.available = False
        try:
            # 数据落在本地磁盘，重启不丢；路径跟着 data_dir 走，别处都是这么取的
            self.client = chromadb.PersistentClient(path=str(get_settings().data_dir / "chroma"))
            self.available = True
        except Exception as exc:
            # 建不起来就先带着标记活着，上层会判断 available 再用
            print(f"[chroma] 初始化失败，长期记忆向量检索不可用: {exc}")

    def get_collection(self, collection_name: str, metadata: dict = None):
        """按名字拿集合，没有就顺手建一个。

        不把向量化函数挂到集合上（chroma 1.x 会校验挂载的函数，容易踩坑），
        向量由上层自己算好再传进来。连接不可用会抛异常，由调用方兜底。
        """
        if not self.available or self.client is None:
            raise RuntimeError("Chroma 不可用")
        return self.client.get_or_create_collection(collection_name, metadata=metadata)

    def delete_collection(self, collection_name: str) -> None:
        """整个删掉一个集合，维度守护重建集合时用。"""
        if not self.available or self.client is None:
            raise RuntimeError("Chroma 不可用")
        self.client.delete_collection(collection_name)
