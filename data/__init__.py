"""数据层（Data）

职责：纯存储 —— 被动被访问，不主动调用任何模块。
技术选型：向量记忆用 Chroma，结构化数据统一进 SQLite（WAL，见 sqlite_store.py），
KVStoreTool 作为 JSON 门面（tools/storage.py）。
"""
