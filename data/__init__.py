"""数据层（Data）

职责：纯存储 —— 被动被访问，不主动调用任何模块。

技术选型：向量记忆用 Chroma；会话态数据（档案/画像/关系/她的身份/聊天记录/关系账本/
记忆候选池/召回热度）全在**一个 SQLite 库文件**里（WAL + `PRAGMA user_version`
线性迁移，见 sqlite_store.py）。JSON 只是 KV 四张表在 SQLite 里的**值编码**，
不是存储介质——"存 JSON 文件、后续可换 SQLite"的旧说法早已作废。

仍然落在文件里的只有四处（各有各的理由，别顺手搬进 SQLite）。
路径都跟着 `get_settings().data_dir` 走，下面写的是默认 `DATA_DIR=storage/` 时的样子：
- data/persona_config.json：人设种子，**正主就是这个文件**，只读；
- <DATA_DIR>/images/index.json：生成图片的索引；
- <DATA_DIR>/route_config/<session>.json：语音路由配置；
- <DATA_DIR>/tool_calls.jsonl：工具调用日志，追加写。

上层统一门面是 KVStoreTool / VectorStoreTool（tools/storage.py）。
"""
