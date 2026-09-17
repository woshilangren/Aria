# CLAUDE.md

本文件指导 Claude Code 在此仓库工作。项目背景见 README.md，架构详见 ARCHITECTURE.md，迭代方案见 计划与设计.md。

## 项目是什么

Aria——本地自托管的 AI 伴侣（Python 3.10+ / FastAPI / SQLite / Chroma / 原生 HTML 前端）。核心设计目标只有一个：**像人**。所有改动先问一句"这让她的连续性、情绪、记忆、主动权更像一个真人，还是更机器？"

**人设立场（2026-09 拍板，动代码前必读）**：
- **沉浸式**：她是一个真实的人。绝不承认/暗示自己是 AI——这条与早期版本相反，别按旧文档理解；
- **种子+涌现**：`data/persona_config.json` 只是一颗极薄的种子（年轻女孩/20–25/国内某城/名字留白，性格词零定义）。她的名字、年龄、城市、职业、住处由她本人在对话里**第一次说出口时定下并冻结**（KV 表 `self`，机制在 `capability/self_identity.py`）——不许预写、不许代答；
- **性格不变性（宪法）**：用户反馈永远只进关系收敛层（relationship.calm，"收敛不是改变"），绝不碰性格内核；特质涌现只能"发现"不能"安装"；
- **主动性宁缺毋滥**：事件触发 + 用户在场触发，绝不定时冷 ping；`proactive.enabled` 默认关，不招人烦是安全阀不是可选项。

## 常用命令

```bash
.venv\Scripts\python main.py        # 启动（必须用这个，uvicorn main:app 只有单端口 HTTP 且有口令兜底缺口）
.venv\Scripts\python -m pytest      # 全部测试
.venv\Scripts\python -m pytest -m "not integration"   # 只跑纯逻辑单测（不联网不需密钥）
```

- 测试环境变量由 `tests/conftest.py` 注入，`DATA_DIR` 指向临时目录，不碰真实 `.env` 和 `storage/`。
- `requirements.lock` 是本机 pip freeze 快照，不是安装清单；安装用 `requirements.txt`。

## 分层规则（只许向下调用）

```
interaction → orchestration → capability → tools → data
```

- `bootstrap.py` 是**唯一**允许跨层向上 import 的组合根，装配只发生在 `main.py` lifespan 的 startup；`import main` 必须无副作用。
- `shared/singletons.py` 只是注册表（dict + get/mark_error），不 import 任何上层。
- 层间数据契约在 `shared/types.py`（`PromptPackage` / `MemoryBundle` / `FinalReply` 等），字段名是契约，改动需同步所有消费方。
- 能力层需要"该不该停"回调时用参数注入（如 `ToolCallOrchestrator(should_cancel=...)`），不许反向 import 调度层。

## 关键单例与不变式

- `orchestration/pipeline.py` 顶部的 `KEEPER`（短期记忆管家）全项目一份——文字聊天与语音写回共用同一段上下文，谁再 new 一份就是对不上的开始。
- `TurnRegistry`（`orchestration/cancellation.py`）维护"每会话同一时刻最多一个活跃轮"，新轮 start 自动取消旧轮；`cancel(turn_id=None)` **一律拒绝**（F4 防误杀）——**不要**指望客户端先 cancel 再发新消息的时序。
- `SessionMemoryKeeper`：锁内只做内存操作，LLM 摘要/读库等慢调用一律锁外；`get_context` 返回深副本；`append_turn` 成对写入问答。
- 硬事实入库双通道：**正则抽取走 `apply_fact()` 直通**（值没变不入库、值变连改口一起记）；**LLM 抽取必须走 `MemoryGatekeeper`**（S4：候选池 ≥2 次晋升 + 矛盾仲裁 + 14 天过期）——LLM 幻觉不许焊死在档案里。
- 关系/画像/档案/她的身份（self）的更新**必须走 `KVStoreTool.update` 原子闭包**（F3，四个并发写方）；闭包是纯计算，锁内禁网络禁 LLM。
- `writeback` 顺序纪律：`note_user_reply` 必须在 tracker 之后——放前面会预创建空 relationship、吞掉 default_intimacy（实测踩中）。
- 她的身份（名字/年龄/城市/…）唯一来源是 `self` 表（懒生成冻结），取名口子统一走 `self_identity.display_name()`，不许硬编码名字。

## 有意的设计取舍（别当 bug"修"）

改以下任何一条之前先读对应注释和 计划与设计.md 附节：

1. **流式句级审核的统一规则**：还没推出任何内容才允许重写（最多 `_MAX_REWRITE=2` 次），推出过就只能整轮降级。判据用 `pushed` 标志，**不能用** `streamer.emitted_content`（"装配出来过"≠"真推出去了"）。
2. **工具链不做审核重写**，只有人设链做。
3. **取消之后什么都不写**：不写 KEEPER、chat_log、蒸馏、亲密度、画像。写任何东西都会让"这轮当没发生过"不彻底。
4. **KV read/write 表故意不对称**：`persona_config` 只读（正主是 `data/persona_config.json`，现在是"种子"）；`storage.py` 里的"别顺手对齐"注释是写给后来人的。原子更新白名单 `_KV_UPDATE_TABLES` 收紧到整包语义的表，别扩。
5. **工具结果用 `role:"user"` 文本回填、丢弃 call_id**：GLM 兼容接口不吃 `role:"tool"`。换 OpenAI 系模型时可按 `toolcall.py` 注释列的三处升级为标准写法。
6. **e2e / realtime 两条语音链路没有本地输出审核**——这是缺口不是取舍，见 计划与设计.md。
7. 长期记忆全局共享不按 session 过滤（跨设备的"同一个人"）。
8. `@r` 反应前缀协议已接通（F8）：persona_engine 注入规则 + 配置 `expression.reaction_tag`；writeback 前剥标记。
9. **LLM 参数按模型家族发**（llm_client._is_qwen）：`enable_thinking`/`temperature` 是 qwen 专有/Sonnet5 已移除——对 Claude 一律不发，否则 400 → 熔断全切备用。
10. **召回重排是调制不是重写**（S1）：负面记忆半衰更短是自我保护，别"修"成对称；低分记忆配"不确定"措辞（S7）。

## 平台坑（Windows / 语音）

- 音频铁律：前端固定 16kHz 单声道裸 PCM16（**不用 MediaRecorder**）；裸 PCM 进 ASR 前用 `pcm_to_wav` 包头；回程 24k PCM 同样包头；TTS 回 mp3；前端按 RIFF 魔数区分。
- WS 控制信号必须是**文本帧**（"END"/"ping"）——`TextEncoder().encode()` 会变成二进制帧，后端当音频收（开发日志.md Bug 档案）。
- 回程音频必须从用户手势解锁的 AudioContext 播放，否则手机静默拒绝。
- `_DASHSCOPE_LOCK` 把"设 key + 调用"整段串行化，因为 `dashscope.api_key` 是模块级全局。
- Chrome 只在有消费者时保持麦克风打开：采集图建一次常驻，靠 `recording` 标志丢数据，不要反复拆建。

## 配置体系

- `.env`：密钥与地址。主 LLM = claude-sonnet-5（nonelinear OpenAI 兼容端点 `/v1`，注意 `/anthropic` 是另一套协议没接）；备用 GLM；ASR / TTS / 向量 / 画图走百炼——**LLM_API_KEY 与语音/向量 key 是独立变量**，换 LLM 供应商不影响语音。
- `config.json`：业务参数，全部有内置默认值；`load_app_config()` 是 lru_cache 的（运行时改参数靠 API 端点原地改缓存对象）。`proactive.enabled` 默认关。
- `data/persona_config.json`：**种子**（极薄：身份锚点 + 沉浸式铁律 + 性格不变性），不是完整人设——她的名字与过去在 KV `self` 表里由她自己冻结，代码机制在 `capability/self_identity.py`。
- config.json / persona_config.json 写坏都不能让程序起不来——解析失败走默认值并告警。

## 设计原则（规划中的迭代以此为纲，详见 计划与设计.md 第三节（设计总纲））

- **单一真相源 + 冻结**：感受、当下细节这类主观状态在生成时定死，召回时只调制不重写（同事件两次回忆两种说法 = 自我打脸）。
- **对 LLM 抽取的记忆先怀疑、再验证、慢接受**：候选晋升（≥2 次）+ 矛盾仲裁 + 过期清理，幻觉不许焊死在档案里。
- **数值更新必须原子**（`UPDATE ... score=MAX(0,MIN(100,score+?))` 写后回读），禁止快照回写；每次变动进账本留痕。
- **任何 LLM 评判者必须能看到人设**——否则会系统性惩罚人设规定的言行（姊妹项目实测教训）。
- **主动性宁缺毋滥**：事件触发 + 用户在场触发，绝不定时冷 ping；不招人烦是安全阀不是可选项。

## 兜底哲学

工程量大头在兜底上。新功能必须回答：LLM 挂了怎样？存储挂了怎样？参数坏了对不对？原则：
- 存储读不到返回默认值而不是抛异常；单个服务初始化失败不拦启动（`bootstrap.py` 每服务独立 try）。
- 出错路径上的人设兜底话（`FallbackController.fallback_reply`）读不到人设也必须无缝落回内置默认。
- 日记/画像失败静默，聊天照常；但"静默遗忘"（记忆写入失败）是需要补偿队列的问题，见 计划与设计.md 批次 J2。

## 代码风格

- 中文注释，注释密度高且解释"为什么"而非"是什么"——新代码保持这个密度；有意的取舍必须写成注释（参照 `toolcall.py` 的 call_id 注释格式）。
- 每个模块顶部 docstring 讲清职责与设计动机。
- 不引入重量级编排框架（已有一次去 LangGraph 的教训）；能用普通 async 管道就不用图。
