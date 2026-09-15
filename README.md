# Aria — 可自定义人设的本地 AI 伴侣

Aria 是一个跑在自己电脑上的 AI 伴侣程序：能聊天、能打电话、能记住你说过的事、会在闲置时主动写日记。它不依赖任何在线服务平台，对话数据全部落在本地 `storage/` 目录，换人设只改一个配置文件。

它解决的问题是：大多数 AI 伴侣产品是"无状态的请求-响应"——秒回、全知、上句不接下句、记不住你上周说过什么。Aria 把"连续的记忆、有过程感的情绪、会遗忘的画像、自己的日程"做进了架构里，而不只是写进提示词。

## 核心特性

**记忆系统**
- 长期记忆：对话中的重要事实自动沉淀进向量库（Chroma），按语义召回，而不是靠上下文窗口硬塞
- 记忆防污染：值没变不重复入库；值变了会连同"改口"一起记录，以新的为准
- 用户画像：聊天中逐步积累兴趣标签、核心需求、相处感受，标签 14 天不出现自动衰减
- 关系数值：亲密度、信任、关系阶段随交互演化，影响说话的分寸
- 日记：闲置半小时以上自动把当天对话整理成一篇第一人称日记，落进向量库，之后聊天能自然提起

**语音通话**
- 三条可热切换的链路：级联（ASR → LLM → TTS）、端到端、Realtime 实时专线（直连多模态模型）
- 情绪语音：回复开头的 `[情绪]` 标记会转成 TTS 情绪指令，合成前剥除，不进聊天记录
- 手机浏览器直接可用：自签证书 + PWA，"添加到主屏幕"后全屏独立运行

**工具调用**
- 内置天气、联网搜索、图片生成、写日记四个工具
- 三级降级：模型原生 tool call → 模型文本规划 → 规则表，模型不支持 function calling 也能跑

**对话体验**
- 逐句流式：模型生成到哪一句就发哪一句，不用等整条说完。文字聊天走 SSE，一句一个气泡
- 随时打断：正生成时能插话，上一句不会被丢掉——新输入会和它合并成一条，一起回复
- 时间感知：每轮注入实时时间 + 时段分寸提示，凌晨三点不会问"吃晚饭了吗"

**稳定性设计**
- 主备 LLM 熔断切换：主模型连续失败到阈值就打开熔断，冷却期（默认 120 秒）内完全不碰它；冷却到期放行一次探测，探测用 5 秒短超时，成功即闭合。这样设计是因为交互式聊天是低频场景——按请求计数或复用 60 秒超时，都会让"几乎每条消息白等一分钟"
- 句级内容审核：每生成一句就过一次违规检查，过了才推给你
- 审核没过的裁决是一条统一规则：**还没推出任何内容就允许重写，一旦推出过就只能整轮替换成兜底话**
- `<voice>` 里的语音内容同样过审，且先审后合成：不会被念出来，也不会落库
- 轮次可取消：被打断的那一轮什么都不写，不会污染记忆和关系数值

## 快速开始

环境要求：Python 3.10+（项目在 Windows 上开发，其他平台理论可用）。

```bash
git clone https://github.com/<你的用户名>/aria-companion.git
cd aria-companion

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 复制环境变量模板，填入你的 API Key
copy .env.example .env

.venv\Scripts\python main.py
```

启动后：

| 地址 | 用途 |
|---|---|
| http://127.0.0.1:8000/ | 主界面（本机桌面） |
| https://<本机局域网IP>:8443/ | 主界面（手机浏览器，Chrome 等严格浏览器需要 HTTPS 才能开麦克风） |
| http://127.0.0.1:8000/ui | 旧版 Gradio 聊天页（兜底） |
| http://127.0.0.1:8000/api/system/health | 健康检查 |

首次启动会自动生成自签名证书（`storage/certs/`，十年期，含全部网卡 IP）。手机浏览器访问 `https://IP:8443` 时点"高级 → 继续访问"即可；也可以下载根证书安装（`/ca.crt`）实现无警告访问。

首次启动时如果 `.env` 里没配 `ACCESS_TOKEN`、而监听地址又是 `0.0.0.0`（默认），程序会**自动生成一个随机访问口令写进 `.env` 并在控制台打印**。浏览器第一次访问会弹框，把控制台那串粘进去就行，之后存在本地不用再输。只想本机使用的话把 `BIND_HOST` 改成 `127.0.0.1`，那时不配口令也不会自动生成。

要 HTTPS 就必须用 `python main.py`——`uvicorn main:app` 只有单端口 HTTP，应用对象管不了第二个端口。

## 配置

配置分两处，各管各的：

**`.env`（密钥与地址）** — 从 `.env.example` 复制。主 LLM / 语音识别 / 语音合成 / 向量 / 画图走阿里云百炼（DashScope），一把 key 通吃；备用 LLM 默认配 GLM，只在主模型连不上时顶上。天气用 Open-Meteo（免费无 key），搜索用 DuckDuckGo（免费无 key）。

**`config.json`（业务参数）** — 温度、重试次数、闲置分钟数、记忆条数上限等，全部有内置默认值，不建也能跑。其中 `personality.quirk_rate` 控制随机小动作的触发概率（0~1，0 为关闭），`llm.enable_thinking` 控制 Qwen 系模型的思考开关。运行中可通过设置面板在线改，写回文件立即生效。

**`data/persona_config.json`（人设）** — 见下节。

## 自定义人设

人设不走代码，走配置。`data/persona_config.json` 是唯一的人设来源，改这个文件就能换掉整个角色：

| 字段 | 作用 |
|---|---|
| `char_name` / `char_id` | 角色名字（界面、日记署名、对话标签都从这取） |
| `background_story` | 人设正文：身份、性格内核、喜好、雷点、说话风格 |
| `taboos` | 硬性规矩：永远不许越过的线 |
| `mode_tones` | 分模式语气：`chat`（日常）和 `comfort`（安抚）各一份 |
| `self_introduction` / `age` / `hobbies` | 基础信息 |
| `default_intimacy` | 初始亲密度（0~100） |
| `fallback_replies` | 异常兜底话术，按错误类型给一句符合角色的台词 |
| `ask_templates` | 缺信息时的追问话术（问城市、生日、称呼、职业） |
| `diary_notes` | 写日记的口吻补充要求 |

仓库自带一套中性示例人设「Aria」。想要一个毒舌助手、温柔树洞或者文言文说书人，把 `background_story` 和 `mode_tones` 换掉即可，代码一行不用动——兜底话术、追问话术、日记口吻都会跟着人设文件走。

## 项目结构

```
main.py                  程序入口：建应用 + lifespan 装配 + 双端口启动
run_local.py             等价于 python main.py（兼容旧的启动习惯）
bootstrap.py             组合根：全项目唯一允许跨层向上 import 的地方
config.json              业务参数（可在线修改并写回）
config/settings.py       环境变量与业务配置的读取、默认值
interaction/             Web 层：REST/WS/SSE 路由、消息网关、Gradio 兜底页
orchestration/           调度层：pipeline.py（对话主流程）、cancellation.py（轮次取消）、managers.py（路由与降级）
capability/              能力层：感知、记忆、人设组装、回复生成、日记、工具编排
tools/                   工具层：LLM 客户端、语音 SDK、外部 API、日志、证书
data/                    数据层：SQLite、Chroma、人设正主文件
shared/                  层间契约（types.py）与服务注册表（singletons.py）
frontend/                自定义主界面（单文件 HTML）+ PWA 资产
docs/                    两份深度解读：项目全景、代码逐层导读
storage/                 运行时数据（.gitignore 已排除，备份这一个目录即可）
```

分层规则：`interaction → orchestration → capability → tools → data`，只许向下调用。服务的装配不在 import 期做，而是放在应用 startup 的 lifespan 里——所以 `import main` 没有副作用。

一轮对话的完整链路、兜底逻辑、语音三条通道的取舍，见 [ARCHITECTURE.md](ARCHITECTURE.md)。这一轮改了什么、为什么改，见 [修改日志.md](修改日志.md)。

## 数据与隐私

- 所有数据（聊天记录、记忆库、日记、档案、证书、上传文件）都在 `storage/` 一个目录里，不上云、不上报
- `.env` 里的 API Key 不会进 git。对外监听时务必配 `ACCESS_TOKEN`——没配的话程序会自己生成一个（见上面"快速开始"）
- 上传的文件（`storage/uploads/`）默认需要口令才能访问，不再对局域网里任何人开放
- 想清空重来，删掉 `storage/` 目录重启即可，程序会自动重建

## 已知未验证的地方

不假装都测过（详见 [修改日志.md](修改日志.md) 第十节）：

- SSE 流式**没有经过内网穿透实测**。如果你用 cpolar 之类把服务穿到手机上，要自己确认句子是不是真的逐句到达——部分穿透服务会缓冲 SSE，症状是攒到最后一起冒出来，本机测不出来
- `requirements.lock` 是本机 `pip freeze` 出来的参考，没在全新机器上验证过安装

## 本地跑测试

测试资产都在 `tests/` 下，用项目自己的虚拟环境跑：

```bash
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest
```

- `tests/unit/` —— 纯逻辑单测（对话管道的判定函数、感知的词表兜底、工具调用的规则与安检），
  **不启应用、不联网、不需要任何密钥**。
- `tests/integration/` —— 真起 FastAPI 应用的冒烟测试（健康检查 / 口令门卫 / 路由注册），
  标记为 `integration`。环境不满足（导不进应用、起不动应用）时会**优雅 skip，不会 fail**。

只跑单元测试、跳过集成测试：

```bash
.venv\Scripts\python -m pytest -m "not integration"
```

测试进程由 `tests/conftest.py` 注入**假环境变量**，`DATA_DIR` 一律指向临时目录——
**不会读写 `.env` 里的真实密钥，也不会写进仓库的 `storage/`**。

CI 见 `.github/workflows/ci.yml`：`push` / `pull_request` 到 `master` 时用 Python 3.14
装依赖并跑 `pytest`（不配置任何 secrets，离线可跑）。

## License

[MIT](LICENSE)
