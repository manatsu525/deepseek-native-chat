# DeepSeek Native Chat

一个面向低配置 VPS 的私人 DeepSeek / Custom 聊天站。DeepSeek 使用服务端原生联网；Custom 使用标准 OpenAI Chat Completions，并可调用本地 DuckDuckGo 搜索和 Jina Reader，不安装 SearXNG 或浏览器。

## 功能

- DeepSeek V4 Flash Responses API 原生多轮联网搜索
- Custom OpenAI 兼容 Chat Completions 外部 DuckDuckGo 多轮搜索
- 流式回答、思考过程、搜索步骤和来源折叠展示
- 输入、缓存命中、输出和推理 Token 统计
- 后台生成：刷新页面、切换应用或锁屏后任务继续运行
- 随时停止生成，保留已经输出的内容
- Markdown、表格和代码块渲染；代码块可下载
- 消息复制与重新回答
- 每账号最多保留 100 个对话，历史每页显示 10 个
- 最多 3 个账号，管理员可在前端新增或删除账号
- 各账号的对话和 API 配置相互隔离
- 前端测试 API、读取 `/models` 全部模型并勾选、手动填写模型名、删除 API
- Custom 参数：MiMo 专用 thinking 开关、普通 Custom 采样参数和生成上限；搜索预算由后端固定保护
- Custom 按需网页读取：模型可自动调用免费 Jina Reader 读取公开网页、文档页或 PDF，无需额外 API Key
- SQLite 单文件数据库、自签 HTTPS、systemd 守护
- 手机和桌面端响应式界面

## 资源占用

运行时只有一个 Python/FastAPI 进程，不需要 Docker。systemd 默认限制最大内存为 180 MB，适合约 350 MB 内存并带有 Swap 的小型 VPS。SQLite 使用 WAL 模式，不需要独立数据库服务。

## 安装

Debian/Ubuntu 上使用 root 执行：

```bash
git clone https://github.com/manatsu525/deepseek-native-chat.git
cd deepseek-native-chat
sudo ./install.sh admin '至少八位的密码'
```

不传第二个参数时，初始管理员密码为 `admin123456`。首次登录后请立即在“账号管理”中修改为个人管理密码：

```bash
sudo ./install.sh
```

重复运行安装脚本会更新程序并保留已有数据库、证书和原管理员登录凭据。

默认访问地址为 `https://服务器IP:8000`。证书为自签证书，首次访问需要在浏览器中确认继续。

进入页面后，在右上角打开“API”，接口类型只分为 DeepSeek 和 Custom。填写 Custom 的 OpenAI 兼容 API 地址和 API Key 后，点击“测试并读取模型”；`/models` 返回的模型会以复选框显示，也可以手动填写不在列表中的模型名并测试。DeepSeek 使用 `https://api.deepseek.com`，Custom 默认使用 `https://api.openai.com/v1`，可改成任意兼容接口地址。

升级旧版本时，原来的 MiMo API 配置会自动迁移为 Custom，已保存的模型和对话不会删除。

## 卸载

交互确认：

```bash
sudo ./uninstall.sh
```

无人值守完全删除：

```bash
sudo ./uninstall.sh --yes
```

卸载会删除 `/opt/deepseek-native-chat`，包括账号、API Key、聊天记录和自签证书。

## 设计说明

DeepSeek 使用官方 `/responses` 路由，当前原生 `web_search` 适配 `deepseek-v4-flash`。Custom 使用标准 `/chat/completions`，不依赖供应商原生搜索；后端注册一个模型可见的 `web_search` Function Tool，直接请求 DuckDuckGo Lite，失败时再尝试 DuckDuckGo HTML 页面。这样任何兼容 Chat Completions 的模型都可以复用同一套本地联网工具，搜索结果 URL 也由后端统一登记，再交给读取器使用。两种协议由后端分别解析，不能共用 DeepSeek 的 Responses SSE 事件处理器。

Custom 每个工具轮最多执行 1 个工具，最多 6 个工具轮；每个回答最多搜索 3 次、最多读取 3 个网页，每次搜索最多返回 10 条结果。模型可以在资料足够时直接停止工具调用。搜索查询按标准化文本去重，不会重复请求同一个查询。`fetch_webpage` 只能读取用户提供或 DuckDuckGo 返回的 URL，搜索引擎结果页、编造 URL、本机和内网地址都会被拒绝；同一个 URL 也不会重复注入正文。

MiMo 模型使用小米接口的 `thinking` 与 `max_completion_tokens` 字段；普通 Custom 模型使用标准 `max_tokens`、`temperature` 和 `top_p`，不会收到 MiMo 专用字段。工具额度结束后，后端会强制进入最终作答阶段；模型若把工具请求伪装成 `<tool_call>` 文本，该输出不会被保存为答案，而会进行一次受限纠正。

`fetch_webpage` 把公开 URL 交给 `https://r.jina.ai/`，得到干净 Markdown 后作为 `tool` 结果回传给 Custom。Jina Reader 不需要 API Key；后端按约 20 次/分钟做进程级节流，每个回答最多读取 3 页，每页最多保留约 8,000 字符，并设置 `X-Respond-With: markdown`、`X-Timeout: 30` 和 `X-Remove-Selector`，自动移除常见 header、nav、aside、footer、sidebar、菜单、广告和 cookie 弹窗元素。Jina 官方支持通过 `X-Remove-Selector` 排除这些 CSS 选择器；如果目标站点有明确的文章容器，后续还可以针对该站点增加 `X-Target-Selector`。

DeepSeek 当前会忽略 Responses API 的 `max_tool_calls`。项目在系统提示中要求单次回答最多搜索五次，但无法像客户端工具循环一样做强制的逐次拦截；实际搜索次数以 DeepSeek 服务端执行结果为准。

DeepSeek V4 Flash Responses API 当前不支持图片或文件输入，所以项目没有提供会误导用户的上传按钮。Custom 的多模态能力取决于所接入的兼容服务和模型，本项目当前只传文本与工具调用。

## 数据与安全

- 数据库：`/opt/deepseek-native-chat/data/chat.db`
- API Key：保存在仅 root 可读的本地 SQLite 数据库中，前端只返回掩码
- 会话 Cookie：HttpOnly、Secure、SameSite=Lax
- 每账号超过 100 个对话时自动删除最旧对话及关联任务
- 生产环境如有域名，建议把自签证书换成受信任证书
