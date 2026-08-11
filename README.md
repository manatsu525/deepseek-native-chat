# DeepSeek Native Chat

一个面向低配置 VPS 的私人 DeepSeek / Custom 聊天站。DeepSeek 使用服务端原生联网；Custom 使用标准 OpenAI Chat Completions，可选择 Parallel、Keenable、Tavily、Firecrawl、You.com 或 DDG + Jina 匿名联网方案，不安装 SearXNG 或浏览器。

## 功能

- DeepSeek V4 Flash Responses API 原生多轮联网搜索
- Custom OpenAI 兼容 Chat Completions 外部多轮搜索与网页读取
- 流式回答、思考过程、搜索步骤和来源折叠展示
- 输入、缓存命中、输出和推理 Token 统计
- 后台生成：刷新页面、切换应用或锁屏后任务继续运行
- 随时停止生成，保留已经输出的内容
- 每条消息最多上传 10 个图片/文件，原始附件合计不超过 50MB
- 图片低内存缩放后发送给 Custom 多模态模型；PDF、DOCX、XLSX、文本和代码文件在本地限量提取文字
- Markdown、可换行横向滚动表格、KaTeX 行内/块级公式和代码块渲染；代码块可复制、下载
- 消息复制与重新回答
- 每账号最多保留 100 个对话，历史每页显示 10 个
- 最多 3 个账号，管理员可在前端新增或删除账号
- 各账号的对话和 API 配置相互隔离
- 前端测试 API、读取 `/models` 全部模型并勾选、手动填写模型名、重新编辑已保存 API 的模型列表、删除 API
- Custom 参数：所有模型均可开关 `thinking`，可独立开关 `reasoning_effort` 并使用顶部 High/Max，也可按需发送 `include_reasoning: true`，另有普通采样参数、默认 65,536 的生成上限和联网方案切换
- Custom 匿名联网方案：Parallel Search + Fetch、Keenable Search + Live Fetch、Tavily Keyless Search + Extract、Firecrawl Keyless Search + Scrape、You Search + Jina Fetch、DDG Search + Jina Fetch
- SQLite 单文件数据库、自签 HTTPS、systemd 守护
- 手机和桌面端响应式界面
- 浏览器自动上报 IANA 时区；Custom/MiMo 每次回答都会获得对应的当前本地日期，并要求按绝对日期核对“今天/最新”等时效性问题
- 每个账号可以一键清空自己的全部聊天记录；级联删除消息、思考、搜索记录和任务后会截断 WAL 并压缩 SQLite，实际释放 VPS 磁盘空间
- NVIDIA Build 的 DeepSeek V4 Flash/Pro 会按官方协议发送 `chat_template_kwargs.thinking` 与 High/Max `reasoning_effort`，并兼容解析 `reasoning`/`reasoning_content`
- OpenCode Zen 的 `deepseek-v4-flash(-free)` 可选用独立 DSML fallback：新配置默认不勾选，用户可在任意 Custom 配置中手动开关；仅匹配 OpenCode host + 对应模型时生效，并且只在原生 `tool_calls` 为空时恢复工具调用，也可用 `OPENCODE_DSML_FALLBACK=0` 全局停用
- NVIDIA Nemotron 3 Ultra 使用 `enable_thinking`、工具兼容标志和 16K reasoning budget；GLM-5.2 使用 `thinking.enabled` 与 High/Max `reasoning_effort`

## 资源占用

运行时只有一个 Python/FastAPI 进程，不需要 Docker。systemd 默认限制最大内存为 180 MB，适合约 350 MB 内存并带有 Swap 的小型 VPS。SQLite 使用 WAL 模式，不需要独立数据库服务。附件原始数据流式落盘，图片和文档处理串行执行，带附件的模型任务也全局串行，避免多账号同时构造图片请求导致内存峰值叠加。

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

DeepSeek 使用官方 `/responses` 路由，当前原生 `web_search` 适配 `deepseek-v4-flash`。Custom 使用标准 `/chat/completions`，不依赖模型供应商的原生搜索；后端把联网能力适配成模型可见的 `web_search` 与 `fetch_webpage` Function Tools。默认方案匿名连接 Parallel 官方 Streamable HTTP MCP；也可以手动选择 Keenable、Tavily Keyless、Firecrawl Keyless、You.com Free 或 DDG + Jina。选择后，搜索与抓取固定使用对应的一组实现，不自动切换、不 fallback、不同 provider 之间不并发。You.com Free 只提供搜索，因此它是唯一复用现有 Jina Reader 抓取的新增方案。查询词和待读取 URL 会发送给当前选择的服务。两种模型协议由后端分别解析，不能共用 DeepSeek 的 Responses SSE 事件处理器。

新增匿名 MCP 由独立的轻量客户端适配：Keenable 调用 `https://api.keenable.ai/mcp` 的 `search_web_pages` / `fetch_page_content`，抓取固定发送 `live=true`；Tavily 调用 `https://mcp.tavily.com/mcp/` 的 `tavily_search` / `tavily_extract`，每次请求固定发送 `X-Tavily-Access-Mode: keyless`；Firecrawl 调用 `https://mcp.firecrawl.dev/v2/mcp`，只使用 `firecrawl_search` / `firecrawl_scrape`；You.com 调用 `https://api.you.com/mcp?profile=free` 的 `you-search`，抓取则直接复用原来的 Jina Reader。四种服务的原始响应都会归一化为相同的标题、URL、摘要和正文格式，再作为 `tool` 消息回传给 LLM。

Custom 每个工具轮最多执行 1 个工具，最多 6 个工具轮；每个回答最多搜索 3 次、最多读取 3 个网页，每次搜索最多向模型返回 10 条结果。模型可以在资料足够时直接停止工具调用。搜索查询按标准化文本去重，不会重复请求同一组查询。`fetch_webpage` 只能读取用户提供或当前搜索方案返回的 URL，搜索引擎结果页、编造 URL、本机和内网地址都会被拒绝；同一个 URL 也不会重复注入正文。Parallel 搜索每条摘录最多保留约 1,200 字符，读取默认使用相关摘录而非完整页面，每页最多约 8,000 字符，以限制上下文增长。

前端发起问题时会同时提交浏览器的 IANA 时区（如 `Asia/Shanghai`）。后端只把“当前本地日期 + 时区”追加到 Custom 的固定系统提示词末尾，并要求模型把“今天、昨天、明天、目前、最新”等相对时间转换为绝对日期，核对来源的发布/事件日期，禁止把搜索返回的最新一篇误当作当天资料。日期每天只变化一次，且放在静态提示之后，以尽量保留固定前缀的缓存价值；旧客户端未提交时区时使用 UTC。

Custom 的 `thinking`、`reasoning_effort` 和 `include_reasoning` 都由用户控制：`thinking` 开/关会发送给每个模型，`reasoning_effort` 有独立开关，开启时使用顶部 High/Max；`include_reasoning` 默认不发送，勾选后才发送顶层 `include_reasoning: true`。MiMo、NVIDIA DeepSeek V4 和 NVIDIA Nemotron 3 Ultra 使用其官方请求方言，其他 Custom 使用通用顶层 `thinking` / `reasoning_effort`；OpenAI 兼容协议本身并未标准化这些扩展字段，因此不兼容的供应商可能返回参数错误，此时可在 Custom 参数中分别关闭。后端兼容解析 `reasoning` / `reasoning_content` 输出。工具额度结束后，后端会强制进入最终作答阶段；模型若把工具请求伪装成 `<tool_call>` 文本，该输出不会被保存为答案，而会进行一次受限纠正。

备用方案的 `fetch_webpage` 把公开 URL 交给 `https://r.jina.ai/`，得到干净 Markdown 后作为 `tool` 结果回传给 Custom。Jina Reader 不需要 API Key；后端按约 20 次/分钟做进程级节流，每个回答最多读取 3 页，每页最多保留约 8,000 字符，并设置 `X-Respond-With: markdown`、`X-Timeout: 30` 和 `X-Remove-Selector`，自动移除常见 header、nav、aside、footer、sidebar、菜单、广告和 cookie 弹窗元素。Jina 官方支持通过 `X-Remove-Selector` 排除这些 CSS 选择器；如果目标站点有明确的文章容器，后续还可以针对该站点增加 `X-Target-Selector`。

DeepSeek 当前会忽略 Responses API 的 `max_tool_calls`。项目在系统提示中要求单次回答最多搜索五次，但无法像客户端工具循环一样做强制的逐次拦截；实际搜索次数以 DeepSeek 服务端执行结果为准。

输入框左下角的回形针可以一次选择多个附件。每条消息最多 10 个，所有原始文件合计最多 50MB。浏览器逐个上传，服务器以数据流落盘，处理完成后立即删除原文件：

- JPG、PNG、WebP、GIF、BMP：只处理第一帧，最多约 1600 万像素，最长边缩到 1600 像素，并压缩为不超过约 1.5MB 的 JPEG；随后通过 OpenAI Chat Completions 的 `image_url` Base64 格式发送。必须选择真正支持视觉输入的 Custom 模型（例如官方 `mimo-v2.5`）；普通文本模型会由对应 API 拒绝。
- PDF：使用受 96MB 地址空间、35 秒 CPU 和 45 秒墙钟限制的 `pdftotext` 子进程，最多提取前 30 页、每个文件最多 30,000 字符。扫描版 PDF 不做 OCR。
- DOCX：流式读取正文 XML；XLSX：最多读取前 5 个工作表和每表 2,000 行，并限制共享字符串内存。
- 常见文本、Markdown、CSV、JSON 和代码文件：按文本读取。旧版 `.doc`、`.xls` 需要先另存为 `.docx`、`.xlsx`。

一次回答中的全部文档摘录最多 80,000 字符。附件本体只用于本次回答，任务完成、停止或失败后立即删除；历史记录仅保留名称和原始大小，因此“重新回答”不会自动重传旧附件。尚未发送的附件按草稿保留，刷新页面可以恢复，超过 24 小时会自动清理。DeepSeek Responses 路由仍不接收图片，但可以使用已提取为文字的文档；图片请切换到支持视觉的 Custom 模型。

## 数据与安全

- 数据库：`/opt/deepseek-native-chat/data/chat.db`
- API Key：保存在仅 root 可读的本地 SQLite 数据库中，前端只返回掩码
- 会话 Cookie：HttpOnly、Secure、SameSite=Lax
- 每账号超过 100 个对话时自动删除最旧对话及关联任务
- 生产环境如有域名，建议把自签证书换成受信任证书
