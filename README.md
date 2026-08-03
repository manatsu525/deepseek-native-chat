# DeepSeek Native Chat

一个面向低配置 VPS 的私人 DeepSeek / Xiaomi MiMo 聊天站。联网搜索由所选服务商的原生工具执行，不安装 SearXNG、浏览器或网页抓取器。

## 功能

- DeepSeek V4 Flash Responses API 原生多轮联网搜索
- 小米 MiMo V2.5 / V2.5 Pro Chat Completions 原生联网搜索
- 流式回答、思考过程、搜索步骤和来源折叠展示
- 输入、缓存命中、输出和推理 Token 统计
- 后台生成：刷新页面、切换应用或锁屏后任务继续运行
- 随时停止生成，保留已经输出的内容
- Markdown、表格和代码块渲染；代码块可下载
- 消息复制与重新回答
- 每账号最多保留 100 个对话，历史每页显示 10 个
- 最多 3 个账号，管理员可在前端新增或删除账号
- 各账号的对话和 API 配置相互隔离
- 前端测试 API、读取模型列表、手动填写模型名、删除 API
- MiMo 联网参数：最大关键词数、每轮结果数、强制搜索、用户位置、思考开关和生成上限
- MiMo 按需网页读取：模型可自动调用免费 Jina Reader 读取公开网页、文档页或 PDF，无需额外 API Key
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

进入页面后，在右上角打开“API”，选择服务商并填写对应 API Key，点击“测试并读取模型”后保存。DeepSeek 使用 `https://api.deepseek.com`；MiMo 使用 `https://api.xiaomimimo.com/v1`。MiMo 联网搜索需要先在小米控制台开通联网服务插件。

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

DeepSeek 使用官方 `/responses` 路由，当前原生 `web_search` 适配 `deepseek-v4-flash`。MiMo 的联网搜索按官方示例使用 `/v1/chat/completions`，工具参数由 MiMo 服务端处理并在流式响应的 annotations 中返回来源；MiMo 同时支持与自定义 Function 混合调用，后端将 `fetch_webpage` 注册为 `tool_choice=auto` 的函数工具，模型自己决定是否读取页面。两种协议由后端分别解析，不能共用 DeepSeek 的 Responses SSE 事件处理器。

`fetch_webpage` 不直接抓网页，而是把模型选中的公开 URL 交给 `https://r.jina.ai/`，得到干净 Markdown 后再作为 `tool` 结果回传给 MiMo。Jina Reader 不需要 API Key；后端按约 20 次/分钟做进程级节流，每个回答最多读取 5 页，每页最多保留约 12,000 字符，并拒绝本机、内网和带账号密码的 URL。模型仍然自行判断什么时候搜索、什么时候细读，后端只负责执行工具和防止异常循环。

DeepSeek 当前会忽略 Responses API 的 `max_tool_calls`。项目在系统提示中要求单次回答最多搜索五次，但无法像客户端工具循环一样做强制的逐次拦截；实际搜索次数以 DeepSeek 服务端执行结果为准。

DeepSeek V4 Flash Responses API 当前不支持图片或文件输入，所以项目没有提供会误导用户的上传按钮。MiMo 的多模态能力不在本次聊天功能范围内。

## 数据与安全

- 数据库：`/opt/deepseek-native-chat/data/chat.db`
- API Key：保存在仅 root 可读的本地 SQLite 数据库中，前端只返回掩码
- 会话 Cookie：HttpOnly、Secure、SameSite=Lax
- 每账号超过 100 个对话时自动删除最旧对话及关联任务
- 生产环境如有域名，建议把自签证书换成受信任证书
