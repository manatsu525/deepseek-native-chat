# DeepSeek Native Chat

一个面向低配置 VPS 的私人 DeepSeek 聊天站。它只使用 DeepSeek 官方 `deepseek-v4-flash` Responses API，并把 `web_search` 作为服务端工具交给 DeepSeek 执行，不安装 SearXNG、浏览器或网页抓取器。

## 功能

- DeepSeek V4 Flash 原生多轮联网搜索
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

也可以不传密码，安装脚本会生成随机初始密码：

```bash
sudo ./install.sh
```

重复运行安装脚本会更新程序并保留已有数据库、证书和原管理员登录凭据。

默认访问地址为 `https://服务器IP:8000`。证书为自签证书，首次访问需要在浏览器中确认继续。

进入页面后，在右上角打开“API”，填写 DeepSeek 官方 API Key，点击“测试并读取模型”，确认存在 `deepseek-v4-flash` 后保存。

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

DeepSeek 官方 Responses API 当前只有 `deepseek-v4-flash` 支持服务端 `web_search`。Chat Completions 路由以及 NVIDIA NIM 的 Chat Completions 接口没有同等的 provider-executed search，因此本项目固定使用 DeepSeek 官方 `/responses` 路由。

DeepSeek 当前会忽略 Responses API 的 `max_tool_calls`。项目在系统提示中要求单次回答最多搜索五次，但无法像客户端工具循环一样做强制的逐次拦截；实际搜索次数以 DeepSeek 服务端执行结果为准。

DeepSeek V4 Flash Responses API 当前不支持图片或文件输入，所以项目没有提供会误导用户的上传按钮。

## 数据与安全

- 数据库：`/opt/deepseek-native-chat/data/chat.db`
- API Key：保存在仅 root 可读的本地 SQLite 数据库中，前端只返回掩码
- 会话 Cookie：HttpOnly、Secure、SameSite=Lax
- 每账号超过 100 个对话时自动删除最旧对话及关联任务
- 生产环境如有域名，建议把自签证书换成受信任证书
