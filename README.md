# LangChain 文章投稿智能体

## 运行

1. 创建并激活虚拟环境，安装依赖：`pip install -r requirements.txt`。
2. 安装 Playwright Chromium：`playwright install chromium`。
3. 复制 `.env.example` 为 `.env`，配置 `OPENAI_API_KEY` 或 `DASHSCOPE_API_KEY`，并填写平台用户名和密码。
4. 按目标站点设置 `.env` 中的 `SITE_A_URL`、`SITE_A_USERNAME` 和 `SITE_A_PASSWORD`。
5. 运行 `python agent.py`，输入类似“投 10 篇文章到 A 平台”的需求。

当前流程为：大模型从请求中识别站点名称，Playwright 打开站点登录页，大模型识别用户名、密码和登录按钮的 CSS 选择器并将其传给 `read_configs_login` 工具执行登录；登录成功后，大模型分析当前页面并返回 `SubmissionSchema` 格式的必填项。登录选择器无需手动配置。账号密码只从 `.env` 读取，不会传给大模型。

当前仅实现登录和识别当前页面必填项；尚未实现逐步填写和提交文章。请将 `.env.example` 中示例站点的 URL 和登录凭据替换为目标站点的真实值。
