# 大模型每日知识点推送到飞书 (daily-llm-tip)

这个仓库实现：
- 每天自动调用大模型随机生成一个关于“大模型”的知识点（主题、解释、举例、参考），以 JSON 格式返回并解析；
- 将知识点发送到飞书（通过自定义机器人 Webhook）；
- 使用 GitHub Actions 定时每天运行（也支持手动触发）。

主要文件
- `src/generate_and_send.py` — 主脚本，调用 LLM 并发送到飞书
- `.github/workflows/daily.yml` — GitHub Actions 定时任务（每天运行）和手动触发
- `requirements.txt` — Python 依赖
- `.env.example` — 环境变量示例

快速开始
1. 在本地创建并保存这些文件，然后将仓���推送到 GitHub（见下面命令）。
2. 在仓库设置 -> Secrets and variables -> Actions 中添加以下 Secrets：
   - `FEISHU_WEBHOOK`：飞书自定义机器人 Webhook（完整 URL）
   - `LLM_API_KEY`：大模型服务的 API Key（例如 OpenAI API Key）
   可选：
   - `LLM_PROVIDER`：默认 `openai`，支持 `openai` 或 `generic`
   - `LLM_MODEL`：默认 `gpt-3.5-turbo`
   - `LLM_API_URL`：当 `LLM_PROVIDER=generic` 时设置为模型提供者的请求 URL
3. 手动测试（本地）：
   - 创建 `.env`（或在环境中导出）：
     - `export FEISHU_WEBHOOK="https://open.feishu.cn/..."`  
     - `export LLM_API_KEY="sk-..."`  
   - 安装依赖并运行：
     - `pip install -r requirements.txt`
     - `python src/generate_and_send.py`
4. 工作流说明：
   - `.github/workflows/daily.yml` 默认每天 UTC 09:00 触发（可修改为你想要的 cron 时间）。
   - 你也可以在 GitHub Actions 页面使用 “Run workflow” 手动触发。

注意事项
- 飞书 Webhook：请在飞书群里添加自定义机器人并获取 Webhook URL（或使用企业自建机器人），确保机器人可发送消息到目标群/用户。
- 模型费用/速率：注意模型调用会产生成本；确保 API Key 有配额。
- 如果使用非 OpenAI 服务，请设置 `LLM_PROVIDER=generic` 并配置 `LLM_API_URL`，并根据厂商 API 调整 `src/generate_and_send.py` 中的请求解析（若必要）。
