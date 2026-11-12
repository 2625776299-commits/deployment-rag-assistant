# 查看记录与本地运行

## 1. 无需服务：先看保存的案例

在仓库根目录执行，两个脚本只读本地文件，不调用模型或数据库：

```bash
python scripts/show_recorded_case.py
python scripts/show_recorded_case.py --turn M2
python scripts/check_artifacts.py
```

第一个展示保存的 E05 对话；第二个选择某一轮；第三个核对固定版本、资料与向量哈希、记录的来源文件哈希。不将保存的对话当成新运行结果。

## 2. 安装与离线测试

已验证环境为 Windows、CPython 3.13.9。依赖版本来自实际项目环境，其他平台尚未单独验收。

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS／Linux 激活：

```bash
source .venv/bin/activate
```

安装并测试：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

不设置 `RAG_CONVERSATION_TEST_PORT` 时，90 项需要隔离 PostgreSQL 的测试会显示 `skipped`，其余使用本地资料或替身回调，不调用模型。公开版将一项依赖开发机绝对路径的旧迁移测试替换为当前随附构建的加载验证，其余选入的测试保持原样，详见 [source-provenance.json](../evidence/source-provenance.json)。

`requirements.txt` 支持当前冻结构建流程。早期 Chroma CLI 保留在依赖源码中，但历史数据未随附，也不是本次公开版运行入口。

## 3. 真实 HTTP 多轮服务

此入口会调用你配置的模型服务，需要本地 PostgreSQL、生成模型 API Key 与嵌入 API Key。正文向量已随附，不会每次用户提问都重新生成整个语料的向量。

1. 在 PostgreSQL 中准备一个专用演示数据库和可创建 schema 的用户。
2. 将 `.env.example` 复制为 `.env`，填写本地配置。`.env` 已被 Git 忽略。
3. 设置 `RAG_CONVERSATION_DATABASE_URL` 和一个自定义的 `RAG_CONVERSATION_TOKEN`。密码可通过本地 `PGPASSWORD` 或 `.pgpass` 提供。
4. 设置 `GENERATION_API_KEY`。若嵌入使用同一官方地址与凭据，`EMBEDDING_API_KEY` 可留空；否则单独填写。

PowerShell 复制配置：

```powershell
Copy-Item -LiteralPath .env.example -Destination .env
```

历史生成模型为 `gpt-6.1-sol`；若账号无权使用，应选择可用且支持当前请求格式的模型，并将新的结果视为新实验。固定语料对应 `text-embedding-3-large`、3072 维，查询必须使用相同嵌入模型与维数。不能仅修改嵌入模型名称就混用旧向量。

启动服务：

```bash
python -m app.conversation_runtime
```

服务绑定 `127.0.0.1:8000`。它会在该演示数据库中创建专用的会话与构建 schema，采用本地 Token 到演示身份的映射，不是完整的生产登录系统。

新开终端，在同一仓库与环境下：

```bash
python scripts/demo_client.py create
```

复制响应里的 `conversation_id`，替换下面的 `YOUR_CONVERSATION_ID`：

```bash
python scripts/demo_client.py send YOUR_CONVERSATION_ID --message-id M1 --text "现在应用在宿主机，数据库在同机Compose容器里，服务名db，映射5433:5432。当前连接地址怎么填？"
python scripts/demo_client.py send YOUR_CONVERSATION_ID --message-id M2 --text "我暂时还没迁移。如果以后把应用放进数据库所在的同一个Compose网络，连接地址会怎么变？"
python scripts/demo_client.py history YOUR_CONVERSATION_ID
```

如果未收到响应，保留原 `message_id`，先查询该消息状态：

```bash
python scripts/demo_client.py read YOUR_CONVERSATION_ID M2
```

原话已保存不代表处理完成。客户端不自动创建新 ID 重发。运行记录写入 `data/architecture_demos/`，该目录被 Git 忽略。

## 4. 固定问题集的实验入口

完成 `.env` 配置后，不带 `--live` 的预检只检查版本和输入，不发送模型请求：

```bash
python -m app.run_current_paired_comparison
```

需要实际复跑时显式指定 `--live`，例如仅运行 TX02：

```bash
python -m app.run_current_paired_comparison --live --cases TX02
```

该固定题集检查生成模型、服务地址与构建身份是否和原实验一致。若需要换模型，应复制并定义新的题集配置，避免把不同条件的结果混在同一报告里。

多轮 E05 的 HTTP／PostgreSQL 隔离验证使用本机 PostgreSQL 的 `initdb` 和 `pg_ctl`，会启动新的临时集群、选取本地端口，并在结束时停止。Windows 示例，按安装情况修改 `--pg-bin`：

```powershell
python -m app.run_conversation_http_probe --live --pg-bin 'C:\Program Files\PostgreSQL\18\bin' --suite data/evaluations/current_version_20261105/multiturn_cases.json --case-id E05 --release data/corpus_releases/deploy_faults_20261102_v1/release_openai-embedding-20261105T113923Z_3caf7a.json
```

该脚本需要配置真实模型；它不是离线数据库单元测试。完整多轮套件入口为 `app.run_multiturn_suite`，也需显式传入本仓库的 `--suite` 和 `--release`，默认旧套件未随附。

## 5. 本次公开范围

- 包含当前入口所需源码、41 个测试文件、官方资料片段、当前向量构建和开发题集。
- 旧构建清单中的 `baseline` 和迁移记录路径用于保留历史身份；旧向量与完整中间目录未发布。当前入口读取 `candidate`。
- 公开记录来自既有实验，不包含本地密钥、数据库文件、完整私有日志或全部学习资料。
- 原始题集的 `execution_status: not_executed` 是冻结定义时的字段，不在运行后改写；实际执行和复核结果见 `evidence/`。
