# 架构与源码导航

当前公开版本支持的是 **PostgreSQL 多轮会话 + 固定语料构建 + 混合检索 + 需求覆盖评价**。主要启动入口为 `app.conversation_runtime`。部分早期模块因被测试或公共函数引用而保留，不代表所有历史 CLI 的数据都已包含。

## 1. 完整请求顺序

| 顺序 | 负责模块 | 做什么、为什么放在这里 |
| --- | --- | --- |
| 1 | [conversation_api.py](../app/conversation_api.py) | HTTP 身份与参数校验；身份来自服务端映射；生成请求编号并映射错误响应 |
| 2 | [conversation_service.py](../app/conversation_service.py) → [conversation_store.py](../app/conversation_store.py) | 保存用户原话、消息 ID 与输入版本；同 ID 同内容可恢复／回放，同 ID 不同内容拒绝 |
| 3 | [conversation_preparation.py](../app/conversation_preparation.py) | 逐条准备尚未处理的消息；识别新任务、补充、切换或指代不明 |
| 4 | [context_proposal.py](../app/context_proposal.py)、[analysis_scope.py](../app/analysis_scope.py) | 模型提出更新；校验目标、来源和条件，区分实际环境与假设 |
| 5 | [conversation_service.py](../app/conversation_service.py)、[build_runtime.py](../app/build_runtime.py) | 准备版本追上输入后，保存固定任务、输入快照和知识库 `build_id`，领取执行尝试 |
| 6 | [conversation_rag.py](../app/conversation_rag.py) | 由冻结快照构造本轮分析上下文，接入检索、评价与生成回调 |
| 7 | [scoped_retrieval.py](../app/scoped_retrieval.py)、[hybrid_retrieval.py](../app/hybrid_retrieval.py) | 同一语料版本内做精确余弦检索、BM25 与 RRF；过滤已有／不适用片段 |
| 8 | [requirement_evaluation.py](../app/requirement_evaluation.py)、[rag_service.py](../app/rag_service.py) | 评价需求覆盖；控制器决定有限补检、追问、说明不足或生成回答 |
| 9 | [answering.py](../app/answering.py) | 根据原问题、本轮上下文和最终证据组织回复，检查输出结构与引用 |
| 10 | [conversation_attempts.py](../app/conversation_attempts.py) | 短事务保存尝试结果并检查版本、活跃尝试及租约，决定是否成为当前结果 |
| 11 | [conversation_service.py](../app/conversation_service.py) → HTTP | 转换为对外响应，保留历史与当前结果的区分 |

模型／网络调用在数据库短事务之外进行。HTTP 路由为同步函数，避免把同步数据库与模型调用直接写进异步路由后误认为不会阻塞。

## 2. 为什么需要这些不同的标识

| 标识 | 含义 | 不能替代什么 |
| --- | --- | --- |
| `message_id` | 一条用户原话及其重试标识 | 有原话不代表状态准备或回答已经完成 |
| `task_id` | 可跨多轮继续的排错目标 | 一次成功追问不代表任务已解决 |
| `input_version` | 会话输入推进到哪一版 | 同一输入仍可能发生多次执行尝试 |
| `attempt_id` | 本轮运行的一次执行尝试 | 租约到期或被接管后不能发布当前结果 |
| `build_id` | 固定的正文、分块与嵌入模型版本 | 不能因最新语料发布而在一次运行中途切换 |

检查发布资格必须与更新处于同一个短事务中。只在应用层先检查再另开连接写入，会留下竞态窗口。

## 3. 检索和评价的边界

`RetrievalRequest` 位于 [retrieval_contract.py](../app/retrieval_contract.py)。检索器接受查询、固定版本、过滤条件及数量约束。控制器负责剩余候选容量与下一步动作，检索器不能擅自扩大预算或返回其他版本片段。

RRF 汇总不同检索路线的**排名贡献**，避免直接相加量纲不同的原始分数。排名靠前只是相关性信号，不保证覆盖全部需求。评价器检查覆盖情况；后端检查其结构、证据引用、数量与状态约束。结构校验无法证明所有语义判断正确。

缺少通用规则时可以补检；缺少当前故障的报错与日志时通常应追问。用户明确说明本次日志已不可取得时，保留具有任务范围和来源的约束，避免无效重复追问，但不能把另一故障的约束自动继承过来。

## 4. 存储与知识库版本

- PostgreSQL 保存会话原话、准备状态、任务、执行与尝试记录，以及知识库构建清单和活动版本指针。
- 公开演示语料为本地不可变文件，包含正文、来源元数据与已生成向量。当前版本不是把全部正文在线入库的文档管理平台。
- `load_build()` 校验文件 SHA-256、片段与向量一一对应、维数和模型身份。
- 新构建验证就绪后才可发布；已开始的请求继续使用自己的固定版本。

当前语料仅 53 个片段，用精确余弦便于核验。若扩大语料，需要另外评估 ANN 服务、索引发布与性能；这部分不冒充已实现。

## 5. 排错时从哪里查

| 现象 | 先检查 |
| --- | --- |
| 回答引用了另一个任务的环境 | 任务选择 → 冻结上下文 → 检索输入 |
| 把假设当作已发生 | 分析范围和原话来源 → 环境提取 → 生成提示词 |
| 回答缺一个关键点 | 候选中是否已有依据；有则查选证，没有则查召回／语料 |
| 引用合法但端口写错 | 最终证据 → 生成输入、提示词和输出；不能只检查引用 ID |
| 新消息后出现旧答案 | 输入版本 → 活跃尝试 → 租约 → 事务内发布条件 |
| 模型服务失败 | 请求编号和 provider 错误；不能伪装成“用户还需提供日志” |

相关离线测试包括 `test_requirement_evaluation.py`、`test_scoped_retrieval.py`、`test_build_pinning.py`；持久化测试集中在 `test_conversation_*.py`，运行方式见 [run.md](run.md)。
