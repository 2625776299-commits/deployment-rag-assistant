# 验证材料索引

| 文件 | 内容与适用范围 |
| --- | --- |
| [multiturn-dialogue.md](multiturn-dialogue.md) | E05 五轮原话、回答与保存状态，便于直接阅读 |
| [case-records.json](case-records.json) | 上述对话的结构化摘录、TX02 A/B 结果、本人参与设计与分析的示例 |
| [paired-review-20261105.json](paired-review-20261105.json) | 15 题、30 次执行的 AI 辅助逐题复核，含未完整覆盖的样本 |
| [multiturn-review-20261105.json](multiturn-review-20261105.json) | 10 场景、30 轮的 AI 辅助语义复核 |
| [contribution-analysis-CP01.json](contribution-analysis-CP01.json) | 区分初检重选与补检贡献，记录后续消融实验该固定哪些变量 |
| [historical-closeout.json](historical-closeout.json) | 2026-11-05 原项目测试与多轮运行摘要，仅提取必要结果字段 |
| [export-verification.json](export-verification.json) | 2026-11-12 公开子集检查：离线测试、哈希与边界说明 |
| [source-provenance.json](source-provenance.json) | 从原项目复制文件的 SHA-256 及公开版适配说明 |

`paired-review` 的记录注明其对应生成运行在后续终态回复修复之前，不能当成修复后重新跑出的结果。历史记录不随着源码快照变动而自动更新。

对话摘录仅将内部引用 ID 换成数字编号，中文状态概述是编辑说明。没有新增模型运行冒充旧记录，没有把 AI 复核称为用户独立逐题打分；也没有把结构检查通过率转写成回答准确率。
